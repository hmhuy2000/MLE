import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.BC import *
#------------------------------------------#
def main():
    import gymnasium
    from Sources.utils import VectorizedWrapper

    sample_env = gymnasium.make(args.env_name)
    if (args.eval_num_envs):
        test_env = [gymnasium.make(id=args.env_name) for _ in range(args.eval_num_envs)]
        test_env = VectorizedWrapper(test_env)
    else:
        test_env = None

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    print(sample_env.observation_space,sample_env.action_space)
    sample_env.close()
    #------------------------------------------#
    from Sources.algos.BC import BC
    from Sources.utils.buffers import Trajectory_Buffer
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    from tqdm import trange
    import wandb
    #------------------------------------------#
    def evaluate(actor, env,max_episode_length):
        def exploit(actor,state):
            state = torch.tensor(state, dtype=torch.float, device=args.device)
            with torch.no_grad():
                action = actor(state)
            return action.cpu().numpy()
        global max_value
        mean_return = 0.0
        for step in range(args.num_eval_episodes//args.eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            for iter in range(max_episode_length):
                action = exploit(actor,state)
                state, reward, done, _, _ = env.step(action)
                episode_return += np.sum(reward*(1-done))
            mean_return += episode_return
        mean_return = mean_return/args.num_eval_episodes

        value = mean_return
        if (value>max_value):
            max_value = value
            algo.save_models(f'{args.weight_path}/({mean_return:.2f})')
        else:
            max_value*=0.999
        

        args.eval_return.write(f'{mean_return:.3f}\n')
        args.eval_return.flush()
        print(f'[Eval] R: {mean_return:.2f}, maxR: {max_value:.2f}')

    def train(test_env,algo,eval_actor):
        eval_thread = None
        print('start training')
        for step in range(1,args.num_training_step//args.num_envs+1):
            if (step%100 == 0):
                print(f'train: {step/(args.num_training_step//args.num_envs)*100:.2f}% {step}/{args.num_training_step//args.num_envs}', end='\r')
            if algo.is_update(step*args.num_envs):
                    log_info = {'log_cnt':(step*args.num_envs)//args.log_freq}
                    algo.update(log_info)
                    if (algo.learning_steps%args.log_freq == 0):
                        try:
                            wandb.log(log_info, step = log_info['log_cnt'])
                        except:
                            print(log_info)
                        algo.return_reward = []
                        algo.ep_len = []
                    
            if step % (args.eval_interval//args.num_envs) == 0:
                algo.save_models(f'{args.weight_path}/s{args.seed}-latest')
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                    eval_actor.load_state_dict(algo.actor.state_dict())
                    eval_actor.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_actor,test_env,args.max_episode_length))
                    eval_thread.start()
        algo.save_models(f'{args.weight_path}/s{args.seed}-finish')

    expert_buffer = Trajectory_Buffer(
        buffer_size=args.num_traj,
        traj_len=args.max_episode_length, 
        state_shape=state_shape, 
        action_shape=action_shape, 
        device=args.device,
    )
    expert_buffer.load('./buffers/HalfCheetah-v4/e0/1000.pt')
    print(f'load {args.num_traj} trajectories from {"./buffers/HalfCheetah-v4/e0/1000.pt"}'
        +f', max: {expert_buffer.total_rewards.max().item():.2f}'
        +f', min: {expert_buffer.total_rewards.min().item():.2f}'
        +f', mean: {expert_buffer.total_rewards.mean().item():.2f}'
        +f', std: {expert_buffer.total_rewards.std().item():.2f}'
        )

    setproctitle.setproctitle(f'{args.env_name}-BC-{args.seed}')
    algo = BC(buffer=expert_buffer,state_shape=state_shape,action_shape=action_shape,
              device=args.device,seed=args.seed,gamma=args.gamma,hidden_units_actor=args.hidden_units_actor,
              lr_actor=args.lr_actor,batch_size=args.SAC_batch_size,max_grad_norm=args.max_grad_norm)
    eval_actor = deepcopy(algo.actor)
    
    # wandb.init(project='offline-mujoco', settings=wandb.Settings(_disable_stats=True), \
    #     group=args.env_name,job_type=f'MLE{args.num_traj}', name=f'{args.seed}', entity='hmhuy',config=args)
    print(args)
    train(test_env=test_env,algo=algo,eval_actor=eval_actor)

    if (test_env):
        test_env.close()

if __name__ == '__main__':
    main()