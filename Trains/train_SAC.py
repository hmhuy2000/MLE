import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.SAC import *
#------------------------------------------#
def main():
    import gymnasium
    from Sources.utils import VectorizedWrapper

    sample_env = gymnasium.make(args.env_name)
    env = [gymnasium.make(id=args.env_name) for _ in range(args.num_envs)]
    env = VectorizedWrapper(env)
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
    from Sources.algos.SAC import SAC_continuous
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    import wandb
    #------------------------------------------#
    def evaluate(algo, env,max_episode_length):
        global max_value
        mean_return = 0.0

        for step in range(args.num_eval_episodes//args.eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            for iter in range(max_episode_length):
                # action = algo.exploit(state)
                action,_ = algo.explore(state)
                state, reward, done, _, _ = env.step(action)
                episode_return += np.sum(reward*(1-done))
            mean_return += episode_return
        mean_return = mean_return/args.num_eval_episodes

        value = mean_return
        if (value>max_value):
            max_value = value
        else:
            max_value*=0.999
        algo.save_models(f'{args.weight_path}/({value:.3f})-({mean_return:.2f})')

        print(f'[Eval] R: {mean_return:.2f}, '+
            f'V: {value:.2f}, maxV: {max_value:.2f}')

    def train(env,test_env,algo,eval_algo):
        t = np.array([0 for _ in range(args.num_envs)])
        eval_thread = None
        state,_ = env.reset()

        print('start training')
        for step in range(1,args.num_training_step//args.num_envs+1):
            if (step%100 == 0):
                print(f'train: {step/(args.num_training_step//args.num_envs)*100:.2f}% {step}/{args.num_training_step//args.num_envs}', end='\r')
            state, t = algo.step(env, state, t)
            if algo.is_update(step*args.num_envs):
                    log_info = {'log_cnt':(step*args.num_envs)//args.log_freq}
                    algo.update(log_info)
                    if ((step*args.num_envs)%args.log_freq == 0):
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
                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_algo,test_env,args.max_episode_length))
                    eval_thread.start()
        algo.save_models(f'{args.weight_path}/s{args.seed}-finish')

    setproctitle.setproctitle(f'{args.env_name}-SAC-{args.seed}')
    algo = SAC_continuous(
            state_shape=state_shape, action_shape=action_shape, device=args.device, seed=args.seed, gamma=args.gamma,
                 SAC_batch_size=args.SAC_batch_size, buffer_size=args.buffer_size, lr_actor=args.lr_actor, lr_critic=args.lr_critic, 
                 lr_alpha=args.lr_alpha, hidden_units_actor=args.hidden_units_actor, hidden_units_critic=args.hidden_units_critic, 
                 start_steps=args.start_steps,tau=args.tau,max_episode_length=args.max_episode_length, reward_factor=args.reward_factor,
                 max_grad_norm=args.max_grad_norm)
    eval_algo = deepcopy(algo)
    
    # wandb.init(project=f'test-offline-RL', settings=wandb.Settings(_disable_stats=True), \
    #     group=args.env_name,job_type='SAC', name=f'{args.seed}', entity='hmhuy',config=args)
    print(args)
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

if __name__ == '__main__':
    main()