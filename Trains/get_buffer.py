import sys
sys.path.append('..')
sys.path.append('./')
#------------------------------------------#
def main():
    from Sources.utils import VectorizedWrapper
    import gymnasium
    num_envs = 25
    env_name = 'HalfCheetah-v4'
    sample_env = gymnasium.make(id=env_name)
    env = [gymnasium.make(id=env_name) for _ in range(num_envs)]
    env = VectorizedWrapper(env)
    
    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    device = 'cuda'
    sample_env.close()

    #------------------------------------------#
    from Sources.networks.policy import StateIndependentPolicy
    from Sources.utils.buffers import Trajectory_Buffer
    from tqdm import trange
    import torch
    from torch import nn
    import numpy as np

    def exploit(actor,state):
        state = torch.tensor(state, dtype=torch.float, device=device)
        with torch.no_grad():
            action = actor(state)
        return action.cpu().numpy()

    def explore(actor,state):
        state = torch.tensor(state, dtype=torch.float, device=device)
        with torch.no_grad():
            (action,log_pi) = actor.sample(state)
        return action.cpu().numpy(),log_pi.cpu().numpy()

    expert_actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256,256],
            hidden_activation=nn.Tanh()
        ).to(device)
    expert_actor.load_state_dict(torch.load(
        # './weights/HalfCheetah-v4/SAC/(11233.639)-(11233.64)/actor.pth'
        # './weights/HalfCheetah-v4/SAC/(8892.292)-(8892.29)/actor.pth'
        # './weights/HalfCheetah-v4/SAC/(6346.880)-(6346.88)/actor.pth'
        './weights/HalfCheetah-v4/SAC/(3061.660)-(3061.66)/actor.pth'
    ))
    expert_actor.eval()
    buffer_size = 1000
    rollout_traj_buffer = Trajectory_Buffer(
        buffer_size=buffer_size,
        traj_len=1000, 
        state_shape=state_shape, 
        action_shape=action_shape, 
        device=device,
    )

    while(rollout_traj_buffer._n<buffer_size):
        state,_ = env.reset()
        episode_return = np.array([0.0 for _ in range(num_envs)])
        tmp_buffer = [[] for _ in range(num_envs)]

        for iter in range(1000):
            # action = exploit(expert_actor,state)
            action,log_pi = explore(expert_actor,state)
            next_state, reward, done, _, _ = env.step(action)
            for idx in range(num_envs):
                tmp_buffer[idx].append((state[idx], action[idx], next_state[idx]))
            episode_return += reward
            state = next_state
        
        for idx in range(num_envs):
            if (rollout_traj_buffer._n==buffer_size):
                break
            arr_states = []
            arr_actions = []
            arr_next_states = []
            arr_dones = []
            for (tmp_state,tmp_action,tmp_next_state) in tmp_buffer[idx]:
                arr_states.append(tmp_state)
                arr_actions.append(tmp_action)
                arr_next_states.append(tmp_next_state)
                arr_dones.append(0.0)
            arr_dones[-1] = 1.0
            arr_states = np.array(arr_states)
            arr_actions = np.array(arr_actions)
            arr_next_states = np.array(arr_next_states)
            arr_dones = np.array(arr_dones).reshape(-1,1)
            rollout_traj_buffer.append(arr_states,arr_actions,episode_return[idx], arr_next_states,
                                       arr_dones)
            
        print(f'{rollout_traj_buffer._n}/{buffer_size}, {rollout_traj_buffer.total_rewards[:rollout_traj_buffer._n].mean():.2f}, {rollout_traj_buffer.total_rewards[:rollout_traj_buffer._n].max():.2f}, '
              +f'{rollout_traj_buffer.total_rewards[:rollout_traj_buffer._n].min():.2f}, {rollout_traj_buffer.total_rewards[:rollout_traj_buffer._n].std():.2f}'
              ,end='\r')
            
    print(rollout_traj_buffer.total_rewards.mean().item(),rollout_traj_buffer.total_rewards.max().item(),rollout_traj_buffer.total_rewards.min().item(),rollout_traj_buffer.total_rewards.std().item())
    rollout_traj_buffer.save(f'./buffers/{env_name}/e3/{buffer_size}.pt')
    env.close()

if __name__ == '__main__':
    main()