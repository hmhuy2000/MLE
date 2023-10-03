"""Loads D4RL dataset from pickle files."""
import typing

import d4rl
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def qlearning_and_window_dataset(env, sliding_window=1,
                                 dataset=None, terminate_on_end=False,
                                 num_traj=None, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    window_obs_ = []
    window_next_obs_ = []
    window_action_ = []
    window_reward_ = []
    window_done_ = []
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    episode_return = 0
    sliding_obs = []
    sliding_act = []
    sliding_reward = []
    sliding_done = []
    episode_start = True
    arr_returns, arr_lens = [],[]
    traj_cnt = 0
    for i in range(N-1):
        if (num_traj is not None and traj_cnt==num_traj):
            break
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])

        if episode_start:
            sliding_obs = [obs] * sliding_window
            sliding_act = [0 * action] * sliding_window
            sliding_reward = [0 * reward] * sliding_window
            sliding_done = [-1.] * sliding_window  # -1 for 'before start'.

        sliding_obs.append(obs)
        sliding_act.append(action)
        sliding_reward.append(reward)
        sliding_done.append(done_bool)

        sliding_obs.pop(0)
        sliding_act.pop(0)
        sliding_done.pop(0)
        sliding_reward.pop(0)

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            arr_returns.append(episode_return)
            arr_lens.append(episode_step)
            episode_step = 0
            episode_return = 0
            episode_start = True
            traj_cnt += 1
            continue
        if done_bool or final_timestep:
            arr_returns.append(episode_return)
            arr_lens.append(episode_step)
            episode_step = 0
            episode_return = 0
            episode_start = True
            traj_cnt += 1
        else:
            episode_start = False
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        window_obs_.append(sliding_obs[:])
        window_next_obs_.append(sliding_obs[1:] + [new_obs])
        window_action_.append(sliding_act[:])
        window_reward_.append(sliding_reward[:])
        window_done_.append(sliding_done[:])

        episode_step += 1
        episode_return += reward
    
    print(np.mean(arr_lens),np.mean(arr_returns))
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }, {
        'observations': np.array(window_obs_),
        'actions': np.array(window_action_),
        'next_observations': np.array(window_next_obs_),
        'rewards': np.array(window_reward_),
        'terminals': np.array(window_done_),
    }

class EnvDataset(Dataset):
    def __init__(self, states,actions,rewards,discounts,
                 next_states, labels):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.discounts = discounts
        self.next_states = next_states
        self.labels = labels

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index],self.actions[index],\
            self.rewards[index],self.discounts[index],\
            self.next_states[index], self.labels[index]

def create_d4rl_env_and_dataset(
    task_name,
    batch_size,
    sliding_window = None,
    num_traj = None,
    state_mask_fn = None,
    shuffle=True,
):
    expert_env = None
    for id,(task,traj) in enumerate(zip(task_name,num_traj)):
        print('-'*10, task,traj, '-'*10)
        if (expert_env is None):
            expert_env = gym.make(task)
        env = gym.make(task)
        _, window_dataset = qlearning_and_window_dataset(
            env, sliding_window=sliding_window or 1,num_traj=traj)

        if id == 0:
            window_states = window_dataset['observations']
            window_actions = window_dataset['actions']
            window_rewards = window_dataset['rewards']
            window_discounts = np.logical_not(window_dataset['terminals'])
            window_next_states = window_dataset['next_observations']
            window_labels = np.full(window_dataset['rewards'].shape,id,dtype=np.int32)
        else:
            window_states = np.concatenate((window_states,window_dataset['observations']),axis=0)
            window_actions = np.concatenate((window_actions,window_dataset['actions']),axis=0)
            window_rewards = np.concatenate((window_rewards,window_dataset['rewards']),axis=0)
            window_discounts = np.concatenate((window_discounts,np.logical_not(window_dataset['terminals'])),axis=0)
            window_next_states = np.concatenate((window_next_states,window_dataset['next_observations']),axis=0)
            window_labels = np.concatenate((window_labels,np.full(window_dataset['rewards'].shape,id,dtype=np.int32)),axis=0)

    window_states = torch.tensor(window_states)
    window_actions = torch.tensor(window_actions)
    window_rewards = torch.tensor(window_rewards)
    window_discounts = torch.tensor(window_discounts)
    window_next_states = torch.tensor(window_next_states)
    window_labels = torch.tensor(window_labels)
    

    window_dataset = EnvDataset(states=window_states,actions=window_actions,rewards=window_rewards,
                         discounts=window_discounts,next_states=window_next_states,
                         labels=window_labels)

    window_data_loader = DataLoader(window_dataset, batch_size=batch_size, shuffle=shuffle)
    return expert_env,window_data_loader


# if id == 0:
#     states = dataset['observations']
#     actions = dataset['actions']
#     rewards = dataset['rewards']
#     discounts = np.logical_not(dataset['terminals'])
#     next_states = dataset['next_observations']
#     labels = np.full(dataset['rewards'].shape,id,dtype=np.int32)
# else:
#     states = np.concatenate((states,dataset['observations']),axis=0)
#     actions = np.concatenate((actions,dataset['actions']),axis=0)
#     rewards = np.concatenate((rewards,dataset['rewards']),axis=0)
#     discounts = np.concatenate((discounts,np.logical_not(dataset['terminals'])),axis=0)
#     next_states = np.concatenate((next_states,dataset['next_observations']),axis=0)
#     labels = np.concatenate((labels,np.full(dataset['rewards'].shape,id,dtype=np.int32)),axis=0)
    
# states = torch.tensor(states)
# actions = torch.tensor(actions)
# rewards = torch.tensor(rewards)
# discounts = torch.tensor(discounts)
# next_states = torch.tensor(next_states)
# labels = torch.tensor(labels)

# dataset = EnvDataset(states=states,actions=actions,rewards=rewards,
#                         discounts=discounts,next_states=next_states,
#                         labels=labels)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
