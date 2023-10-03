import os
import numpy as np
import torch

class Trajectory_Buffer:
    def __init__(self, buffer_size,traj_len, state_shape, action_shape, device):
        self._n             = 0
        self._p             = 0
        self.buffer_size    = buffer_size
        self.total_size     = buffer_size
        self.traj_len       = traj_len

        self.device = device
        self.states         = torch.empty(
            (self.total_size, traj_len, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, traj_len, *action_shape), dtype=torch.float, device=device)
        self.total_rewards  = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states    = torch.empty(
            (self.total_size, traj_len, *state_shape), dtype=torch.float, device=device)
        self.dones          = torch.empty(
            (self.total_size, traj_len, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action,total_reward, next_state, done):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.total_rewards[self._p]     = float(total_reward)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.dones[self._p].copy_(torch.from_numpy(done))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.total_rewards[idxes],
            self.next_states[idxes],
            self.dones[idxes],
        )
    
    def sample_state_action(self, batch_size):
        idxes_1 = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes_2 = np.random.randint(low=0, high=self.traj_len, size=batch_size)
        return (
            self.states[idxes_1,idxes_2],
            self.actions[idxes_1,idxes_2],
            self.total_rewards[idxes_1],
            self.next_states[idxes_1,idxes_2],
            self.dones[idxes_1,idxes_2],
        )
    
    def load(self,path):
        tmp = torch.load(path)
        self._n = tmp['states'].size(0)
        assert self.total_size<=self._n,'buffer too big'
        self.states             = tmp['states'][:self.total_size].clone().to(self.device)
        self.actions            = tmp['actions'][:self.total_size].clone().to(self.device)
        self.total_rewards      = tmp['total_rewards'][:self.total_size].clone().to(self.device)
        self.next_states        = tmp['next_states'][:self.total_size].clone().to(self.device)
        self.dones              = tmp['dones'][:self.total_size].clone().to(self.device)
        self._n = self.total_size

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'states': self.states.clone().cpu(),
            'actions': self.actions.clone().cpu(),
            'total_rewards': self.total_rewards.clone().cpu(),
            'next_states': self.next_states.clone().cpu(),
            'dones': self.dones.clone().cpu(),
        }, path)

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n             = 0
        self._p             = 0
        self.buffer_size    = buffer_size
        self.total_size     = buffer_size

        self.states         = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards        = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.total_rewards  = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis        = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states    = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward,total_reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p]           = float(reward)
        self.total_rewards[self._p]     = float(total_reward)
        self.dones[self._p]             = float(done)
        self.log_pis[self._p]           = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        start = 0
        idxes = slice(start, self._n)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )