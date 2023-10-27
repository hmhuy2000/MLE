import numpy as np
import os
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from .base_algo import Algorithm
from Sources.networks.policy import StateIndependentPolicy

class BC(Algorithm):
    def __init__(self,buffer, state_shape, action_shape, device, seed, gamma,
                 hidden_units_actor,lr_actor,batch_size,max_grad_norm):
        super().__init__(device, seed, gamma)
        
        self.buffer = buffer
        self.actor = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_actor,
                hidden_activation=nn.Tanh(),
            ).to(device)
        
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
    def is_update(self, step):
        return True
    
    def update(self, log_info):
        self.learning_steps += 1
        exp_states, exp_actions, exp_total_rewards, exp_next_states, exp_dones = \
            self.buffer.sample_state_action(batch_size = self.batch_size)
        
        log_pi = self.actor.evaluate_log_pi(exp_states,exp_actions)
        loss = -log_pi.mean()
        self.optim_actor.zero_grad()
        loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        log_info.update({
            'log_pi':log_pi.mean().item(),
            'loss':loss.item(),
        })
        
    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def load_models(self,load_dir):
        if not os.path.exists(load_dir):
            raise
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth'))

    def copyNetworksFrom(self,algo):
        self.actor.load_state_dict(algo.actor.state_dict())

    def save_models(self,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')