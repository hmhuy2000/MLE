import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import build_mlp, reparameterize, evaluate_lop_pi

class StateIndependentPolicy(nn.Module):
    
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),add_dim=0):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0]+add_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)
    
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        # return super().log_prob(actions)
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean
    
class new_StateIndependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),std_x_coef=1.0,std_y_coef=0.5,add_dim=0):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0]+add_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.std_x_coef = std_x_coef
        self.std_y_coef = std_y_coef
        self.log_stds = nn.Parameter(torch.ones(1, action_shape[0])*self.std_x_coef)
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        mean = self.forward(states)
        std = torch.sigmoid(self.log_stds/self.std_x_coef)*self.std_y_coef
        dist =  FixedNormal(mean,std)
        action = dist.sample()
        return action, dist.log_probs(action)

    def evaluate_log_pi(self, states, actions):
        mean = self.forward(states)
        std = torch.sigmoid(self.log_stds/self.std_x_coef)*self.std_y_coef
        dist =  FixedNormal(mean,std)
        
        return dist.log_probs(actions)