from .SAC import *
from Sources.networks.value import StateActionFunction

class MLE(SAC_continuous):
    def __init__(self, expert_dataset,add_dataset, state_shape, action_shape, device, seed, gamma,
            SAC_batch_size, buffer_size, lr_actor, lr_critic, 
            lr_alpha, hidden_units_actor, hidden_units_critic, 
            start_steps, tau,max_episode_length,reward_factor,
            max_grad_norm,args, primarive=True):
        super().__init__(state_shape, action_shape, device, seed, gamma,
            SAC_batch_size, buffer_size, lr_actor, lr_critic, 
            lr_alpha, hidden_units_actor, hidden_units_critic, 
            start_steps, tau,max_episode_length,reward_factor,
            max_grad_norm,primarive=False)
        self.expert_dataset = expert_dataset
        self.add_dataset = add_dataset
        self.args  = args
        if (primarive):
            # Actor.
            self.actor = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_actor,
                hidden_activation=nn.Tanh(),
            ).to(device)

            # Critic.
            self.critic = StateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.critic_target = StateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
            
            self.buffer = RolloutBuffer(
                buffer_size=buffer_size,
                state_shape=state_shape,
                action_shape=action_shape,
                device=device,
            )
            soft_update(self.critic_target, self.critic, 1.0)
            disable_gradient(self.critic_target)
            self.alpha = 1.0
            self.log_alpha = torch.zeros(1, device=device, requires_grad=True)

            self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
            self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
    def is_update(self, steps):
        return steps>self.start_steps
    
    def update(self, log_info):
        self.learning_steps += 1
        pi_states, pi_actions, env_rewards,_, pi_dones, pi_log_pis, pi_next_states =\
              self.buffer.sample(self.SAC_batch_size)
        add_states, add_actions, _, add_next_states, add_dones = \
            self.expert_dataset.sample_state_action(batch_size = self.SAC_batch_size)
        exp_states, exp_actions, _, exp_next_states, exp_dones = \
            self.expert_dataset.sample_state_action(batch_size = self.SAC_batch_size)
        self.update_critic( pi_states, pi_actions,pi_next_states,pi_dones,
                            add_states, add_actions, add_next_states,add_dones,
                            exp_states, exp_actions, exp_next_states,exp_dones, log_info)
        self.update_actor(torch.cat((exp_states,add_states,pi_states),dim=0), log_info)
        self.update_target()
        log_info.update({
            'Train/return':np.mean(self.return_reward),
            'Train/epLen':np.mean(self.ep_len),
            'Train/num_traj':len(self.return_reward),
            'Update/env_rewards':env_rewards.mean().item(),
        })
        
    def get_targetV(self, states):
        with torch.no_grad():
            action, log_prob = self.actor.sample(states)
            curr_qs = self.critic_target(states, action)
            target_V = curr_qs - self.alpha * log_prob
            return target_V
        
    def getV(self, states):
        action, log_prob = self.actor.sample(states)
        curr_qs = self.critic(states, action)
        current_V = curr_qs - self.alpha * log_prob
        return current_V
    
    def update_critic(self, pi_states, pi_actions, pi_next_states,pi_dones, 
                            add_states, add_actions, add_next_states,add_dones, 
                            exp_states, exp_actions, exp_next_states,exp_dones, log_info):
        exp_dones = exp_dones.mean(-1,keepdim=True)
        add_dones = add_dones.mean(-1,keepdim=True)
        assert exp_dones.mean() <= 1.0 and exp_dones.mean() >= 0.0 and exp_dones.min() >= 0.0 and exp_dones.max() <= 1.0
        assert add_dones.mean() <= 1.0 and add_dones.mean() >= 0.0 and add_dones.min() >= 0.0 and add_dones.max() <= 1.0
        
        exp_current_qs = self.critic(exp_states, exp_actions)
        exp_y = (1.0 - exp_dones)*self.gamma*self.get_targetV(exp_next_states)
        add_current_qs = self.critic(add_states,add_actions)
        add_y = (1.0 - add_dones)*self.gamma*self.get_targetV(add_next_states)
        pi_current_qs = self.critic(pi_states, pi_actions)
        pi_y = (1.0 - pi_dones)*self.gamma*self.get_targetV(pi_next_states)

        pi_rewards = (pi_current_qs - pi_y)
        
        exp_rewards = (exp_current_qs - exp_y)
        exp_reward_loss = -exp_rewards.mean() 
        
        add_rewards = (add_current_qs - add_y)
        add_reward_loss = -add_rewards.mean()
        
        chi2_loss = 1/4 * (exp_rewards**2).mean() + 1/2 * (torch.cat((add_rewards,pi_rewards),dim=0)**2).mean()
        
        exp_value_dif = self.getV(exp_states) - exp_y
        add_value_dif = self.getV(add_states) - add_y
        pi_value_dif = self.getV(pi_states) - pi_y
        value_dif = torch.cat((exp_value_dif,add_value_dif,pi_value_dif),dim=0)
        value_loss = value_dif.mean() + 1/8 * (value_dif**2).mean()
        
        # positive_loss = torch.exp(-torch.cat((exp_current_qs,add_current_qs),dim=0)).mean()
        # positive_loss = torch.exp(-exp_current_qs).mean()
        
        total_loss = exp_reward_loss + add_reward_loss + chi2_loss + value_loss# + positive_loss
        self.optim_critic.zero_grad()
        total_loss.backward()
        self.optim_critic.step()
              
        log_info.update({
            'reward_loss':exp_reward_loss.item(),
            'value_loss':value_loss.item(),
            'total_loss':total_loss.item(),
            
            'expert_Q':exp_current_qs.mean().item(),
            'add_Q':add_current_qs.mean().item(),
            'pi_Q':pi_current_qs.mean().item(),
            
            'exp_reward':exp_rewards.mean().item(),
            'add_reward':add_rewards.mean().item(),
            'pi_reward':pi_rewards.mean().item(),

        })
        
    def update_actor(self, states, log_info):
        actions, log_pis = self.actor.sample(states)
        qs = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - qs.mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        
        entropy = -log_pis.detach_().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)
        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()
        log_info['Update/log_alpha'] = self.log_alpha.item()

        log_info.update({
            'Loss/actor_loss':loss_actor.item(),
            'Loss/entropy_loss':self.alpha * log_pis.mean().item(),
            'Update/entropy':entropy.item(),
            'Update/alpha':self.alpha,
            'Update/log_pis':log_pis.mean().item(),
        })