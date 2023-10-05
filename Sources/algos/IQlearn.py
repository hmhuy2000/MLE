from .SAC import *
from Sources.networks.value import StateActionFunction

class IQlearn(SAC_continuous):
    def __init__(self, expert_dataset,suplement_buffers, state_shape, action_shape, device, seed, gamma,
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
        self.suplement_buffers = suplement_buffers
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
        exp_states, exp_actions, exp_total_rewards, exp_next_states, exp_dones = \
            self.expert_dataset.sample_state_action(batch_size = self.SAC_batch_size)
        self.update_critic(
            exp_states, exp_actions, exp_next_states,exp_dones, log_info)
        self.update_actor(exp_states, log_info)
        self.update_target()
        
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
    
    def update_critic(self, exp_states, exp_actions, exp_next_states,exp_dones, log_info):
        current_qs = self.critic(exp_states, exp_actions)
        y = (1.0 - exp_dones)*self.gamma*self.get_targetV(exp_next_states)
        
        rewards = (current_qs - y)
        reward_loss = -rewards.mean() + 1/4*(rewards**2).mean()
        
        value_dif = self.getV(exp_states) - y
        value_loss = value_dif.mean() + 1/32*(value_dif**2).mean()
        
        total_loss = reward_loss + value_loss
        self.optim_critic.zero_grad()
        total_loss.backward()
        self.optim_critic.step()
        
        if (self.learning_steps%self.args.log_freq==0):
            with torch.no_grad():
                exp_log_prob = self.actor.evaluate_log_pi(exp_states,exp_actions)
                _,pi_log_prob = self.actor.sample(exp_states)
                for id,buffer in enumerate(self.suplement_buffers):
                    buffer_states, buffer_actions, _, _, _ = \
                        buffer.sample_state_action(batch_size = self.SAC_batch_size)
                    buffer_log_prob = self.actor.evaluate_log_pi(buffer_states,buffer_actions)
                    log_info.update({
                        f'log_prob/buffer_{id}_log_prob':buffer_log_prob.mean().item(),
                    })
                    
            log_info.update({
                'log_prob/exp_log_prob':exp_log_prob.mean().item(),
                'log_prob/pi_log_prob':pi_log_prob.mean().item(),
            })
                            
        log_info.update({
            'reward_loss':reward_loss.item(),
            'value_loss':value_loss.item(),
            'total_loss':total_loss.item(),
            'reward':rewards.mean().item(),
            'value_dif':value_dif.mean().item(),
            
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

        log_info.update({
            'Loss/actor_loss':loss_actor.item(),
            'Loss/entropy_loss':self.alpha * log_pis.mean().item(),
            'Update/entropy':entropy.item(),
            'Update/alpha':self.alpha,
            'Update/log_alpha':self.log_alpha.item(),
            'Update/log_pis':log_pis.mean().item(),
        })