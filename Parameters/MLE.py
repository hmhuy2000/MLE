import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('SAC_training')
training_group.add_argument('--task_name',type=str,default='ant-expert-v2')
training_group.add_argument('--num_traj',type=int,default=10)

training_group.add_argument('--device', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=0)
training_group.add_argument('--hidden_units_actor',type=int,default=256)
training_group.add_argument('--number_layers',type=int,default=3)
training_group.add_argument('--lr_actor', type=float, default=0.0001)

training_group.add_argument('--sliding_window',type=int,default=5)
training_group.add_argument('--batch_size',type=int,default=256)

training_group.add_argument('--max_grad_norm', type=float, default=1.0)
training_group.add_argument('--num_training_step',type=int,default=int(1e6))
training_group.add_argument('--eval_interval',type=int,default=int(1e4))
training_group.add_argument('--num_eval_episodes',type=int,default=100)
training_group.add_argument('--max_episode_length',type=int,default=1000)
training_group.add_argument('--reward_factor',type=float,default=1.0)
training_group.add_argument('--log_freq',type=int,default=int(1e4))
training_group.add_argument('--weight_path', type=str, default='./weights')
training_group.add_argument('--eval_num_envs',type=int,default=25)

training_group.add_argument('--expert_tasks',default=[], action='append')
training_group.add_argument('--expert_trajs',default=[], action='append', type=int)

#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()
hidden_units_actor                      = []
for _ in range(args.number_layers):
    hidden_units_actor.append(args.hidden_units_actor)

args.hidden_units_actor                 = hidden_units_actor
max_value                               = -np.inf

args.weight_path = os.path.join(args.weight_path,args.task_name,'MLE')
args.log_path = f'{args.weight_path}/log_data'
os.makedirs(args.log_path,exist_ok=True)
