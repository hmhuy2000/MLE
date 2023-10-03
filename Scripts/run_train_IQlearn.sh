python Trains/train_IQlearn.py --device='cuda:0' --buffer_size=1 --eval_num_envs=25 \
--gamma=0.97 --lr_actor=0.00001 --lr_alpha=0.0003 --tau=0.005 --start_steps=10000 \
--log_freq=5000 --num_traj=1000