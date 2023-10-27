python Trains/train_BC.py --device='cuda:3' --buffer_size=1 --eval_num_envs=25 \
--gamma=0.97 --lr_actor=0.00001 --lr_alpha=0.0003 --tau=0.005 --start_steps=0 \
--log_freq=5000 --num_traj=5 --weight_path='./weight_BC'