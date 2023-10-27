python Trains/train_IQlearn.py --device='cuda:1' --eval_num_envs=25 \
--gamma=0.97 --lr_actor=0.00001 --lr_alpha=0.0003 --tau=0.005 --start_steps=5000 \
--log_freq=5000 --num_traj=100 --weight_path='./weight_IQ_e0' 