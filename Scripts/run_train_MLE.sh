python Trains/train_MLE.py --device='cuda:3' --eval_num_envs=25 \
--gamma=0.97 --lr_actor=0.00001 --lr_alpha=0.0003 --tau=0.005 --start_steps=5000 \
--log_freq=5000 --num_traj=25 --weight_path='./weight_MLE-test-1' 