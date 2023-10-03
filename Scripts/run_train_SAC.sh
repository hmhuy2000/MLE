python Trains/train_SAC.py \
--env_name='HalfCheetah-v4' --device='cuda:1' \
--number_layers=2 --hidden_units_actor=256 --hidden_units_critic=256 \
--gamma=0.97 --reward_factor=0.01 --lr_actor=0.00003