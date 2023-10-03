def main():
    from Sources.utils import d4rl_utils
    from Parameters.MLE import args,max_value
    
    gym_env, window_dataset = d4rl_utils.create_d4rl_env_and_dataset(
      task_name=[args.task_name] + args.expert_tasks,
      batch_size=args.batch_size,
      num_traj=[args.num_traj] + args.expert_trajs,
      sliding_window=args.sliding_window)
    
if __name__ == '__main__':
    main()