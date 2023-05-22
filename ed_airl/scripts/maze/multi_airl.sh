
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --deterministic_choice [0,9,12,8,18] --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED --name "multi_airl_maze" --max_iter 50 --irl_algo "MULTI_AIRL" --timelimit 250 --timelimit_noise 10
