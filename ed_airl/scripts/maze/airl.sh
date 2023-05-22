
python3 -m main airl --max_iter 45 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0 --name "airl_maze" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
