
python3 -m main airl --max_iter 200 --env_config_name demo=0 --env HopperED --timelimit_noise 50

python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0 --name "airl_hopper" --max_iter 250 --irl_algo AIRL --timelimit_noise 50
