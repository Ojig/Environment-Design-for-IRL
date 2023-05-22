
#python3 -m test multi_airl --env WalkerED --env_config_name "demo" --max_iter 350 --deterministic_choice [0,1,3,5,8]

python3 -m test eval_on --env WalkerED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_WalkerED --name "multi_airl_walker" --max_iter 320 --irl_algo "MULTI_AIRL"
