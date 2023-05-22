
# Care with ed-airl, fix all to max iter 250
python3 -m test multi_airl --env HopperED --env_config_name "demo" --max_iter 120 --deterministic_choice [0,1,2,3,4] --timelimit_noise 50

python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED --name "multi_airl_hopper" --max_iter 250 --irl_algo "MULTI_AIRL" --timelimit_noise 50
