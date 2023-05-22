
python3 -m test multi_airl --env AntED --env_config_name "demo" --max_iter 350 --deterministic_choice [0,1,2,3,4]

python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED --name "multi_airl_ant" --max_iter 320 --irl_algo "MULTI_AIRL"
