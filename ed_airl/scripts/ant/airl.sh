
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED

python3 -m test eval_reward_function --env_designs demo --design_id 0 --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0
