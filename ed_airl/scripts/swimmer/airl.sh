#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0] --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v1 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0] --timelimit_noise 50

#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50
#python3 -m main airl --max_iter 55 --env_config_name demo=0 --env SwimmerED --timelimit_noise 50


#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v2 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v3 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v4 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v5 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v6 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v7 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v8 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v9 --name "airl_swimmer_run2_demo" --max_iter 300 --irl_algo AIRL --which [0,14,15,16,17,18] --timelimit_noise 50

python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v1 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v2 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v3 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v4 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v5 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v6 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v7 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v8 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_SwimmerED_0_v9 --name "airl_swimmer_run2_test" --max_iter 300 --irl_algo AIRL --which [1,8,9] --timelimit_noise 50
