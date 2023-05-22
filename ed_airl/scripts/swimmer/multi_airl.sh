# Domain randomization

#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v1 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v2 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test multi_airl --env SwimmerED --timelimit_noise 200 --env_config_name "demo" --max_iter 35
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v3 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v4 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v5 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v6 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v7 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v8 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env_set demo --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v9 --name "domain_randomization_swimmer_run2_demo" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50
#
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v1 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v2 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v3 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v4 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v5 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v6 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v7 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v8 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]
#python3 -m test eval_on --env_set test --env SwimmerED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_SwimmerED_v9 --name "domain_randomization_swimmer_run2_test" --max_iter 300 --irl_algo MULTI_AIRL  --entropy_coeff 0.003 --timelimit_noise 50 --which [1,8,9]

# ED-AIRL


#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run1 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run2 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run3 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run4 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run5 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run6 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run7 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run8 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run9 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200
#python3 -m test ed_airl --env SwimmerED --env_designs "demo" --out_path edairl_swimmer_run20 --timelimit_noise 200 --max_iter_airl 55 --max_iter_multi_airl=35 --max_iter_rl=200

#python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run1/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run2/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
#python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run3/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run4/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run5/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run6/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run7/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run8/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run9/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run20/best_guess_reward_function --name "ed_airl_swimmer_run2_demo" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
#
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run1/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run2/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run3/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run4/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run5/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run6/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run7/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run8/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run9/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
python3 -m test eval_on --env SwimmerED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_swimmer_run20/best_guess_reward_function --name "ed_airl_swimmer_run2_test" --max_iter 300 --irl_algo "MULTI_AIRL" --entropy_coeff 0.003 --timelimit_noise 50
#
