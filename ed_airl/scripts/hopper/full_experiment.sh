# Single env AIRL

python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50
python3 -m main airl --max_iter 120 --env_config_name demo=0 --env HopperED --timelimit_noise 50

python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v1 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v2 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v3 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v4 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v5 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v6 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v7 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v8 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v9 --name "airl_hopper_run1_demo" --max_iter 200 --irl_algo AIRL --timelimit_noise 50

python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v1 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v2 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v3 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v4 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v5 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v6 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v7 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v8 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_HopperED_0_v9 --name "airl_hopper_run1_test" --max_iter 200 --irl_algo AIRL --timelimit_noise 50


# Domain randomization

python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120
python3 -m test multi_airl --env HopperED --timelimit_noise 50 --env_config_name "demo" --max_iter 120

python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v1 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v2 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v3 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v4 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v5 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v6 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v7 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v8 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set demo --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v9 --name "domain_randomization_hopper_run1_demo" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50

python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v1 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v2 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v3 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v4 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v5 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v6 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v7 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v8 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50
python3 -m test eval_on --env_set test --env HopperED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_HopperED_v9 --name "domain_randomization_hopper_run1_test" --max_iter 200 --irl_algo MULTI_AIRL --timelimit_noise 50

# ED-AIRL

python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run1 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run2 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run3 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run4 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run5 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run6 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run7 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run8 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run9 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200
python3 -m test ed_airl --env HopperED --env_designs "demo" --out_path edairl_hopper_run10 --timelimit_noise 50 --max_iter_airl 120 --max_iter_multi_airl=120 --max_iter_rl=200

python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run1/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run2/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run3/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run4/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run5/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run6/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run7/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run8/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run9/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run10/best_guess_reward_function --name "ed_airl_hopper_run1_demo" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50

python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run1/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run2/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run3/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run4/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run5/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run6/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run7/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run8/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run9/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50
python3 -m test eval_on --env HopperED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_hopper_run10/best_guess_reward_function --name "ed_airl_hopper_run1_test" --max_iter 200 --irl_algo "MULTI_AIRL" --timelimit_noise 50

