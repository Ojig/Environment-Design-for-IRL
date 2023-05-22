# Single env AIRL

python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env AntED --timelimit_noise 50

python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0 --name "airl_ant_run1_demo" --max_iter 200 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v1 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v2 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v3 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v4 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v5 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v6 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v7 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v8 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v9 --name "airl_ant_run1_demo" --max_iter 150 --irl_algo AIRL

python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v1 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v2 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v3 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v4 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v5 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v6 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v7 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v8 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_AntED_0_v9 --name "airl_ant_run1_test" --max_iter 150 --irl_algo AIRL


# Domain randomization

python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220
python3 -m test multi_airl --env AntED --timelimit_noise 50 --env_config_name "demo" --max_iter 220

python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v1 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v2 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v3 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v4 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v5 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v6 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v7 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v8 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set demo --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v9 --name "domain_randomization_ant_run1_demo" --max_iter 150 --irl_algo MULTI_AIRL

python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v1 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v2 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v3 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v4 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v5 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v6 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v7 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v8 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL
python3 -m test eval_on --env_set test --env AntED --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_AntED_v9 --name "domain_randomization_ant_run1_test" --max_iter 150 --irl_algo MULTI_AIRL

# ED-AIRL


python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run1 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run2 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run3 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run4 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run5 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run6 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run7 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run8 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run9 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50
python3 -m test ed_airl --env AntED --env_designs "demo" --out_path edairl_ant_run10 --timelimit_noise 50 --max_iter_airl 220 --max_iter_multi_airl=220 --max_iter_rl=50

python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run1/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run2/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run3/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run4/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run5/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run6/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run7/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run8/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run9/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run10/best_guess_reward_function --name "ed_airl_ant_run1_demo" --max_iter 150 --irl_algo "MULTI_AIRL"

python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run1/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run2/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run3/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run4/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run5/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run6/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run7/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run8/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run9/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"
python3 -m test eval_on --env AntED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_ant_run10/best_guess_reward_function --name "ed_airl_ant_run1_test" --max_iter 150 --irl_algo "MULTI_AIRL"

