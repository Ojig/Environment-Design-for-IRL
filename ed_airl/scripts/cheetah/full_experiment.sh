# AIRL
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED
python3 -m main airl --max_iter 220 --env_config_name demo=0 --env CheetahED

python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v1 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v2 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v3 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v4 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v5 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v6 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v7 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v8 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v9 --name "airl_cheetah_run1_demo" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002

python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v1 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v2 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v3 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v4 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v5 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v6 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v7 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v8 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/AIRL/PPO_CheetahED_0_v9 --name "airl_cheetah_run1_test" --max_iter 320 --irl_algo "AIRL" --entropy_coeff 0.002 --timelimit_noise 200

# Domain Randomization

python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200
python3 -m test multi_airl --env CheetahED --env_config_name "demo" --max_iter 350 --n_expert_rollouts 50 --n_remember 2 --entropy_coeff 0.01 --lr 0.0005 --n_discriminator_train_step 8 --timelimit_noise 200

python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v1 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v2 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v3 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v4 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v5 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v6 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v7 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v8 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v9 --name "domain_randomization_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200

python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v1 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v2 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v3 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v4 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v5 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v6 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v7 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v8 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_CheetahED_v9 --name "domain_randomization_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200

# ED-AIRL

python3 -m test ed_airl --out_path edairl_cheetah_run1 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run2 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run3 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run4 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run5 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run6 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run7 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run8 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run9 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200
python3 -m test ed_airl --out_path edairl_cheetah_run10 --max_iter_airl 220 --max_iter_multi_airl 350 --max_iter_rl 220 --timelimit_noise 200

python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run1/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run2/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run3/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run4/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run5/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run6/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run7/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run8/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run9/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run10/best_guess_reward_function --name "ed_airl_cheetah_run1_demo" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200

python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run1/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run2/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run3/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run4/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run5/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run6/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run7/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run8/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run9/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200
python3 -m test eval_on --env CheetahED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_4/edairl_cheetah_run10/best_guess_reward_function --name "ed_airl_cheetah_run1_test" --max_iter 320 --irl_algo "MULTI_AIRL" --entropy_coeff 0.004 --timelimit_noise 200

