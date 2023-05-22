
# ED-AIRL
python3 -m test ed_airl --env MazeED --env_designs "demo" --out_path ed_airl_maze_run1 --timelimit 250 --timelimit_noise 10 --max_iter_airl 45 --max_iter_multi_airl=15 --max_iter_rl=35 --n_iter 10 --subset_size 10
python3 -m test ed_airl --env MazeED --env_designs "demo" --out_path ed_airl_maze_run2 --timelimit 250 --timelimit_noise 10 --max_iter_airl 45 --max_iter_multi_airl=15 --max_iter_rl=35 --n_iter 10 --subset_size 10
python3 -m test ed_airl --env MazeED --env_designs "demo" --out_path ed_airl_maze_run3 --timelimit 250 --timelimit_noise 10 --max_iter_airl 45 --max_iter_multi_airl=15 --max_iter_rl=35 --n_iter 10 --subset_size 10
python3 -m test ed_airl --env MazeED --env_designs "demo" --out_path ed_airl_maze_run4 --timelimit 250 --timelimit_noise 10 --max_iter_airl 45 --max_iter_multi_airl=15 --max_iter_rl=35 --n_iter 10 --subset_size 10
python3 -m test ed_airl --env MazeED --env_designs "demo" --out_path ed_airl_maze_run5 --timelimit 250 --timelimit_noise 10 --max_iter_airl 45 --max_iter_multi_airl=15 --max_iter_rl=35 --n_iter 10 --subset_size 10

# Domain Randomization
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --timelimit 250 --timelimit_noise 10 --n_envs 10
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --timelimit 250 --timelimit_noise 10 --n_envs 10
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --timelimit 250 --timelimit_noise 10 --n_envs 10
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --timelimit 250 --timelimit_noise 10 --n_envs 10
python3 -m test multi_airl --env MazeED --env_config_name "demo" --max_iter 15 --timelimit 250 --timelimit_noise 10 --n_envs 10


# Eval ED-AIRL as we increase the amount of rounds
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED --name "domain_randomization_maze_demo" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v1 --name "domain_randomization_maze_demo" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v2 --name "domain_randomization_maze_demo" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v3 --name "domain_randomization_maze_demo" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v4 --name "domain_randomization_maze_demo" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED --name "domain_randomization_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v1 --name "domain_randomization_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v2 --name "domain_randomization_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v3 --name "domain_randomization_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/estimated_reward_functions/MULTI_AIRL/PPO_random_ed_MazeED_v4 --name "domain_randomization_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run1/single_env_reward_function --name "airl_maze_test" --max_iter 45 --irl_algo "AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run2/single_env_reward_function --name "airl_maze_test" --max_iter 45 --irl_algo "AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run3/single_env_reward_function --name "airl_maze_test" --max_iter 45 --irl_algo "AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run4/single_env_reward_function --name "airl_maze_test" --max_iter 45 --irl_algo "AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run5/single_env_reward_function --name "airl_maze_test" --max_iter 45 --irl_algo "AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set test --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_test" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10


python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run1/single_env_reward_function --name "ed_airl_maze_iter0" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_1/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter1" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_2/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter2" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_3/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter3" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter4" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_5/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter5" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_6/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter6" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_7/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter7" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_8/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter8" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run1/best_guess_reward_function --name "ed_airl_maze_iter9" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run2/single_env_reward_function --name "ed_airl_maze_iter0" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_1/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter1" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_2/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter2" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_3/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter3" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter4" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_5/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter5" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_6/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter6" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_7/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter7" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_8/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter8" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run2/best_guess_reward_function --name "ed_airl_maze_iter9" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run3/single_env_reward_function --name "ed_airl_maze_iter0" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_1/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter1" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_2/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter2" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_3/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter3" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter4" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_5/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter5" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_6/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter6" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_7/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter7" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_8/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter8" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run3/best_guess_reward_function --name "ed_airl_maze_iter9" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run4/single_env_reward_function --name "ed_airl_maze_iter0" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_1/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter1" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_2/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter2" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_3/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter3" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter4" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_5/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter5" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_6/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter6" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_7/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter7" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_8/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter8" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run4/best_guess_reward_function --name "ed_airl_maze_iter9" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_0/ed_airl_maze_run5/single_env_reward_function --name "ed_airl_maze_iter0" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_1/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter1" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_2/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter2" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_3/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter3" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_4/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter4" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_5/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter5" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_6/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter6" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_7/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter7" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_8/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter8" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env MazeED --env_set demo --from_path checkpoints/irl/ed_AIRL/iter_9/ed_airl_maze_run5/best_guess_reward_function --name "ed_airl_maze_iter9" --max_iter 45 --irl_algo "MULTI_AIRL"  --timelimit 250 --timelimit_noise 10

# AIRL

python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0 --name "airl_maze_run1_demo" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10

python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v1 --name "airl_maze_run1_demo" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v2 --name "airl_maze_run1_demo" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v3 --name "airl_maze_run1_demo" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v4 --name "airl_maze_run1_demo" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env_set test --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0 --name "airl_maze_run1_test" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set test --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v1 --name "airl_maze_run1_test" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set test --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v2 --name "airl_maze_run1_test" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set test --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v3 --name "airl_maze_run1_test" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set test --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v4 --name "airl_maze_run1_test" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10


python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 25
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 25
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 25
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 25
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 25

python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 30
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 30
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 30
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 30
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 30

python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 35
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 35
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 35
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 35
python3 -m main airl --max_iter 35 --env_config_name demo=0 --env MazeED --timelimit 250 --timelimit_noise 10 --n_expert_rollouts 35


python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v15 --name "airl_maze_run1_demo_25" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v16 --name "airl_maze_run1_demo_25" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v17 --name "airl_maze_run1_demo_25" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v18 --name "airl_maze_run1_demo_25" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v19 --name "airl_maze_run1_demo_25" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v20 --name "airl_maze_run1_demo_30" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v21 --name "airl_maze_run1_demo_30" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v22 --name "airl_maze_run1_demo_30" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v23 --name "airl_maze_run1_demo_30" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v24 --name "airl_maze_run1_demo_30" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10

python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v10 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v25 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v26 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v27 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v28 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
python3 -m test eval_on --env_set demo --env MazeED --from_path checkpoints/estimated_reward_functions/AIRL/PPO_MazeED_0_v29 --name "airl_maze_run1_demo_35" --max_iter 50 --irl_algo AIRL --timelimit 250 --timelimit_noise 10
