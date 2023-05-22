# Environment Design for Adversarial Inverse Reinforcement Learning

Implements the Environment Design for Adversarial Inverse Reinforcement Learning,
based on RLlib.

## Requirements
python version >= 3.8

gym==0.23.1
tensorflow==2.9.0
ray==2.0.0
imitation==0.3.2
mujoco==2.3.0

(_Tested on Ubuntu 20.04 and 22.04_)

### Tweaks for MuJoCo Environments

Tweaks for each environment can be found in the env_designs/.
Here is detailed how we alter the transition functions of each corresponding simulator.

## Usage

### Creating environment sets

The design config json files for a given environment are saved in env_design/{env}/

cf. **env_design/maze/demo.json** for an example template.

It can be generated randomly, accordingly to env_design/{env}/{env}ED.py:

```shell
    python -m main generate_designs --env="HopperED"
            # By default, sample each customizable parameters
            # if specified, only samples on given parameters
            --params_to_sample_from ["gravity","mass_02"],
            # How many samples do we want, the first sample is always the base environment
            --n_sample 20
            # The designs config file name, defaults with "designs"
            --name="designs"
```

##

### Generating experts and saving rollouts
We can generate expert policies for a specific environment:
```shell
    python -m main generate_experts --env="MazeED"
            # Rl algo to use, ex: "PPO", "TD3"
            --rl_algo algo_name
            # Name of the desired config file
            --designs_file file_name
```
The policies are saved in checkpoints/experts/ 

Their folder names are composed of the algo used, the environment name and the design id from the used file.

To save rollouts:
```shell
    python -m main save_rollouts --env="MazeED"
            # Rl algo to used to generate experts
            --rl_algo algo_name
            # Name of the desired config file
            --env_designs "demo"
            # How many rollouts to save
            --n_rollouts n_rollouts
            # Deploy the expert policy in deterministic mode ? Or allow sampling from the softmax ?
            --deterministic False  
```
The saved rollouts go to checkpoints/experts/rollouts/.

### AIRL

To estimate a reward function with AIRL, you first need to generate rollouts
from an expert on the desired environment.
Once this is done, you can run AIRL with:

```shell
    python -m main airl --env="MazeED"
            # Rl algo to train the generating policy
            --rl_algo algo_name
            # Design syntax -> "config_name=config_id"
            --design "demo=0"
            # Number of iterations to run AIRL for
            --max_iter 100
```

The reward function will be saved in checkpoints/estimated_reward_functions/AIRL/.

### AIRL-ME (AIRL with Multiple Environments)

For AIRL-ME, you have to generate beforehand rollouts from expert policies for every single
environment configuration of the chosen set.

```shell
    python -m main airl --env="MazeED"
            # Rl algo to train the generating policy
            --rl_algo algo_name
            # Environment set
            --env_designs "demo"
            # Number of iterations to run AIRL-ME for
            --max_iter 100
            # How many environments from the set can we use for expert demonstrations ?
            --n_envs 5
            # If None, randomly select environments as in Domain Ranomization AIRL,
            # else, a list of the desired environment indexes
            --environment_indexes None
```
The reward function will be saved in checkpoints/estimated_reward_functions/MULTI-AIRL/.


### ED-AIRL

Like AIRL-ME, you have to generate rollouts from expert policies for every single
environment configuration of the chosen set.

```shell
    python -m main ed_airl --env="MazeED"
            # Rl algo to train the generating policy
            --rl_algo algo_name
            # Environment set
            --env_designs "demo"
            # Number of iterations to run RL for (policy optimization steps in the extended value iteration algorithm)
            --max_iter_rl 100
            # Number of iterations to run AIRL for each point estimates
            --max_iter_airl 100
            # Number of iterations to run AIRL-ME to estimate the best guess each expert-learner round
            --max_iter_multi_airl 100
            # Number of expert-learner rounds, if n_iter=i, we end up with learning from i different environments for 
            # the best estimate
            --n_iter 5
            # How many environments do we sample from the demo set each time we want to select an environment
            --subset_size 5
            # Name of the run, generates folders in checkpoints/irl/ed_AIRL/.
            --out_path "ed_airl"
```
The reward function will be saved as checkpoints/irl/ed_AIRL/iter_{n_iter-1}/{out_path}/best_guess_reward_function.pkl.

### Evaluating a learned reward function

To evaluate a reward function over an environment configuration set, you can run (Operates a policy optimization once for 
each environment in the set, with the specified reward function):

```shell
    python -m test eval_on --env="MazeED"
            # Rl algo used to estimate the rewards, and algo to use to evaluate the rewards
            --rl_algo algo_name
            # IRL algo used to estimate the rewards
            --irl_algo irl_algo_name
            # Environment set
            --env_designs "demo"
            # Number of iterations to run RL before evaluation
            --max_iter 100
            # Path to the pickled reward function
            --from_path reward_path
            # Name of the evaluation folder (will contain a run folder for each environment in the set)
            --name "evaluation"
```

### Configuration and hyperparameters

Everything related to hyperparameters can be found in the config module.

#### Behavior Visualization

You can generate a gif of a trajectory for a policy for a given environment config with:

```shell
    python -m test visualize_policy --env="MazeED"
            # Rl algo that was used to train the expert
            --rl_algo algo_name
            # design syntax -> "config_name=config_id"
            --design design
            # Number of rollouts to visualize
            --n_rollouts n
            # Path to find the policy (RLlib run directory, of the form {rl_algo}_{env_id}_{run_id}), If None, automatically retrieves the expert for the specified environment 
            # configuration in checkpoints/experts/
            --full_path None
            # Name of the outputted gif
            --name "visualization"
```

### Monitoring

Training metrics can always be monitored with tensorboard, scalar data are saved in the checkpoints folder.