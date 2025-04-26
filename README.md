# COMP3071-Design Intelligent Agents / Coursework 2 / Part 1 (30%)

## Introduction

This is COMP3071 Design Intelligent Agents Readme file, hope following information can help you understand our project.

This project use SUMO as simulation environment, choose DDPG(NextDurationRLTSC) as RL agent for each intersection, training mode is CTDE(Centralized Training with Decentralized Execution).

Project use Traci interfact dynamic generate vehicles, there are two generation mode:
1. single: when there is no car in the whole environment, system will generate one, this mode is used to test whether the SUMO env is usable.
2. dynamic: Generate vehicles in a sine wave pattern, starting with fewer, then more, and then fewer again. This is main generation mode used for training and testing.

## Install

<!-- start install -->

### Install SUMO latest version:

1. Install latest SUMO

2. Don't forget to set your "SUMO_HOME"

### IDE:

- Python 3.7.17
- Tensorflow 1.14.0

You can use miniconda to set a Virtual Environment

```bash
conda create -n your_env_name python=3.7.16
```
Activate your VE

```bash
conda activate your_env_name
```

Install Tensorflow 1.14.0 and matplotlib==3.1.1

```bash
pip install tensorflow==1.14.0
pip install matplotlib==3.1.1
```

## Instruction

### Running Command
There are four .sh files:
- train_nsgim_ddpg.sh: Training on NGSIM env command
- test_ngsim_ddpg.sh: Testing on NGSIM env using trained ddpg command
- test_thrnet_ddpg.sh: Testing on THRNET env using trained ddpg command
- test_with_uniform: Testing on NGSIM using uniform method command

You can copy these command to terminal to run the project.

### Files

There are already have trained ddpg model under save_models folder.

You can use test command to test on the net.xml file you want and use this command to generate the results plot.
```bash
python graph_results.py
```

## Parameters of Command
Multi-process parameters:
- n: The number of simulation processes for generating experience in parallel simulation, default value: number of CPU cores - 1
- l: The number of parallel learning processes for generating updates, default value: 1

SUMO simulation parameters:
- sim: simulation scenario, default: lust, options: lust, single, double
- port: port to connect to the server, default: 9000
- netfp: path to the required simulation network file, default: networks/double.net.xml
- sumocfg: path to the required simulation configuration file, default: networks/double.sumocfg
- mode: reinforcement learning mode, training (agents receive updates) or testing (no updates), default: train, options: train, test
- tsc: Traffic signal control algorithm, default: websters, options: sotl, maxpressure, dqn, ddpg
- simlen: Simulation length (seconds/step), default: 10800
- nogui: Disable GUI interface, default: False
- scale: Vehicle generation scale parameter, higher value generates more vehicles, default: 1.4
- demand: Vehicle demand generation mode, single limits the number of network vehicles to 1, dynamic creates a variable number of vehicles, default: dynamic, options: single, dynamic
- offset: The maximum simulation offset as a proportion of the total simulation length, default: 0.25

Traffic signal control shared parameters:
- gmin: minimum green light phase time (seconds), default: 5
- y: yellow light change phase time (seconds), default: 2
- r: all red stop phase time (seconds), default: 3

Reinforcement learning parameters:
- eps: reinforcement learning exploration rate, default: 0.01
- nsteps: n-step reward/maximum experience trajectory, default: 1
- nreplay: maximum size of experience replay, default: 10000
- batch: batch size for training neural network from replay, default: 32
- gamma: reward discount factor, default: 0.99
- updates: total number of batch updates for training, default: 10000
- target_freq: target network batch update frequency, default: 50

Neural network parameters:
- lr: DDPG actor/DQN neural network learning rate, default: 0.0001
- lrc: DDPG critic neural network learning rate, default: 0.001
- lre: neural network optimizer epsilon, default: 0.00000001
- hidden_act: neural network hidden layer activation function, default: elu
- n_hidden: neural network hidden layer scaling factor, default: 3
- save_path: directory to save neural network weights, default: saved_models
- save_replay: directory to save experience replays, default: saved_replays
- load_replay: load experience replays if they exist, default: False
- save_t: interval (seconds) between saves of neural network by the learner, default: 120
- save: use this parameter to save neural network weights, default: False
- load: use this parameter to load existing neural network weights, default: False

DDPG specific parameters
- tau: DDPG online/target weight transfer parameter tau, default: 0.005
- gmax: maximum green phase time (seconds), default: 30

Most of them are not used in real training, because use default value is enough, unless you have some special requirement, or you use the command in the four .sh files that introduced above is enough.
## Results

Training Environment: NGSIM.net.xml

[pic] 

Additional Testing Environment: THRNET.net.xml

[pic]

There are two method testing on Training Environment:
- RL method: DDPG
- Traditional method: uniform

The traditional method is used for comparision with DDPG to demostrate the training outcomes of DDPG.

There is only one method test on THRNET.net.xml
- RL method: DDPG

Use the DDPG trained by NGSIM.net.xml to test on THRNET.net.xml to confirm that the trained DDPG is adaptative.


### Testing Results on NGSIM.net.xml

[pic]

### Testing Results on THRNET

[pic]

<!-- start observation -->

The default observation for each traffic signal agent is a vector:
```python
    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
```
- ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
- ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
- ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
- ```lane_i_queue```is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

You can define your own observation by implementing a class that inherits from [ObservationFunction](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py) and passing it to the environment constructor.

<!-- end observation -->

### Action

<!-- start action -->

The action space is discrete.
Every 'delta_time' seconds, each traffic signal agent can choose the next green phase configuration.

E.g.: In the [2-way single intersection](https://github.com/LucasAlegre/sumo-rl/blob/main/experiments/dqn_2way-single-intersection.py) there are |A| = 4 discrete actions, corresponding to the following green phase configurations:

<p align="center">
<img src="docs/_static/actions.png" align="center" width="75%"/>
</p>

Important: every time a phase change occurs, the next phase is preeceded by a yellow phase lasting ```yellow_time``` seconds.

<!-- end action -->

### Rewards

<!-- start reward -->

The default reward function is the change in cumulative vehicle delay:

<p align="center">
<img src="docs/_static/reward.png" align="center" width="25%"/>
</p>

That is, the reward is how much the total delay (sum of the waiting times of all approaching vehicles) changed in relation to the previous time-step.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py)) with the parameter `reward_fn` in the [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) constructor.

It is also possible to implement your own reward function:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

<!-- end reward -->

## API's (Gymnasium and PettingZoo)

### Gymnasium Single-Agent API

<!-- start gymnasium -->

If your network only has ONE traffic light, then you can instantiate a standard Gymnasium env (see [Gymnasium API](https://gymnasium.farama.org/api/env/)):
```python
import gymnasium as gym
import sumo_rl
env = gym.make('sumo-rl-v0',
                net_file='path_to_your_network.net.xml',
                route_file='path_to_your_routefile.rou.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

<!-- end gymnasium -->

### PettingZoo Multi-Agent API

<!-- start pettingzoo -->

For multi-agent environments, you can use the PettingZoo API (see [Petting Zoo API](https://pettingzoo.farama.org/api/parallel/)):

```python
import sumo_rl
env = sumo_rl.parallel_env(net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

<!-- end pettingzoo -->

### RESCO Benchmarks

In the folder [nets/RESCO](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets/RESCO) you can find the network and route files from [RESCO](https://github.com/jault/RESCO) (Reinforcement Learning Benchmarks for Traffic Signal Control), which was built on top of SUMO-RL. See their [paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf) for results.

<p align="center">
<img src="sumo_rl/nets/RESCO/maps.png" align="center" width="60%"/>
</p>

### Experiments

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/main/experiments) for examples on how to instantiate an environment and train your RL agent.

### [Q-learning](https://github.com/LucasAlegre/sumo-rl/blob/main/agents/ql_agent.py) in a one-way single intersection:
```bash
python experiments/ql_single-intersection.py
```

### [RLlib PPO](https://docs.ray.io/en/latest/_modules/ray/rllib/algorithms/ppo/ppo.html) multiagent in a 4x4 grid:
```bash
python experiments/ppo_4x4grid.py
```

### [stable-baselines3 DQN](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py) in a 2-way single intersection:
Obs: you need to install stable-baselines3 with ```pip install "stable_baselines3[extra]>=2.0.0a9"``` for [Gymnasium compatibility](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).
```bash
python experiments/dqn_2way-single-intersection.py
```

### Plotting results:
```bash
python outputs/plot.py -f outputs/4x4grid/ppo_conn0_ep2
```
<p align="center">
<img src="outputs/result.png" align="center" width="50%"/>
</p>

## Citing

<!-- start citation -->

If you use this repository in your research, please cite:
```bibtex
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```

<!-- end citation -->

<!-- start list of publications -->

List of publications that use SUMO-RL (please open a pull request to add missing entries):
- [Quantifying the impact of non-stationarity in reinforcement learning-based traffic signal control (Alegre et al., 2021)](https://peerj.com/articles/cs-575/)
- [Information-Theoretic State Space Model for Multi-View Reinforcement Learning (Hwang et al., 2023)](https://openreview.net/forum?id=jwy77xkyPt)
- [A citywide TD-learning based intelligent traffic signal control for autonomous vehicles: Performance evaluation using SUMO (Reza et al., 2023)](https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.13301)
- [Handling uncertainty in self-adaptive systems: an ontology-based reinforcement learning model (Ghanadbashi et al., 2023)](https://link.springer.com/article/10.1007/s40860-022-00198-x)
- [Multiagent Reinforcement Learning for Traffic Signal Control: a k-Nearest Neighbors Based Approach (Almeida et al., 2022)](https://ceur-ws.org/Vol-3173/3.pdf)
- [From Local to Global: A Curriculum Learning Approach for Reinforcement Learning-based Traffic Signal Control (Zheng et al., 2022)](https://ieeexplore.ieee.org/abstract/document/9832372)
- [Poster: Reliable On-Ramp Merging via Multimodal Reinforcement Learning (Bagwe et al., 2022)](https://ieeexplore.ieee.org/abstract/document/9996639)
- [Using ontology to guide reinforcement learning agents in unseen situations (Ghanadbashi & Golpayegani, 2022)](https://link.springer.com/article/10.1007/s10489-021-02449-5)
- [Information upwards, recommendation downwards: reinforcement learning with hierarchy for traffic signal control (Antes et al., 2022)](https://www.sciencedirect.com/science/article/pii/S1877050922004185)
- [A Comparative Study of Algorithms for Intelligent Traffic Signal Control (Chaudhuri et al., 2022)](https://link.springer.com/chapter/10.1007/978-981-16-7996-4_19)
- [An Ontology-Based Intelligent Traffic Signal Control Model (Ghanadbashi & Golpayegani, 2021)](https://ieeexplore.ieee.org/abstract/document/9564962)
- [Reinforcement Learning Benchmarks for Traffic Signal Control (Ault & Sharon, 2021)](https://openreview.net/forum?id=LqRSh6V0vR)
- [EcoLight: Reward Shaping in Deep Reinforcement Learning for Ergonomic Traffic Signal Control (Agand et al., 2021)](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2021/43/paper.pdf)

<!-- end list of publications -->
