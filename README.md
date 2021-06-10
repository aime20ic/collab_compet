[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: ./images/agent_performance.png "MADDPG Agent Performance"

# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Installation

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - __Linux__ or __Mac__:
  
    ```bash
    conda create --name collab_compet python=3.6
    source activate collab_compet
    ```

   - __Windows__:

    ```bash
    conda create --name collab_compet python=3.6 
    activate collab_compet
    ```

2. Clone Navigation repository

    ```bash
    git clone git@github.com:aime20ic/collab_compet.git
    ```

3. Install [dependencies](#dependencies)

4. Download [Unity Simulation Environment](#unity-simulation-environment)

### Dependencies

To install required dependencies to execute code in the repository, follow the instructions below.

1. Install [PyTorch](https://pytorch.org/)

    ```bash
    conda install pytorch cudatoolkit=10.2 -c pytorch
    ```

    A fresh installation of [PyTorch](https://pytorch.org/) is recommended due to installation errors encountered when installing the [Udacity Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning), such as [the following Windows error](https://github.com/udacity/deep-reinforcement-learning/issues/13), as well as outdated driver issues when using `torch==0.4`.

2. Install required packages using `pip` from main repository directory

    ```bash
    cd collab_compet
    pip install .
    ```

### Unity Simulation Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Save the file locally and unzip (or decompress) the file.

## Instructions

The `run_agent.py` script can be used to train or evaluate an agent. Logs of agent parameters, agent performance (as shown below), and environment evaluation settings are saved during script execution. Some agent and associated model hyperparameters are configurable via command line arguments. See [help](#help) section for more details about available parameters.

![MADDPG Agent Performance][image2]

Training an agent only requires specifying the path to the [downloaded Unity simulation environment](#getting-started)

```bash
python -m collab_compet.run_agent --sim Tennis_Windows_x86_64/Tennis.exe
```

Model training parameters are configurable via command line. Certain variables such as env name are used for
logging of agent parameters and environment performance results.

```bash
python -m collab_compet.run_agent --sim Tennis_Windows_x86_64/Tennis.exe --n-episodes 1000 --seed 5
```

### Continuing Training

To continue training using a previously trained model, specify the path to the previously saved model using the `--actor` and `--critic` command line arguments.

**NOTE:** Saved model hidden layer sizes must match sizes specified in `model.py`. 

```bash
python -m collab_compet.run_agent --sim Tennis_Windows_x86_64/Tennis.exe --actor example_models/actor1.pth example_models/actor2.pth --critic example_models/critic1.pth example_models/critic2.pth
```

### Evaluating a Trained Agent

Evaluating a trained agent requires using the `--actor` and `--critic` command line arguments as well as `--test` argument simultaneously. The number of evaluation episodes is specified using `--n-episodes` argument, while `--max-t` argument specifies the number of maximum simulation time steps per episode. 

```bash
python -m collab_compet.run_agent Tennis_Windows_x86_64/Tennis.exe --actor example_models/actor1.pth example_models/actor2.pth --critic example_models/critic1.pth example_models/critic2.pth --n-episodes 100 --test --seed 15
```

### Help

For a full list of available command line parameters try

```bash
$ python -m collab_compet.run_agent --help
usage: run_agent.py [-h] [--actor [ACTOR [ACTOR ...]]]
                    [--critic [CRITIC [CRITIC ...]]] [--n-episodes N_EPISODES]
                    [--output OUTPUT] [--run-id RUN_ID] [--sim SIM] [--test]
                    [--window WINDOW] [--seed SEED]

Agent hyperparameters

optional arguments:
  -h, --help            show this help message and exit
  --actor [ACTOR [ACTOR ...]]
                        Path to actor models to load
  --critic [CRITIC [CRITIC ...]]
                        Path to critic models to load
  --n-episodes N_EPISODES
                        Maximum number of training episodes
  --output OUTPUT       Directory to save models, logs, & other output
  --run-id RUN_ID       Execution run identifier
  --sim SIM             Path to Unity Tennis simulation
  --test                Test mode, no agent training
  --window WINDOW       Window size to use for terminal condition check
  --seed SEED           Seed for repeatability
```