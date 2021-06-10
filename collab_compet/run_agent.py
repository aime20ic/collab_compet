import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque

from collab_compet.agent import MADDPG
from collab_compet.unity_env import UnityEnv


def parse_args():
    """
    Parse specified arguments from command line

    Args:
        None

    Returns:
        Argparse NameSpace object containing command line arguments

    """
    parser = argparse.ArgumentParser(description='Agent hyperparameters')
    parser.add_argument('--actor', nargs='*', help='Path to actor models to load')
    parser.add_argument('--critic', nargs='*', help='Path to critic models to load')
    parser.add_argument('--n-episodes', type=int, default=2000, help='Maximum number of training episodes')
    parser.add_argument('--output', type=str, default='./output', help='Directory to save models, logs, & other output')
    parser.add_argument('--run-id', type=int, help='Execution run identifier')
    parser.add_argument('--sim', type=str, default='Tennis_Windows_x86_64/Tennis.exe', 
        help='Path to Unity Tennis simulation')
    parser.add_argument('--test', action='store_true', help='Test mode, no agent training')
    parser.add_argument('--window', type=int, default=100, help='Window size to use for terminal condition check')
    parser.add_argument('--seed', type=int, default=0, help='Seed for repeatability')
    args = parser.parse_args()

    # Convert current time to run ID if none specified
    if args.run_id is None:
        args.run_id = int(time.time())

    # Convert string paths to path objects
    args.output = Path(args.output + '/' + str(args.run_id) + '/')
    args.actor = Path(args.actor) if args.actor else None
    args.critic = Path(args.critic) if args.critic else None

    return args

def write2path(text, path):
        """
        Write text to path object

        Args:
            text (str): Text to log
            path (Path): Path object
        
        Returns:
            None
        
        """

        # Create path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write text to path
        if not path.exists():
            path.write_text(text)
        else:
            with path.open('a') as f:
                f.write(text)
        
        return

def plot_performance(scores, name, window_size):
    """
    Plot summary of DQN performance on environment

    Args:
        scores (list of float): Score per simulation episode
        name (Path): Name for file
        window_size (int): Windowed average size

    """
    window_avg = []
    window_std = []
    window = deque(maxlen=window_size)
    
    # Create avg score
    avg = [np.mean(scores[:i+1]) for i in range(len(scores))]
    for i in range(len(scores)):
        window.append(scores[i])
        window_avg.append(np.mean(window))
        window_std.append(np.std(window))

    # Create 95% confidence interval (2*std)
    lower_95 = np.array(window_avg) - 2 * np.array(window_std)
    upper_95 = np.array(window_avg) + 2 * np.array(window_std)

    # Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(np.arange(len(scores)), scores, color='cyan', label='Scores')
    plt.plot(avg, color='blue', label='Average')
    plt.plot(window_avg, color='red', label='Windowed Average (n={})'.format(window_size))
    plt.fill_between(np.arange(len(window_std)), lower_95, upper_95, color='red', alpha=0.1)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    ax.margins(y=.1, x=.1) # Help with scaling
    plt.show(block=False)

    # Save figure
    name.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name)
    plt.close()

    return

def eval_agent(agent, env, eval_type, **kwargs):
    """
    Train agent

    Args:
        agent (Agent): DDPG Agent
        env (UnityEnv): Unity simulation environment 
        eval_type (str): Training or testing agent
        n_episodes (int): Number of training episodes
        tmax (int): Number of trajectories to collect

    Returns:
        None

    """
    eval_options = ['train', 'test']            # evaluation options

    # Set score variables
    scores = []                                 # scores from each episode
    best_avg_score = -100                       # best averaged window score
    best_avg_score_std = None                   # best averaged window score std
    score_goal = kwargs.get('goal', 0.5)        # goal to get to
    window_size = kwargs.get('window', 100)     # size for rolling window
    scores_window = deque(maxlen=window_size)   # last 100 scores

    # Error check
    if eval_type.lower() not in eval_options:
        raise ValueError(
            'Invalid eval_type specified. Options are {}'.format(eval_options)
        )

    # Initialize key word argument variables
    max_t = kwargs.get('max_t', 999)
    n_episodes = kwargs.get('n_episodes', 2000)
    run_id = kwargs.get('run_id', int(time.time()))
    output = kwargs.get('output', Path('./output/' + str(run_id) + '/'))

    # Create log name
    prefix = str(run_id) + '__' + agent.name + '__' + env.name
    log = output / (prefix + '__performance.log')

    # Start timer
    start_time = time.time()
    elapsed_time = lambda: time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

    # Iterate through episodes
    for episode in range(n_episodes):

        # Reset environment
        score = np.zeros(len(env.agents))
        states = env.reset(train=True if eval_type == 'train' else False)
        
        # Learn for max_t steps
        for t in range(max_t):

            # Get actions for all agents
            actions = agent.act(states)

            # Send actions for all agents
            _, _, rewards, next_states, dones = env.step(actions)

            # Update all agents
            if eval_type == 'train':
                agent.step(states, actions, rewards, next_states, dones)

            # Update next state & score
            states = next_states
            score += rewards

            # Check terminal condition
            if any(dones):
                break
        
        # Get max reward
        score = np.max(score)

        # Save most recent scores
        scores_window.append(score)
        scores.append(score)

        # Calculate average & standard deviation of current scores
        scores_mean = np.mean(scores_window)
        scores_std = np.std(scores_window)

        # Print & log episode performance
        window_summary = ('\rEpisode {}\tAverage Score: {:.2f} ± {:.2f}'
            '\tElapsed Time: {}').format(
            episode, scores_mean, scores_std, elapsed_time())
        print(window_summary, end="")

        # Check terminal condition every window_size episodes
        if episode % window_size == 0:
            
            # Save best performing model (weights)
            if eval_type=='train' and scores_mean >= best_avg_score:
                output.mkdir(parents=True, exist_ok=True)
                agent.save(prefix + '__best')
                best_avg_score = scores_mean
                best_avg_score_std = scores_std

            # Print & log performance of last window_size runs
            window_summary = ('\rEpisode {}\tAverage Score: {:.2f} ± {:.2f}'
                '\tElapsed time: {}').format(episode, scores_mean, scores_std, elapsed_time())
            print(window_summary)
            write2path(window_summary, log)

            # Terminal condition check (early stop / overfitting)
            if eval_type == 'train' and scores_mean < best_avg_score:
                window_summary = ('\rEarly stop at {:d}/{:d} episodes!\rAverage Score: {:.2f} ± {:.2f}'
                    '\rBest Average Score: {:.2f}\tElapsed Time: {}').format(
                        episode, n_episodes, scores_mean, scores_std, best_avg_score, elapsed_time())
                print(window_summary)
                write2path(window_summary, log)
                break

            # Terminal condition check (hit goal)
            if eval_type == 'train' and scores_mean - scores_std >= score_goal:
                window_summary = ('\nEnvironment solved in {:d}/{:d} episodes!\tAverage Score: {:.2f}±{:.2f}'
                    '\tElapsed time: {}').format(episode, n_episodes, scores_mean, scores_std,  elapsed_time())
                print(window_summary)
                write2path(window_summary, log)
                break

    # Save final model (weights)
    if eval_type == 'train': 
        output.mkdir(parents=True, exist_ok=True)
        agent.save(prefix)
    
    # Plot training performance
    plot_performance(scores, output / (prefix + '__training.png'), window_size)
    
    # Save evaluation parameters
    parameters = {
        'n_episodes': n_episodes,
        'eval_type': eval_type, 
        'max_t': max_t,
        'agent_seed': agent.rng_seed,
        'env_seed': env.rng_seed,
        'best_avg_score': best_avg_score,
        'best_avg_score_std': best_avg_score_std,
        'scores_mean': scores_mean,
        'scores_std': scores_std
    }
    with open(output / (prefix + '__parameters.json'), 'w') as file:
        json.dump(parameters, file, indent=4, sort_keys=True)

    return
    
def main(args):
    """
    Run agent in environment

    Args:
        args: Argparse NameSpace object containing command line arguments

    Returns:
        None

    """

    # Create env
    env = UnityEnv(args.sim, name='Tennis', seed=args.seed)

    # Create agent
    agent = MADDPG(env.state_size, env.action_size, n_agents=2, 
        random_seed=args.seed, name='MADDPG', run_id=args.run_id, output=args.output)

    # Load agents
    if args.actor: agent.load(args.actor, 'actor')
    if args.critic: agent.load(args.critic, 'critic')

    # Evaluate agent
    train_mode = 'test' if args.test else 'train'
    eval_agent(agent, env, train_mode, **vars(args))

    # Close env
    env.close()

    return


if __name__ == "__main__":
    """
    Execute script
    """
    args = parse_args()
    main(args)
