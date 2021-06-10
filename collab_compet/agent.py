import time
import json
import numpy as np
from pathlib import Path

from collab_compet.ddpg import DDPG

class MADDPG():
    """
    Multi-Agent DDPG
    """

    def __init__(self, state_size, action_size, n_agents, random_seed, **kwargs):
        """
        Create all agents

        Args:
            state_size (int): Environment observation size
            action_size (int): Environment action size
            n_agents (int): Number of agents in environment
            random_seed (int): Seed for repeatability 
        
        Returns:
            None
        
        """
        self.name = kwargs.get('name', 'MADDPG')
        self.run_id = kwargs.get('run_id', int(time.time()))
        self.output = kwargs.get('output', 
            Path('./output/' + str(self.run_id) + '/'))
        self.rng_seed = random_seed

        # Create agents
        self.agents = [
            DDPG(state_size, action_size, random_seed, name='DDPG-{}'.format(i), 
                run_id=self.run_id, output=self.output) 
            for i in range(n_agents)
        ]
        
        # Reset agents
        [agent.reset() for agent in self.agents]
    
    def load(self, paths, ac):
        """
        Load

        Args:
            path (list of Path): Saved model weights to load
            ac (str): Actor or critic

        Returns:
            None
        """
        [agent.load(path, ac) for agent, path in zip(self.agents, paths)]

    def act(self, observations):
        """
        Get actions for all agents
        
        Args:
            observations (array): Observation for each agent

        Returns:
            actions (array of arrays): Continuous actions for each agent
        
        """
        actions = [agent.act(obs) for agent, obs in zip(self.agents, observations)]
        return np.array(actions)

    def step(self, states, actions, rewards, next_states, dones):
        """
        Add memory to experience replay buffer & learn for each agent
        
        Args:
            states (array): Observations for each agent for current time step
            actions (array): Continuous actions for each agents
            rewards (array): Rewards for each agents
            next_states (array): Observations for each agent for next time step
            dones (array): Environment complete for each agent

        Returns:
            None
        
        """
        memories = zip(states, actions, rewards, next_states, dones)
        [agent.step(*memory) for memory, agent in zip(memories, self.agents)]

    def save(self, prefix):
        """
        Save all agent models

        Args:
            prefix (str): Prefix for saving DDPG models

        Returns:
            None

        """
        [agent.save(prefix + '__' + agent.name) for agent in self.agents]

