import time
import json
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


