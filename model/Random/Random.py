import random

"""Random action model to compare other models with, play a role as a baseline"""

class RandomAction:

    def __init__(self, config):
        self.config = config
        self.action_map = config.model.action_map
        self.num_agents = config.data.num_agents

    def forward(self):
        actions = {}
        for i in range(self.num_agents):
            actions[i] = random.randint(0, len(self.action_map) - 1)
        return actions
    
    def __call__(self):
        return self.forward()