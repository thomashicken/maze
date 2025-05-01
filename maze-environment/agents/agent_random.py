import random

class AgentRandom:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.current_state = None
        self.done = False
        
    def reset(self):
        """Reset the agent's state."""
        self.current_state, _ = self.env.reset()
        self.done = False
        
    def agent_function(self, observation):
        """
        Select a random action (0-3).
        """
        if self.done:
            return None
        return random.choice([0, 1, 2, 3])  # Randomly choose from Up, Right, Down, Left