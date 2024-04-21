from agent import PokerAgent

class RLModel():
    def __init__(self, learning_rate=0.001, reward_function = None, batch_size=1, max_epochs=100):
        self.learning_rate = learning_rate
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.epoch = 0
        self.max_epochs = max_epochs
        self.agents = [] # List of all agents in the current epoch
        self.best_agents = []   # Keep a percentage of the best agents, for the next epoch
        
        # Metrics, stored as lists so the loss / acc for epoch i will be at index i in the lists
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def start_training(self):
        for epoch in range(self.max_epochs):
            # Setup for training
            # Spawn agents and setup environment
            self.spawn_agents()

            # Evaluate each agent, and store the top x percent in self.best_agents

            # Compute loss / acc for the epoch

    def compute_loss(self):
        # Compute the loss given 
        pass

    def spawn_agents(self):
        self.agents = [PokerAgent() for _ in range(self.batch_size)]