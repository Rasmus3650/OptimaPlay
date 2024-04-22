from agent import PokerAgent
from Poker.game_logic.table import Table
class RLModel():
    def __init__(self, learning_rate=0.001, reward_function = None, batch_size=6,batch_iterations=10, max_epochs=100):
        self.learning_rate = learning_rate
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.epoch = 0
        self.batch_iterations = batch_iterations
        self.max_epochs = max_epochs
        
        # Metrics, stored as lists so the loss / acc for epoch i will be at index i in the lists
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def start_training(self):
        for epoch in range(self.max_epochs):
            # Setup for training
            # Spawn agents and setup environment
            for i in self.batch_size // 6:
                agents = self.spawn_agents(6)

                # Agents needs to extend the Player class, so we can give it as input to the Table
                for j in range(self.batch_iterations):
                    table = Table(1000,j)
                    
                # Save the best Agents  (most frequent Table winner) for each Table
                # Play the best agents against each other
                # This ranking can be used to calculate the prob of sampling an agent for next iteration

                
            # Compute loss / acc for the epoch

    def compute_loss(self):
        # Compute the loss given 
        pass

    def spawn_agents(self, number_of_agents):
        retu[PokerAgent() for _ in range(number_of_agents)]
