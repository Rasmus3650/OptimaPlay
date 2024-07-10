from agent import PokerAgent
from Poker.game_logic.table import Table
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = nn.ELU(self.fc1(x))
        x = nn.ELU(self.fc2(x))
        x = self.fc3(x)
        return x


class RLModel():
    def __init__(self, game, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.game = game
        self.model = NeuralNetwork(1000, 1) # Set input size to 1000 for now
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Metrics, stored as lists so the loss / acc for epoch i will be at index i in the lists
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def start_training(self):
        for epoch in range(self.max_epochs):
            # Setup for training
            # Spawn agents and setup environment
            for i in self.batch_size:
                agents = self.spawn_agents(6)

                # Agents needs to extend the Player class, so we can give it as input to the Table
                for j in range(self.batch_iterations):
                    table = Table(1000,j)
                    
                # Save the best Agents  (most frequent Table winner) for each Table
                # Play the best agents against each other
                # This ranking can be used to calculate the prob of sampling an agent for next iteration

                
            # Compute loss / acc for the epoch
        self.compute_loss()

    def spawn_agents(self, number_of_agents):
        return [PokerAgent() for _ in range(number_of_agents)]
