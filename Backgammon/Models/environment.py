from agent import BackgammonAgent
from Backgammon.game_logic.table import Table 
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

class Environment:
    def __init__(self, total_steps_per_episode = 10, steps_increment = 1.2, total_episodes_before_step_increment = 10, episode_amount = 500, learning_rate = 0.00025, optimizer=optim.Adam, criterion= nn.MSELoss, model=NeuralNetwork, model_in_size = 64, model_out_size = 5) -> None:
        self.env = "Backgammon"
        self.agent_map = self.spawn_agents(2)
        
        self.episode_amount = episode_amount
        self.total_reward = 0
        self.total_steps_per_episode = total_steps_per_episode  # Ensure Integer
        self.steps_increment = steps_increment                  # Percentage
        self.total_episodes_before_step_increment = total_episodes_before_step_increment

        self.learning_rate = learning_rate
        self.model = model(model_in_size, model_out_size)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion()

    def train(self):
        # For each episode (game)

        # play amount of steps

        for i in range(self.episode_amount):
            table = Table(i, already_seated_players=self.agent_map)
            table.play_game()
        self.update_agents()

    def update_agents(self):
        self.total_steps_per_episode = round(self.total_steps_per_episode * self.steps_increment)

    def spawn_agents(self, number_of_agents):
        return {i:BackgammonAgent(self.model, self.optimizer, self.criterion, self.learning_rate) for i in range(number_of_agents)}