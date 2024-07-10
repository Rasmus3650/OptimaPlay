from Backgammon.Models.models import NeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from Backgammon.game_logic.player import Player

class BackgammonAgent(Player):
    def __init__(self, model, optimizer, criterion, lr=0.00025):
        super.__init__()
        self.state_size = 88
        self.action_size = 32
        self.sentient = True
        self.memory = []
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def compute_action(self, state):
        """
        Compute the action based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            The computed action.
        """
        board, moves, bar, homes = state
        new_board = []  #48
        new_bar = [0, bar.count(0), 1, bar.count(1)] #4
        homes = [0, len(homes[0]), 1, len(homes[1])] #4
        new_moves = []
        for tile in board:
            if len(tile) == 0:
                new_board += [-1, 0]
            else:
                new_board += [tile[0], len(tile)]

        

        valid_moves = len(moves)
        for i in range(32):
            if i >= len(moves):
                new_moves += [0, 0, 0]
            else:
                new_moves += moves[i]
        print(board)
        print(moves)
        print(bar)
        print(homes)
        

        state = torch.tensor(state, dtype=torch.float32)
        print()
        print(state)
        input("State")
        q_values = self.model(state)
        action = torch.argmax(q_values[:valid_moves]).item()
        return action


    def store_action(self, state, action, reward, next_state, done):
        """
        Store the action in the memory.
        
        Args:
            state: The current state of the environment.
            action: The action taken.
            reward: The reward received.
            next_state: The next state of the environment.
            done: Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Learn from the stored actions.
        """
        pass
