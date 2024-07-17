from agent import BackgammonAgent
from Backgammon.game_logic.table import Table
from Backgammon.game_logic.game import Game 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
    def __init__(self, total_steps_per_episode = 10, steps_increment = 1.2, total_episodes_before_step_increment = 10, total_episodes = 500, epsilon = 0.3, epsilon_decay_rate = 0.01, update_after_actions = 10, learning_rate = 0.00025, batch_size = 32, optimizer=optim.Adam, criterion= nn.MSELoss, model=NeuralNetwork, model_in_size = 64, model_out_size = 5) -> None:
        self.env = "Backgammon"
        self.agent_map = self.spawn_agents(2)
        
        self.total_episodes = total_episodes
        self.total_reward = 0
        self.total_steps_per_episode = total_steps_per_episode  # Ensure Integer
        self.steps_increment = steps_increment                  # Percentage
        self.total_episodes_before_step_increment = total_episodes_before_step_increment
        self.update_after_actions = update_after_actions

        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model(model_in_size, model_out_size)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion()

    def train(self):
        action_history = []
        state_history = []
        next_state_history = []
        done_history = []
        rewards_history = []

        frame_count = 0


        self.agent_map[0].set_color(0)
        self.agent_map[1].set_color(1)

        current_player = 0

        # For each episode (game)
        for episode in range(self.episode_amount):
            # reset game state to fresh start
            env = Game(episode, self.agent_map)
            env.current_player = current_player
            # get gamestate
            rolls = env.roll_dice()
            if rolls[0] == rolls[1]:
                dice = [rolls[0], rolls[0], rolls[1], rolls[1]]
            else:
                dice = [rolls]
            moves = env.board.get_moves(env.current_player.backgammon_color, dice)

            state = env.capture_state(moves)
            episode_reward = 0
            for step in range(self.total_steps_per_episode):
                frame_count += 1

                while len(dice) > 0:
                    # Compute action
                    if torch.rand(1).values <= self.epsilon:
                        action = moves[torch.randint(0, len(moves), (1,)).values] # TODO: check for None moves?
                    else:
                        action = env.current_player.compute_action(moves) # TODO: Check for None moves?

                    # decay epsilon
                    self.epsilon *= self.epsilon_decay_rate

                    # Get new game state given computed action
                    env.board.perform_move(env.current_player.backgammon_color, moves)
                    env.all_actions.append([len(env.all_actions), env.current_player.backgammon_color,  [list(x) for x in self.board.board], list(self.board.bar), [rolls[0], rolls[1]], action, [list(self.board.white_home), list(self.board.black_home)]])
                    dice.remove(abs(action[2]))
                    env.check_winner()
                    done = env.game_ended

                    next_state = env.capture_state(moves) # TODO: give rewards based on move / state
                    episode_reward += 0 # TODO: Reward cumulation here

                    action_history.append(action)
                    state_history.append(state)
                    next_state_history.append(next_state)
                    done_history.append(done)
                    # rewards_history.append(reward) # TODO: append reward
                    state = next_state

                    if self.update_after_actions % frame_count == 0 and len(done_history) >= self.batch_size:
                        # TODO: Update model
                        indices = np.random.choice(range(len(done_history)), size=self.batch_size)

                        # draw sample Not sure if this works correctly by converting it to tensors as such.
                        action_sample = torch.Tensor([action_history[i] for i in indices])
                        state_sample = torch.Tensor([state_history[i] for i in indices])
                        next_state_sample = torch.Tensor([next_state_history[i] for i in indices])
                        done_sample = torch.Tensor([done_history[i] for i in indices])
                        # rewards_sample = torch.Tensor([rewards_history[i] for i in indices])

                        # TODO: Make best model predict on next state samle to get the possible reward
                        # TODO: one_hot encode action sample with all possible actions (optional, might not be needed)
                        # TODO: with gradient predict with current model on state sample
                        # TODO: Compute loss

                    if done:
                        break

                # if terminated break
                if done:
                    break

    def update_agents(self):
        self.total_steps_per_episode = round(self.total_steps_per_episode * self.steps_increment)

    def spawn_agents(self, number_of_agents):
        return {i:BackgammonAgent(self.model, self.optimizer, self.criterion, self.learning_rate) for i in range(number_of_agents)}