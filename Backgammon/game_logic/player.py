import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Backgammon.Strategies.random_strategy import Random_strategy

class Player():
    def __init__(self, p_id, is_us: bool = False, strategy = Random_strategy, table = None) -> None:
        self.player_id = p_id
        self.is_us = is_us
        self.strategy = strategy()
        self.backgammon_color = -1

    def set_color(self, color):
        self.backgammon_color = color
    
    def get_action(self, moves):
        return self.strategy.compute_action(moves)
    