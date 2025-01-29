import os, sys, numpy as np
from .strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Random_strategy(Strategy):
    def __init__(self) -> None:
        pass

    def compute_action(self, moves):
        if len(moves) == 0:
            return None
        return moves[np.random.randint(0, len(moves))]
