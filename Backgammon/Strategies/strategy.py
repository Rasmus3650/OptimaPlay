
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from game_logic.player_action import Player_Action


class Strategy():
    def __init__(self) -> None:
        pass
    
    def compute_action(self, table, player_id: int):# -> Player_Action:
        pass

    def compute_bet_amount(self, table, player_id: int):
        pass