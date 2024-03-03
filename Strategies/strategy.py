
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.player_action import Player_Action


class Strategy():
    def __init__(self) -> None:
        pass
    
    def compute_action(self, table, player_id: int) -> Player_Action:
        if table.seated_players[player_id].folded:
            return None

    def compute_bet_amount(self, table, player_id: int):
        pass