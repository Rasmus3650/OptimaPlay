import os, sys, random
from .strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.player_action import Player_Action

class Random_strategy(Strategy):
    def __init__(self) -> None:
        pass

    def compute_action(self, table, player_id: int) -> Player_Action:
        if super().compute_action(table, player_id) == "NoAction": return None
        #print(f"AAAA: {table.seated_players[player_id].folded}")
        #print(f"PLAYER: {player_id}")
        #print(table.seated_players[player_id].actions[1:-1])
        return random.choice(table.seated_players[player_id].actions)

    def compute_bet_amount(self, table, player_id: int):
        return round(random.uniform(0.01, table.seated_players[player_id].balance), 2)