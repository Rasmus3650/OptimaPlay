import os, sys, random
from .strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Blackjack.game_logic.player_action import Player_Action

class Random_strategy(Strategy):
    def __init__(self) -> None:
        pass

    def compute_action(self, table, player_id, hand_id) -> Player_Action:
        res = random.choice(table.seated_players[player_id].actions)
        return res

    def compute_bet_amount(self, table, player_id):
        minimum = 0
        maximum = table.seated_players[player_id].balance
        return round(random.uniform(minimum + 0.01, maximum), 2)