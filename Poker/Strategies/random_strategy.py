import os, sys, random
from .strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.player_action import Player_Action

class Random_strategy(Strategy):
    def __init__(self) -> None:
        pass

    def compute_action(self, table, player_id: int, max_currently_on_table) -> Player_Action:
        #print(f"Computing action for player {player_id}")
        #print(f"All in: {table.seated_players[player_id].all_in}")
        #print(f"Folded: {table.seated_players[player_id].folded}")
        if super().compute_action(table, player_id) == "NoAction": 
            #print(f"NOACTION")
            return None
        #print(f"ACTION")
        #print(f"AAAA: {table.seated_players[player_id].folded}")
        #print(f"PLAYER: {player_id}")
        #print(table.seated_players[player_id].actions[1:-1])
        res = random.choice(table.seated_players[player_id].actions)
        if res.action_str == "Raise":
            res.bet_amount = self.compute_bet_amount(table, player_id, max_currently_on_table)
        return res

    def compute_bet_amount(self, table, player_id: int, minimum):
        if minimum > table.seated_players[player_id].balance:
            return table.seated_players[player_id].balance
        
        res = round(random.uniform(minimum + 0.01, table.seated_players[player_id].balance), 2)
        if abs(res - table.seated_players[player_id].balance) < 0.02:
            return table.seated_players[player_id].balance
        return res