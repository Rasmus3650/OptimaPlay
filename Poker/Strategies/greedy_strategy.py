from game_logic.player_action import Player_Action
from strategy import Strategy

class Greedy_strategy(Strategy):
    def __init__(self) -> None:
        pass

    def compute_action(self, table, player_id: int) -> Player_Action:
        super().compute_action(table, player_id)
        # Get total table bet

        # Get Expected value given observation

        # If EV over some treshhold, play greed
        pass

    def compute_bet_amount(self, table, player_id: int) -> float:
        super().compute_bet_amount(table, player_id)
        pass
