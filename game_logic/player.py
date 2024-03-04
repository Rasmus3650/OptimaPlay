from .card import Card
from .player_action import Player_Action
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Strategies.random_strategy import Random_strategy

class Player():
    def __init__(self, player_id: int, is_us: bool = False, balance: int = 1.6, strategy = Random_strategy, table = None) -> None: #TESTING (1.6)
        self.player_id = player_id
        self.is_us = is_us
        self.balance = balance
        self.folded = False
        self.all_in = False
        self.action_history: list[Player_Action] = []
        self.table = table
        self.hand = []                                                              # List of 2 cards
        self.play_style = ["Aggressive", "Passive"]                                 # Class labels for detected playstyles
        self.strategy = strategy()                                                  # Call the Strategy with the correct strategy, function
        self.actions =  [Player_Action(table, self.player_id, action) for action in ["Fold", "Check", "Call", "Bet", "Raise"]]       # Set of possible actions



    def perform_action(self):
        action_to_append = self.strategy.compute_action(table=self.table, player_id=self.player_id)
        if action_to_append.action_str == "Bet" or action_to_append.action_str == "Raise":
            amount = self.strategy.compute_bet_amount(self.table, self.player_id)
            self.balance -= amount
            action_to_append.bet_amount = amount
        if action_to_append.action_str == "Fold":
            self.folded = True
        
        self.action_history.append(action_to_append)
        return action_to_append
    
    def set_hand(self, cards: list[Card]):
        self.hand = cards

    def __repr__(self) -> str:
        return_str = f"Player {self.player_id}\nIs Us: {self.is_us}\nHand: {self.hand} - Folded: {self.folded}\nBalance: {self.balance}\n Actions\n"
        for action in self.action_history:
            return_str += f"  {action}\n"
        return return_str