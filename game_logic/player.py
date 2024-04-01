from .card import Card
from .player_action import Player_Action
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Strategies.random_strategy import Random_strategy
from Strategies.GTO_strategy import GTO_strategy


class Player():
    def __init__(self, player_id: int, is_us: bool = False, balance: int = 1.6, strategy = Random_strategy, table = None) -> None: #TESTING (1.6)
        self.player_id = player_id
        self.is_us = is_us
        self.balance = balance
        self.folded = False
        self.all_in = False
        self.total_money_on_table = 0.0                 
        self.current_money_on_table = 0.0
        self.action_history: list[Player_Action] = []
        self.table = table
        self.hand = []                                                              # List of 2 cards
        self.play_style = ["Aggressive", "Passive"]                                 # Class labels for detected playstyles
        self.strategy = strategy()                                                  # Call the Strategy with the correct strategy, function
        self.actions =  [Player_Action(table, self.player_id, action) for action in ["Fold", "Check", "Call", "Raise"]]       # Set of possible actions



    def perform_action(self, somebody_raised, max_currently_on_table = 0):
        #print(f"PLAYER {self.player_id} ACTION")
        if somebody_raised and self.balance < max_currently_on_table:
            self.actions = [Player_Action(self.table, self.player_id, action) for action in ["Fold", "Call"]]
        elif somebody_raised and self.balance >= max_currently_on_table:
            self.actions = [Player_Action(self.table, self.player_id, action) for action in ["Fold", "Call", "Raise"]]
        else:
            self.actions =  [Player_Action(self.table, self.player_id, action) for action in ["Fold", "Check", "Raise"]]

        action_to_append = self.strategy.compute_action(table=self.table, player_id=self.player_id)
        
        if action_to_append is None: return None
        if action_to_append.action_str == "Raise":
            amount = self.strategy.compute_bet_amount(self.table, self.player_id, max_currently_on_table)
            self.current_money_on_table += amount
            self.balance = round(self.balance - amount, 2)
            action_to_append.bet_amount = amount
            if self.balance < 0.01:
                self.all_in = True
        if action_to_append.action_str == "Call":
            needed = max_currently_on_table - self.current_money_on_table
            
            self.current_money_on_table += needed
            action_to_append.bet_amount = needed

            if self.balance <= needed:
                self.balance = 0.0
                self.all_in = True
            else:
                self.balance = round(self.balance - needed, 2)


        if action_to_append.action_str == "Fold":
            self.folded = True
        
        self.action_history.append(action_to_append)
        #print(f"ID {self.player_id}: {action_to_append}")
        return action_to_append
    
    def set_hand(self, cards: list[Card]):
        self.hand = cards
    
    def add_to_balance(self, change):
        #print(f"Adding {change} to player {self.player_id}")
        self.balance = round(self.balance + change, 2)
        #self.balance += change
    
    def new_round(self):
        self.total_money_on_table += self.current_money_on_table
        self.current_money_on_table = 0.0

    def print_actions(self):
        p_str = f"Player {self.player_id} actions:\n"
        for action in self.action_history:
            p_str += f"  {action}\n"

    def __repr__(self) -> str:
        return_str = f"Player {self.player_id}"
        return return_str

    #def __repr__(self) -> str:
    #    return_str = f"Player {self.player_id}\nIs Us: {self.is_us}\nHand: {self.hand} - Folded: {self.folded}\nBalance: {self.balance}\n"
    #    return return_str