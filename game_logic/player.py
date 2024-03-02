from .card import Card


class Player():
    def __init__(self, player_id: int, is_us: bool = False, balance: int = None, strategy = None) -> None:
        self.player_id = player_id
        self.is_us = is_us
        self.balance = balance
        self.bet_history = []
        self.folded = False
        self.action_history = []
        self.hand = []                                              # List of 2 cards
        self.actions = ["Bet", "Call", "Fold", "Raise"]             # Set of possible actions
        self.play_style = ["Aggressive", "Passive"]                 # Class labels for detected playstyles
        self.strategy = strategy                                                # Call the Strategy with the correct strategy, function

    def perform_action(self, action: str, amount: int = None):
        if action == "Fold":
            self.folded = True
        self.bet_history.append(amount)
        self.action_history.append(action)
    
    def set_hand(self, card_1: Card, card_2: Card):
        self.hand = [card_1, card_2]

    def __repr__(self) -> str:
        return_str = f"Player {self.player_id}\nIs Us: {self.is_us}\nHand: {self.hand} - Folded: {self.folded}\nBalance: {self.balance}\n Actions\n"
        for action, bet in zip(self.action_history, self.bet_history):
            return_str += f"  {action} : {bet}\n"
        return return_str