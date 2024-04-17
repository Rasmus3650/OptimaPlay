from .card import Card
from .player_action import Player_Action
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Blackjack.Strategies.random_strategy import Random_strategy
from Blackjack.Strategies.GTO_strategy import GTO_strategy

class Player():
    def __init__(self, player_id: int, is_us: bool = False, balance: int = 1.6, strategy = Random_strategy, table = None) -> None: #TESTING (1.6)
        self.player_id = player_id
        #self.hands = []
        self.is_us = is_us
        self.balance = balance
        self.strategy = strategy()
        self.table = table
        self.hand_values = [[0, 0]] #Hard, Soft
        self.actions =  [Player_Action(table, self.player_id, action, None) for action in ["Hit", "Stand"]]       # Set of possible actions

    def set_hand(self, cards): 
        self.hands = cards
        self.hand_values = self.get_hand_value()
    
    def get_hand_value(self):
        accumulators = []
        for i, hand in enumerate(self.hands):
            accumulators.append([0,0])
            for j, card in enumerate(hand):
                if card.current_rank != 11:
                    accumulators[i][0] += card.current_rank
                    accumulators[i][1] += card.current_rank
                elif card.current_rank == 11:
                    accumulators[i][0] += 1
                    accumulators[i][1] += 11
        return accumulators
    
    def perform_action(self, hand_id):
        action = self.strategy.compute_action(self.table, self.player_id, hand_id)
        if action.action_str == "Hit":
            newcard = self.table.deck.draw_cards(1)[0]
            action.new_card = newcard
            print(self.hands)
            self.hands[hand_id].append(newcard)
            print(self.hands)
        return action