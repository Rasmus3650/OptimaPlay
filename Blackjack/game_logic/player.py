from .card import Card
from .player_action import Player_Action
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Blackjack.Strategies.random_strategy import Random_strategy
from Blackjack.Strategies.GTO_strategy import GTO_strategy

class Player():
    def __init__(self, player_id: int, is_us: bool = False, balance: int = 1.6, strategy = Random_strategy, table = None) -> None: #TESTING (1.6)
        self.player_id = player_id
        self.hands = {} #{0: {"Cards": [Card(7, "Hearts"), Card(10, "Spades")], "Value": [17, 17]}
        self.is_us = is_us
        self.balance = balance
        self.strategy = strategy()
        self.table = table
        #self.hand_values = [[]] #Hard, Soft
        self.actions =  [Player_Action(table, self.player_id, action, None, None) for action in ["Hit", "Stand"]]       # Set of possible actions

    def place_bet(self):
        bet_amount = self.strategy.compute_bet_amount(self.table, self.player_id)
        self.balance = round(self.balance - bet_amount, 2)
        return bet_amount
    
    def clear_hand(self):
        self.hands = {}

    def set_hand(self, cards):
        for hand_id, hand in enumerate(cards): 
            self.hands[hand_id] = {"Cards": hand, "Value": self.get_hand_value(hand)}
    
    def get_hand_value(self, hand):
        accumulators = [0,0]
        ace = False
        for j, card in enumerate(hand):
            if card.current_rank != 11:
                accumulators[0] += card.current_rank
                accumulators[1] += card.current_rank
            elif card.current_rank == 11 and not ace:
                ace = True
                accumulators[0] += 1
                accumulators[1] += 11
            elif card.current_rank == 11 and ace:
                accumulators[0] += 1
                accumulators[1] += 1
        return accumulators
    
    def detect_possible_actions(self, hand_id):
        print(self.hands)
        hand_sum = self.hands[hand_id]["Value"][0]
        hand_cards = self.hands[hand_id]["Cards"]
        if hand_sum < 17 and (len(hand_cards) == 2 and hand_cards[0].current_rank == hand_cards[1].current_rank):
            #self.actions =  [Player_Action(self.table, self.player_id, action, hand_id=hand_id, new_card=None) for action in ["Hit", "Stand", "Split"]]
            self.actions =  [Player_Action(self.table, self.player_id, action, hand_id=hand_id, new_card=None) for action in ["Split"]] #Lige nu er de tvunget til at splitte hvis de kan
        elif hand_sum < 17:
            self.actions =  [Player_Action(self.table, self.player_id, action, hand_id=hand_id, new_card=None) for action in ["Hit", "Stand"]]
        else:
            self.actions =  [Player_Action(self.table, self.player_id, action, hand_id=hand_id, new_card=None) for action in ["Stand"]]
    
    def perform_action(self, hand_id):
        self.detect_possible_actions(hand_id)

        action = self.strategy.compute_action(self.table, self.player_id, hand_id)

        if action.action_str == "Hit":
            newcard = self.table.deck.draw_cards(1)[0]
            action.new_card = newcard
            self.hands[hand_id]["Cards"].append(newcard)
            self.hands[hand_id]["Value"] = self.get_hand_value(self.hands[hand_id]["Cards"])
        elif action.action_str == "Split":
            card_to_move = self.hands[hand_id]["Cards"].pop()
            newcards = self.table.deck.draw_cards(2)
            self.hands[hand_id]["Cards"].append(newcards[0])
            self.hands[hand_id]["Value"] = self.get_hand_value(self.hands[hand_id]["Cards"])
            self.hands[len(list(self.hands.keys()))] = {"Cards": [card_to_move, newcards[1]], "Value": self.get_hand_value([card_to_move, newcards[1]])}

            
        return action