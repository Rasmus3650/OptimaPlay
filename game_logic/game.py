from typing import Any
from .player import Player
from .card import Card
import numpy as np
class Game():
    def __init__(self, game_id, player_list: list[Player], return_function, table, start_balance: int = None) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.active_player_list = {}
        #print(f"ACTTIVE:")
        for i, player in enumerate(self.player_list):
            self.active_player_list[i] = player
            #print(f"Player: {i}:")
            #print(player)
        
        self.rank_list = {"Royal Flush": 10, "Straight Flush": 9, "Four of a Kind": 8, "Full House": 7, "Flush": 6, "Straight": 5, "Three of a Kind": 4, "Two Pairs": 3, "One Pair": 2, "High Card": 1}
        
        self.pot = 0
        self.pot_history = []
        self.all_game_states = ["Pre-round", "Pre-flop", "Flop", "Turn", "River", "Showdown", "Conclusion"]
        self.game_state = "Pre-round"
        self.cards_on_table: list[Card] = []
        self.table = table
        self.return_function = return_function
        self.game_ended = False
        self.dealer = 4
        self.blinds: list[int] = [(self.dealer + 1) % len(self.player_list), (self.dealer + 2) % len(self.player_list)]
        self.current_player: int = (self.dealer + 3) % len(self.player_list)
        self.trans_player:int = (self.current_player - 1) % len(self.player_list)


        self.transition_state()
        
    
    def do_one_round(self):
        for player in self.active_player_list.items():
            player.perform_action()

    def player_performed_action(self):
        player_id = self.current_player

        if self.game_state == "Showdown":
            return None
        action = self.player_list[player_id].perform_action()
        if action.action_str == "Raise" or action.action_str == "Bet" or action.action_str == "Call":
            self.pot = round(self.pot + action.bet_amount, 2)
        if action.action_str == "Raise":
            self.trans_player = player_id - 1 % len(self.player_list)
        if player_id == self.trans_player:
            self.transition_state()
        self.current_player = (self.current_player + 1) % len(self.player_list)
        
        return action

    def get_winner(self, hands_map) -> tuple[Player, int]:
        self.cards_on_table
        
        for player in self.active_player_list:
            self.compute_hand(player.hand, self.cards_on_table)

    def compute_hand(self, hand: list[Card], card_on_table: list[Card]) -> tuple[str, int]: #fx: ("One Pair", 6) Har et par 6
        hand_suits = [hand[0].suit, hand[0].suit]
        table_suits = [elem.suit for elem in card_on_table]
        matching_cards = []
        hand_fitted = np.unique(hand_suits) == 1
        if (hand_fitted and table_suits.count(hand_suits[0]) >= 3) or (table_suits.count(hand_suits[0]) >= 4 or table_suits.count(hand_suits[1]) >= 4) or (np.unique(table_suits) == 1): # Royal Flush
            if hand_fitted:
                matching_cards.append(hand[0])
                matching_cards.append(hand[1])
                for card in card_on_table:
                    if card.suit == hand_suits[0]:
                        matching_cards.append(card)
            elif table_suits.count(hand_suits[0]) >= 4:
                matching_cards.append(hand_suits[0])
                for card in card_on_table:
                    if card.current_suit == hand_suits[0]:
                        matching_cards.append(card)
                        
            elif table_suits.count(hand_suits[1]) >= 4:
                matching_cards.append(hand_suits[1])
                for card in card_on_table:
                    if card.current_suit == hand_suits[1]:
                        matching_cards.append(card)

            elif np.unique(table_suits) == 1:
                for card in hand:
                    if card.current_suit == table_suits[0]:
                        matching_cards.append(card)
                for card in card_on_table:
                    matching_cards.append(card)
            
            rank_list = [elem.current_rank for elem in matching_cards]
            if 10 in rank_list and 11 in rank_list and 12 in rank_list and 13 in rank_list and 14 in rank_list:
                return "Royal Flush"
            


    def transition_state(self):
        if 5 > self.all_game_states.index(self.game_state) and self.all_game_states.index(self.game_state) > 0:
                self.pot_history.append(self.pot)
        new_state = self.all_game_states[self.all_game_states.index(self.game_state) + 1 % len(self.all_game_states)]
        hands_map = {}
        
            
        if new_state == "Pre-flop":
            for i in self.active_player_list.items():
                i.perform_action()
            self.deal_hands()
            self.do_one_round()

        if new_state == "Flop":
            self.deal_table(3)
            self.do_one_round()
        
        if new_state == "Turn" or new_state == "River":
            self.deal_table(1)
            self.do_one_round()
        
        if new_state == "Showdown":
           pass 
            
        if new_state == "Conclusion":
            print(f"Folded: ", end=" ")
            for player in self.player_list:
                print(player.folded, end=" ")
                hands_map[player.player_id] = player.hand
            print()
            print(hands_map)
            self.get_winner(hands_map)


            self.return_function()

        if new_state == "Pre-round":
            print(f"Game ended")
            self.game_ended = True
            self.return_function()
        print(f"Game state transitioned from '{self.game_state}' to '{new_state}'")
        self.game_state = new_state
        
    def deal_hands(self):
        for player in self.player_list:
            if not player.balance <= 0.01:
                self.active_player_list.append(player)
                player.set_hand(self.table.deck.draw_cards(2))
    
    def deal_table(self, amount):
        self.cards_on_table += self.table.deck.deal_cards(amount)
    
    def game_over(self):
        self.dealer = (self.dealer + 1) % len(self.player_list)
        self.current_player: int = (self.dealer + 1) % len(self.player_list)
        self.trans_player:int = (self.current_player - 1) % len(self.player_list)
        self.return_function()
    
    def __repr__(self) -> str:
        return_str = f"Game {self.game_id} (D: {self.dealer}, C: {self.current_player}, T: {self.trans_player})\n  Number of players: {len(self.player_list)}\n  Game State: {self.game_state}\n  Pot: {self.pot}\n  Pot Hist:\n"
        for pot in self.pot_history:
            return_str += f"    {pot}\n"
        return return_str

