import sys
import os
from .player import Player
from .card import Card
import numpy as np




class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, table, start_balance: int = None, save_game = False) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.return_function = return_function
        self.table = table
        self.game_ended = False
        self.save_game = save_game
        self.card_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10, "A": 11}
        self.dealer_upcard = None
        self.dealer_downcard = None
        self.current_player = 0
        self.deal_cards()


    def player_performed_action(self):


        action = self.player_list[self.current_player].perform_action(hand_id=0)
        print(f"Player {self.current_player} performed action {action}")

        if action.action_str == "Stand":
            self.current_player += 1
            if self.current_player >= len(self.player_list):
                self.game_over()



    def game_over(self):
        self.game_ended = True
        self.return_function()
        

    def deal_cards(self):
        for p_id in list(self.player_list.keys()):
            player = self.player_list[p_id]
            if player.balance > 0.01:
                c = self.table.deck.draw_cards(2)
                player.set_hand([c])
        self.dealer_upcard = self.table.deck.draw_cards(1)
        self.dealer_downcard = self.table.deck.draw_cards(1)

    def print_cards(self):
        for p_id in list(self.player_list.keys()):
            player = self.player_list[p_id]
            print(f"Player {p_id}: {player.hands}")
        print(f"Dealer upcard: {self.dealer_upcard}")
        print(f"Dealer downcard: {self.dealer_downcard}")
