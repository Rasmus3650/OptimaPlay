import sys
import os
from .player import Player
from .card import Card
import numpy as np




class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, table, start_balance: int = None, save_game = False, game_folder = None) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.return_function = return_function
        self.table = table
        self.game_ended = False
        self.save_game = save_game
        self.card_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10, "A": 11}
        self.dealer_upcard = None
        self.dealer_downcard = None
        self.dealers_cards = [self.dealer_upcard, self.dealer_downcard]
        self.upcard_revealed = False
        self.current_player = list(self.player_list.keys())[0]
        self.player_hand_id = 0
        self.bets = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.results = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.game_folder = game_folder
        self.all_actions = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.get_bets()
        self.deal_cards()
    
    def get_bets(self):
        for p_id in list(self.player_list.keys()):
            self.bets[p_id] = self.player_list[p_id].place_bet()
    
    def get_dealer_value(self):
        accumulators = [0,0]
        ace = False
        for j, card in enumerate(self.dealers_cards):
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

    def get_final_dealer_value(self, dealer_values):
        if dealer_values[1] <= 21:
            return dealer_values[1]
        else:
            return dealer_values[0]


    def player_performed_action(self):
        print(f"##########################################")
        print(f"Plyer {self.current_player}'s turn")
        print(list(self.player_list.keys()))
        player = self.player_list[self.current_player]
        print(f"Hand before: {player.hands}")
        action = player.perform_action(self.player_hand_id)
        self.all_actions[player.player_id].append(action)
        print(f"Player {self.current_player} performed action {action}")
        print(f"Hand after: {player.hands}")
        print(f"##########################################")


        if action.action_str == "Stand":
            self.player_hand_id += 1
            if len(player.hands) <= self.player_hand_id:
                id_list = list(self.player_list.keys())
                next_player_idx = id_list.index(self.current_player) + 1
                if next_player_idx >= len(id_list):
                    self.game_over()
                else:
                    self.current_player = id_list[id_list.index(self.current_player) + 1]
                    self.player_hand_id = 0

        
        
        

    
    def get_results(self):
        dealer_val = self.get_final_dealer_value(self.get_dealer_value())
        for p_id in list(self.player_list.keys()):
            hands = self.player_list[p_id].hands
            for hand_id in list(hands.keys()):
                hand_value = hands[hand_id]["Value"][1]
                if hand_value > 21:
                    hand_value = hands[hand_id]["Value"][0]
                
                if hand_value > 21:
                    self.results[p_id][hand_id] = "Busted"
                    continue
                if dealer_val > 21:
                    self.results[p_id][hand_id] = "Won"
                    continue
                if hand_value == dealer_val:
                    self.results[p_id][hand_id] = "Push"
                    continue
                if hand_value < dealer_val:
                    self.results[p_id][hand_id] = "Lost"
                    continue
                if hand_value > dealer_val:
                    self.results[p_id][hand_id] = "Won"
                    continue
            

    def dealers_turn(self):
        self.upcard_revealed = True
        dealer_value = self.get_dealer_value()

        while dealer_value[0] < 17:
            self.dealers_cards.append(self.table.deck.draw_cards(1)[0])
            dealer_value = self.get_dealer_value()



    def game_over(self):
        self.dealers_turn()
        self.get_results()
        self.print_cards()
        print(f"________________")
        print(f"Dealers cards: {self.dealers_cards}")
        print(self.results)
        print(f"________________")
        self.game_ended = True
        if self.save_game:
            self.record_game()
        self.return_function()
        

    def deal_cards(self):
        for p_id in list(self.player_list.keys()):
            player = self.player_list[p_id]
            if player.balance > 0.01:
                c = self.table.deck.draw_cards(2)
                player.set_hand([c])
        self.dealer_upcard = self.table.deck.draw_cards(1)[0]
        self.dealer_downcard = self.table.deck.draw_cards(1)[0]
        self.dealers_cards = [self.dealer_upcard, self.dealer_downcard]

    def print_cards(self):
        for p_id in list(self.player_list.keys()):
            player = self.player_list[p_id]
            print(f"Player {p_id}: {player.hands}")
        print(f"Dealer upcard: {self.dealer_upcard}")
        print(f"Dealer downcard: {self.dealer_downcard}")

    def record_game(self):
        print(f"ALJKDNGFALKJSFN")
        max_len = -1
        header_str = f""
        for p_id in list(self.all_actions.keys()):
            header_str += f"P {p_id}, "
            if len(self.all_actions[p_id]) > max_len:
                max_len = len(self.all_actions[p_id])
        header_str = header_str[:-2] + "\n"

        entire_ac_str = f""

        for i in range(max_len):
            ac_str = f""
            for p_id in list(self.all_actions.keys()):
                if len(self.all_actions[p_id]) <= i:
                    ac_str += f"[], "
                else:
                    ac_str += f"[{p_id};{self.all_actions[p_id][i].action_str};{self.all_actions[p_id][i].hand_id}], "
            ac_str = ac_str[:-2] + "\n"
            entire_ac_str += ac_str
        print(self.game_folder)
        csv_file = open(os.path.join(self.game_folder, f"Actions.csv"), "w")
        csv_file.write(header_str)
        csv_file.write(entire_ac_str)
        csv_file.close()

