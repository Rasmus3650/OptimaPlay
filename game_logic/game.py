from typing import Any
from .player import Player
from .card import Card
import numpy as np
from .hand_evaluator import Hand_Evaluator
import os
import csv

class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, table, start_balance: int = None, save_game = False) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.active_player_list = {}
        #for i, player in enumerate(self.player_list):   #Is overwritten in deal_hands
        #    self.active_player_list[i] = player
        
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
        self.current_player: int = self.get_next_player(self.get_next_player(self.get_next_player(self.dealer, use_standard_player_list=True), use_standard_player_list=True), use_standard_player_list=True)
        self.trans_player:int = self.get_next_player(self.get_next_player(self.dealer, use_standard_player_list=True), use_standard_player_list=True)
        self.hand_evaluator = Hand_Evaluator()
        self.save_game = save_game

        self.action_map = {"Pre-flop": [], "Flop": [], "Turn": [], "River": []}
        self.winner = None

        self.transition_state()
        
    
    def do_one_round(self):
        #print(f"Starting this round: {self.current_player}")
        #print(f"Ending this round: {self.trans_player}\n")

        while self.current_player != self.trans_player:
            action_performed = self.player_performed_action()

        action_performed = self.player_performed_action()

    def get_next_player(self, id = None, use_standard_player_list = False):
        
        if id is None:
            curr = self.current_player
        else:
            curr = id

        
        print(f"\nGETNEXTPLAYER CALLED {curr}")

        player_list = self.active_player_list
        if use_standard_player_list:
            player_list = self.player_list
        print(list(player_list.keys()))
        res_player = None
        counter = 0
        while res_player is None:
            curr = (curr + 1) % len(list(player_list.keys()))
            if curr not in list(player_list.keys()):
                continue
            if not player_list[curr].folded and not player_list[curr].all_in:
                res_player = curr
            counter += 1
            if counter == len(list(player_list.keys())):
                print(f"NO PLAYERS")
                input("!!")
        self.current_player = curr
        print(f"GETNEXTPLAYER RETURNED {curr}\n")
        return curr

    def player_performed_action(self):
        player_id = self.current_player
        #print(f"Game state: {self.game_state}")
        
        if self.game_state == "Showdown":
            return None
        #print(list(self.active_player_list.keys()))
        #print(player_id)
        #print(f"BAL {player_id}: {self.active_player_list[self.current_player].balance}")
        action = self.active_player_list[player_id].perform_action()
        if action is None:
            self.current_player = self.get_next_player()
            return None

        if self.game_state in list(self.action_map.keys()):
            self.action_map[self.game_state].append(action)

        #print(f"Player {player_id} performed {action}\n")
        if action.action_str == "Raise" or action.action_str == "Bet" or action.action_str == "Call":
            self.pot = round(self.pot + action.bet_amount, 2)
        if action.action_str == "Raise":
            self.trans_player = player_id - 1 % len(self.active_player_list)
        if player_id == self.trans_player:
            self.transition_state()                 #TODO: This line never happens...... (i think)
        else:
            print(action)
            self.current_player = self.get_next_player()
        
        return action

    def get_winner(self) -> tuple[Player, str, int]:

        #Håndends primære rank er Royal Flush = 10, Straight Flush = 9, osv.
        #En straigh med tallene 4, 5, 6, 7, 8 har primær rank 5, fordi self.rank_list["Straight"] = 5,
        #og den har sekundær rank 8, fordi 8 er det højeste tal i straighten
        highest_hand_primary_rank = 0
        highest_hand_secondary_rank = 0
        highest_hand_str = ""
        winner = None
        
        for player_id in list(self.active_player_list.keys()):
            curr_player = self.active_player_list[player_id]
            hand_str, hand_secondary_rank = self.hand_evaluator.compute_hand(curr_player.hand, self.cards_on_table)
            if self.rank_list[hand_str] >= highest_hand_primary_rank:
                if hand_secondary_rank < highest_hand_secondary_rank: #Måske håndtere hvis de er lig med hinanden???
                    continue
                highest_hand_primary_rank = self.rank_list[hand_str]
                highest_hand_secondary_rank = hand_secondary_rank
                highest_hand_str = hand_str
                winner = curr_player
        
        return (winner, highest_hand_str, highest_hand_secondary_rank)

    
    def transition_state(self):
        if 5 > self.all_game_states.index(self.game_state) and self.all_game_states.index(self.game_state) > 0:
                self.pot_history.append(self.pot)
        new_state = self.all_game_states[self.all_game_states.index(self.game_state) + 1 % len(self.all_game_states)]
        print(f"Game state transitioned from '{self.game_state}' to '{new_state}'")
        self.game_state = new_state

        if new_state in ["Flop", "Turn", "River"]:
            self.current_player = self.get_next_player(self.dealer)
            self.trans_player = self.dealer


        if new_state == "Pre-flop":
            self.deal_hands()
            #self.do_one_round()

        if new_state == "Flop":
            self.deal_table(3)
            #self.do_one_round()
        
        if new_state == "Turn" or new_state == "River":
            self.deal_table(1)
            #self.do_one_round()
        
        if new_state == "Showdown":
           new_state = "Conclusion" 
            
        if new_state == "Conclusion":
            self.winner, winner_hand, winner_hand_rank = self.get_winner()
            
            for key in list(self.active_player_list.keys()):
                print(f"Player {key} hand: {self.active_player_list[key].hand}")
            print(f"Table: {self.cards_on_table}")
            print(f"Winner: Player {self.winner.player_id} with {winner_hand} (rank {winner_hand_rank})")
            self.winner.add_to_balance(self.pot)
            self.game_over()

        
        
    def deal_hands(self):
        print(f"Dealing hands...")
        for player_id in list(self.player_list.keys()):
            player = self.player_list[player_id]
            if not player.balance <= 0.01:
                self.active_player_list[player_id] = player
                player.set_hand(self.table.deck.draw_cards(2))
                print(f"  P {player_id}: {player.hand}")
        print(f"Hands dealt")
        
    
    def deal_table(self, amount):
        self.cards_on_table += self.table.deck.draw_cards(amount)
    
    def game_over(self):
        self.game_ended = True
        self.return_function()
    
    def record_game(self, game_folder):

        header_str = f""

        for key in list(self.action_map.keys()):
            header_str += f"{key}, "
        header_str = header_str[:-2] + "\n"
        
        csv_file = open(os.path.join(game_folder, f"Actions.csv"), "w")
        csv_file.write(header_str)


        for i in range(len(self.action_map[list(self.action_map.keys())[0]])):
            line_str = f""
            for key in list(self.action_map.keys()):
                if len(self.action_map[key]) > i:
                    action = self.action_map[key][i]
                    line_str += f"[{action.player_id};{action.action_str};{action.bet_amount}],"
                else:
                    line_str += f"[],"
            line_str = line_str[:-1] + "\n"
            csv_file.write(line_str)
     
        for i in range(len(self.action_map[list(self.action_map.keys())[0]])):
            line_str = f""
            for key in list(self.action_map.keys()):
                if len(self.action_map[key]) > i:
                    action = self.action_map[key][i]
                    line_str += f"[{action.player_id};{action.action_str};{action.bet_amount}],"
                else:
                    line_str += f"[],"
            line_str = line_str[:-1] + "\n"
            csv_file.write(line_str)
        
        
        csv_file.close()


    def __repr__(self) -> str:
        return_str = f"Game {self.game_id} (D: {self.dealer}, C: {self.current_player}, T: {self.trans_player})\n  Number of players: {len(self.player_list)}\n  Game State: {self.game_state}\n  Pot: {self.pot}\n  Pot Hist:\n"
        for pot in self.pot_history:
            return_str += f"    {pot}\n"
        return return_str

