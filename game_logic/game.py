from typing import Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.player import Player
from game_logic.card import Card
import numpy as np
from game_logic.hand_evaluator import Hand_Evaluator
from functools import cmp_to_key
from Input.statistics import PokerStatistics


class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, table, start_balance: int = None, save_game = False, blinds_amount = [0.01, 0.02]) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.initial_balances = {}
        for p_id in list(self.player_list.keys()):
            self.initial_balances[p_id] = float(self.player_list[p_id].balance)
        self.active_player_list = {}
        
        self.rank_list = {"Royal Flush": 10, "Straight Flush": 9, "Four of a Kind": 8, "Full House": 7, "Flush": 6, "Straight": 5, "Three of a Kind": 4, "Two Pairs": 3, "One Pair": 2, "High Card": 1}
        
        self.pot = 0
        self.current_pot = 0
        self.side_pots = {}
        self.pot_history = []
        self.all_game_states = ["Pre-round", "Pre-flop", "Flop", "Turn", "River", "Showdown", "Conclusion"]
        self.game_state = "Pre-round"
        self.cards_on_table: list[Card] = []
        self.table = table
        self.return_function = return_function
        self.game_ended = False
        self.blinds_amount = blinds_amount
        self.dealer = np.random.choice(list(self.player_list.keys()))
        self.blinds: list[int] = [(self.dealer + 1) % len(self.player_list), (self.dealer + 2) % len(self.player_list)]
        self.current_player: int = self.get_next_player(self.get_next_player(self.get_next_player(self.dealer, use_standard_player_list=True), use_standard_player_list=True), use_standard_player_list=True)
        self.trans_player:int = self.get_next_player(self.get_next_player(self.dealer, use_standard_player_list=True), use_standard_player_list=True)
        self.hand_evaluator = Hand_Evaluator()
        self.save_game = save_game
        self.active_side_pots = []
        self.somebody_raised = False
        self.action_map = {"Pre-flop": [], "Flop": [], "Turn": [], "River": []}
        self.winner_arr = []
        self.stats = PokerStatistics(self.player_list)
        self.all_in_players = []
        self.log_str = f""

        self.transition_state()
        
    
    def do_one_round(self):
        #print(f"Starting this round: {self.current_player}")
        #print(f"Ending this round: {self.trans_player}\n")

        while self.current_player != self.trans_player:
            action_performed = self.player_performed_action()

        action_performed = self.player_performed_action()
    
    def get_player_after_dealer(self):
        player_list = list(self.player_list.keys())

        res_player = None
        counter = 0
        curr_index = player_list.index(self.dealer)

        curr_player = self.dealer

        while res_player is None:
            curr_index = (curr_index + 1) % len(player_list)
            if player_list[curr_index] in list(self.active_player_list.keys()):
                return player_list[curr_index]
            counter += 1

            if counter == len(player_list):
                print(f"NO PLAYERS!!!!!!!")
                print((self.active_player_list.keys()))
                input("")


    def get_next_active_player(self, id):
        print(list(self.player_list.keys()))
        lst = [0, 1, 2, 3, 4, 5]
        curr_idx = (lst.index(id) + 1) % len(lst)
        counter = 0
        while curr_idx not in list(self.active_player_list.keys()):
            curr_idx = (curr_idx + 1) % len(lst)
            if counter > len(lst):
                print(f"COULD NOT GET NEXT ACTIVE PLAYER!!")
                print(f"TRYING TO GET PLAYER AFTER PLAYER {id}")
                print(f"PLAYER_LIST: {self.player_list}")
                print(f"ACTIVE_PLAYER_LIST: {self.active_player_list}")
                input(f"[ERROR]...")
            counter += 1

        return self.active_player_list[curr_idx]




    def get_next_player(self, id = None, use_standard_player_list = False, reverse = False):
        player_list = list(self.active_player_list.keys())
        player_dict = self.active_player_list
        if use_standard_player_list:
            player_dict = self.player_list
            player_list = list(self.player_list.keys())
        
        if reverse: keyword = "before"
        else: keyword = "after"

        if id is None:
            #print(f"Finding player {keyword}: {self.current_player} - ", end="")
            curr_idx = player_list.index(self.current_player)
        else:
            #print(f"Finding player {keyword}: {id} - ", end="")
            curr_idx = player_list.index(id)
        #print(player_list)
        
        if reverse:
            res_player = player_list[(curr_idx - 1) % len(player_list)]
        else:
            res_player = player_list[(curr_idx + 1) % len(player_list)]

        #print(f"Found {res_player}")
        return res_player


    def create_side_pot(self, player_id, amount):
        self.active_side_pots.append(player_id)
        self.side_pots[player_id] = self.pot
        for p_id in list(self.player_list.keys()):
            self.side_pots[player_id] = round(self.side_pots[player_id] + min(self.player_list[p_id].current_money_on_table, amount), 2)
            #self.side_pots[player_id] += min(self.player_list[p_id].current_money_on_table, amount)
    
    def update_active_side_pots(self, amount):
        for p_id in self.active_side_pots:
            if p_id not in list(self.side_pots.keys()):
                self.side_pots[p_id] = round(self.pot + amount, 2)
                
            else:
                self.side_pots[p_id] = round(self.side_pots[p_id] + min(amount, self.player_list[p_id].current_money_on_table), 2)
                #self.side_pots[p_id] += min(amount, self.player_list[p_id].current_money_on_table)
        #if len(self.active_side_pots) > 0:
        #    input(f"Sidepots?")
    

    def player_performed_action(self):
        player_id = self.current_player
        #print(f"PLAYER TO PERFORM ACTION: {player_id}")
        next_player = self.get_next_player(player_id)
        previous_player = self.get_next_player(player_id, reverse=True)
        #print(f"Game state: {self.game_state}")
        
        if self.game_state == "Showdown":
            return None
        
        max_currently_on_table = max([self.player_list[p_id].current_money_on_table for p_id in self.player_list])

        action = self.active_player_list[player_id].perform_action(self.somebody_raised, max_currently_on_table)
        if action is None:
            self.current_player = next_player
            return None

        self.log_str += f"Player {player_id} performed action {action.action_str} ($ {action.bet_amount})\n"
        print(f"Player {player_id} performed action {action.action_str} ($ {action.bet_amount})")

        if self.active_player_list[player_id].folded:
            self.active_player_list.pop(player_id)
        
        
        if self.game_state in list(self.action_map.keys()):
            self.action_map[self.game_state].append(action)
        
        if action.action_str == "Raise" or action.action_str == "Call":
            if self.active_player_list[player_id].all_in:
                self.create_side_pot(player_id, action.bet_amount)
                self.all_in_players.append(self.active_player_list.pop(player_id))
            else:
                self.update_active_side_pots(action.bet_amount)
            self.current_pot = round(self.current_pot + action.bet_amount, 2)

        if action.action_str == "Raise":
            self.somebody_raised = True
            self.trans_player = previous_player
            self.log_str += f"New Trans player: {self.trans_player} (in list: {list(self.active_player_list.keys())})\n"
            print(f"New Trans player: {self.trans_player} (in list: {list(self.active_player_list.keys())})")


        if len(list(self.active_player_list.keys())) == 1 and self.player_list[player_id].folded:
            self.transition_state(showdown=True)
            return action

        if player_id == self.trans_player:
            self.transition_state()
        else:
            self.current_player = next_player

        return action

    def get_winner(self) -> tuple[Player, str, int]:

        #Håndends primære rank er Royal Flush = 10, Straight Flush = 9, osv.
        #En straigh med tallene 4, 5, 6, 7, 8 har primær rank 5, fordi self.rank_list["Straight"] = 5,
        #og den har sekundær rank 8, fordi 8 er det højeste tal i straighten
        highest_hand_primary_rank = 0
        highest_hand_secondary_rank = 0
        highest_hand_str = ""
        highest_hand_primary_secondary_rank = 0
        winner = None

        player_results = {} #fx. player_results[0] = ["One Pair", 11, [14, 12, 9], None]
        
        all_in_players = [self.player_list[p_id] for p_id in list(self.player_list.keys()) if self.player_list[p_id].all_in]
        
        players = all_in_players + [self.active_player_list[p] for p in list(self.active_player_list.keys())]

        self.hand_evaluator.set_cards_on_table(self.cards_on_table)
        #print([p.player_id for p in players])
        players = sorted(players, key=cmp_to_key(self.hand_evaluator.compare_players))
        player_groups = [[players[0]]]

        for p in players[1:]:
            compare_res = self.hand_evaluator.compare_players(p, player_groups[-1][-1])
            if compare_res == 0:
                player_groups[-1].append(p)
            else:
                player_groups.append([p])

        return player_groups

    def transition_state(self, showdown = False):
        self.somebody_raised = False
        self.pot = round(self.pot + self.current_pot, 2)
        self.current_pot = 0
        for p_id in list(self.player_list.keys()):
            self.player_list[p_id].new_round()
        self.active_side_pots = []
        if 5 > self.all_game_states.index(self.game_state) and self.all_game_states.index(self.game_state) > 0:
            self.pot_history.append(self.pot)
        if showdown or (len(list(self.active_player_list.keys())) <= 1 and self.game_state != "Pre-round"):
            new_state = "Showdown"
        else:
            new_state = self.all_game_states[self.all_game_states.index(self.game_state) + 1 % len(self.all_game_states)]
        self.log_str += f"Game state transitioned from '{self.game_state}' to '{new_state}'\n"
        print(f"Game state transitioned from '{self.game_state}' to '{new_state}'")
        self.game_state = new_state

        if new_state in ["Flop", "Turn", "River"]:
            self.current_player = self.get_player_after_dealer()
            self.trans_player = self.dealer
        player_list = []
        for _,player in self.active_player_list.items():
            player_list.append(player)
        
        self.stats.update_pot_odds(self.pot + self.current_pot, player_list)
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
            self.winner_arr = self.get_winner()

            print(f"Winners:")
            print(self.winner_arr)
            print(f"Side pots: {self.side_pots}")
            print(f"Total pot: {self.pot}")

            self.log_str += f"Winners:\n{self.winner_arr}\nSide pots: {self.side_pots}\nTotal pot: {self.pot}\n"

            for winner_group in self.winner_arr:
                is_all_in = []
                is_not_all_in = []
                for winner in winner_group:
                    if winner.all_in:
                        is_all_in.append(winner)
                    else:
                        is_not_all_in.append(winner)

                for player in is_all_in:
                    to_add = min(self.side_pots[player.player_id] / len(winner_group), self.pot)
                    self.log_str += f"Adding {to_add} to player {player.player_id}\n"
                    print(f"Adding {to_add} to player {player.player_id}")
                    player.add_to_balance(to_add)
                    self.pot -= to_add
                    if self.pot == 0:
                        break
                    
                if self.pot == 0:
                    break

                for winner in is_not_all_in:
                    to_add = min(self.pot / len(is_not_all_in), self.pot)
                    self.log_str += f"Adding {to_add} to player {winner.player_id}\n"
                    print(f"Adding {to_add} to player {winner.player_id}")
                    winner.add_to_balance(to_add)

                if len(is_not_all_in) > 0:
                    break
            print()

            


            #input()
            #for winner in winners:
            #    print(f"  Player {winner[0].player_id}")
            #    #print(f"    {winner[1]} ({winner[2]})")
            #    winner[0].add_to_balance(round(self.pot / len(winners), 2))
            #    self.winner_arr.append(winner[0])

            self.game_over()

        
        
    def deal_hands(self):
        print(f"Dealing hands...")
        self.log_str += f"Dealing hands...\n"
        for player_id in list(self.player_list.keys()):
            player = self.player_list[player_id]
            if not player.balance <= 0.01:
                player.folded = False
                player.all_in = False
                self.active_player_list[player_id] = player
                c = self.table.deck.draw_cards(2)
                player.set_hand(c)
                #for card in c:
                #    self.stats.update_pov_count(card.current_rank, player.player_id)
                self.log_str += f"  P {player_id}: {player.hand}\n"
                print(f"  P {player_id}: {player.hand}")
        #print(f"Hands dealt")
        
    
    def deal_table(self, amount):
        c = self.table.deck.draw_cards(amount)
        for card in c:
            self.stats.update_true_count(card.current_rank)
        for player in self.active_player_list:
            for card in c:
                self.stats.update_pov_count(card.current_rank, player)
        print(f"Dealing on table - {c}")
        self.log_str += f"Dealing on table - {c}\n"
        self.cards_on_table += c
    
    def game_over(self):
        self.game_ended = True
        self.stats.print_stats()
        #input()
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
        
        csv_file.close()


        header_str = f""
        bal_str = f""
        for p_id in list(self.initial_balances.keys()):
            header_str += f"P {p_id}, "
            bal_str += f"{self.initial_balances[p_id]}, "
        header_str = header_str[:-2] + "\n"
        bal_str = bal_str[:-2]

        csv_file = open(os.path.join(game_folder, f"InitBals.csv"), "w")
        csv_file.write(header_str)
        csv_file.write(bal_str)

        csv_file.close()

        bal_after_str = f""
        header_after_str = f""
        for p_id in list(self.player_list.keys()):
            header_after_str += f"P {p_id}, "
            bal_after_str += f"{self.player_list[p_id].balance}, "
        header_after_str = header_after_str[:-2] + "\n"
        bal_after_str = bal_after_str[:-2]

        csv_file = open(os.path.join(game_folder, f"PostgameBals.csv"), "w")
        csv_file.write(header_after_str)
        csv_file.write(bal_after_str)

        csv_file.close()

        csv_file = open(os.path.join(game_folder, f"Winners.csv"), "w")
        csv_file.write(str(self.winner_arr))

        csv_file.close()

        cards_str = f""

        for p_id in list(self.player_list.keys()):
            cards_str += f"P {p_id}: {self.player_list[p_id].hand}\n"
        cards_str += f"\n{self.cards_on_table}\n"
        csv_file = open(os.path.join(game_folder, f"Cards.csv"), "w")
        csv_file.write(cards_str)

        csv_file.close()

        log_file = open(os.path.join(game_folder, f"log.txt"), "w")
        log_file.write(self.log_str)
        log_file.close()

        meta_file = open(os.path.join(game_folder, f"metadata.txt"), "w")
        meta_file.write(f"Dealer: {self.dealer}")
        meta_file.close()

        

        if len(self.action_map[list(self.action_map.keys())[0]]) == 0:
            input("HHHHHH")







    def __repr__(self) -> str:
        return_str = f"Game {self.game_id} (D: {self.dealer}, C: {self.current_player}, T: {self.trans_player})\n  Number of players: {len(self.player_list)}\n  Game State: {self.game_state}\n  Pot: {self.pot}\n  Pot Hist:\n"
        for pot in self.pot_history:
            return_str += f"    {pot}\n"
        return return_str

