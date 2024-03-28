from typing import Any
from .player import Player
from .card import Card
import numpy as np
from .hand_evaluator import Hand_Evaluator
from functools import cmp_to_key
import os
import csv

class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, table, start_balance: int = None, save_game = False, blinds_amount = [0.01, 0.02]) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.initial_balances = {}
        for p_id in list(self.player_list.keys()):
            self.initial_balances[p_id] = self.player_list[p_id].balance
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

        self.all_in_players = []

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

        print([p.player_id for p in players])
        print(f"Player_groups: ", player_groups)
        if len([pg for pg in player_groups if len(pg) > 1]) > 0:
            print(self.cards_on_table)
            for p in players:
                print(f"Player {p.player_id} hand: {p.hand}")
                print(self.hand_evaluator.get_hand_result(p.hand))
                print()
            input()


        for player_id in list(self.active_player_list.keys()):
            curr_player = self.active_player_list[player_id]
            (hand_str, hand_secondary_rank), kicker, hand_primary_secondary_rank = self.hand_evaluator.compute_hand(curr_player.hand, self.cards_on_table)
            player_results[player_id] = [hand_str, hand_secondary_rank, kicker, hand_primary_secondary_rank]

        """
        for p_id in list(player_results.keys()):    #Godt for debugging
            p_str = f"Player {p_id}: '{player_results[p_id][0]}' - Rank: {player_results[p_id][1]}"
            if player_results[p_id][3] is not None:
                p_str += f" ({player_results[p_id][3]})"
            p_str += f" - Kicker: {player_results[p_id][2]}"
            print(p_str)
        """

        sorted_players = dict(sorted(player_results.items(), key=lambda x: self.rank_list[x[1][0]], reverse=True))
        highest_hand_str = sorted_players[list(sorted_players.keys())[0]][0]
        possible_winners = {k: v for k, v in sorted_players.items() if sorted_players[k][0] == highest_hand_str}

        if len(list(possible_winners.keys())) == 1:
            winning_id = list(possible_winners.keys())[0]
            return [[self.active_player_list[winning_id], possible_winners[winning_id][0], possible_winners[winning_id][1]]]
        
        sorted_possible_winners = dict(sorted(possible_winners.items(), key=lambda x: x[1][1], reverse=True))
        highest_hand_primary_rank = sorted_possible_winners[list(sorted_possible_winners.keys())[0]][1]

        possible_rank_winners = {k: v for k, v in sorted_possible_winners.items() if sorted_possible_winners[k][1] == highest_hand_primary_rank}

        if len(list(possible_rank_winners.keys())) == 1:
            winning_id = list(possible_rank_winners.keys())[0]
            return [[self.active_player_list[winning_id], possible_rank_winners[winning_id][0], possible_rank_winners[winning_id][1]]]
        
        new_possible_winners = possible_rank_winners

        if highest_hand_str == "Two Pairs" or highest_hand_str == "Full House":
            sorted_edgecase_winners = dict(sorted(possible_rank_winners.items(), key=lambda x: x[1][3], reverse=True))
            highest_hand_primary_secondary_rank = sorted_edgecase_winners[list(sorted_edgecase_winners.keys())[0]][3]
            new_possible_winners = {k: v for k, v in sorted_edgecase_winners.items() if sorted_edgecase_winners[k][3] == highest_hand_primary_secondary_rank}
            if len(list(new_possible_winners.keys())) == 1:
                winning_id = list(new_possible_winners.keys())[0]
                return [[self.active_player_list[winning_id], new_possible_winners[winning_id][0], new_possible_winners[winning_id][1]]]

        p_ids = list(new_possible_winners.keys())
        kickers = len(new_possible_winners[p_ids[0]][2])

        if kickers == 0:
            return [[self.active_player_list[winning_id], new_possible_winners[winning_id][0], new_possible_winners[winning_id][1]] for winning_id in list(new_possible_winners.keys())]

        highest_kicker = 0
        highest_kicker_pid = []

        for i in range(kickers):
            highest_kicker = 0
            highest_kicker_pid = []
            for p_id in p_ids:
                curr_kick = new_possible_winners[p_id][2][i]
                if curr_kick > highest_kicker:
                    highest_kicker = curr_kick
                    highest_kicker_pid = [p_id]
                elif curr_kick == highest_kicker:
                    highest_kicker_pid.append(p_id)
    
            if len(highest_kicker_pid) == 1:
                break

        winners = {k: v for k, v in new_possible_winners.items() if k in highest_kicker_pid}

        return [[self.active_player_list[winning_id], winners[winning_id][0], winners[winning_id][1]] for winning_id in list(winners.keys())]


    
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
        print(f"Game state transitioned from '{self.game_state}' to '{new_state}'")
        self.game_state = new_state

        if new_state in ["Flop", "Turn", "River"]:
            self.current_player = self.get_player_after_dealer()
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
            winners = self.get_winner()

            print(f"Winners:")
            print(winners)
            print(f"Side pots: {self.side_pots}")
            print(f"total pot: {self.pot}")

            for winner_group in winners:
                is_all_in = []
                is_not_all_in = []
                for winner in winner_group:
                    if winner.all_in:
                        is_all_in.append(winner)
                    else:
                        is_not_all_in.append(winner)

                for player in is_all_in:
                    to_add = self.side_pots[player.player_id] / len(winner_group)
                    print(f"Adding {to_add} to player {player.player_id}")
                    player.add_to_balance(to_add)
                    self.pot -= to_add

                for winner in is_not_all_in:
                    to_add = self.pot / len(is_not_all_in)
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
        for player_id in list(self.player_list.keys()):
            player = self.player_list[player_id]
            if not player.balance <= 0.01:
                player.folded = False
                player.all_in = False
                self.active_player_list[player_id] = player
                player.set_hand(self.table.deck.draw_cards(2))
                print(f"  P {player_id}: {player.hand}")
        #print(f"Hands dealt")
        
    
    def deal_table(self, amount):
        c = self.table.deck.draw_cards(amount)
        print(f"Dealing on table - {c}")
        self.cards_on_table += c
    
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

        if len(self.action_map[list(self.action_map.keys())[0]]) == 0:
            input("HHHHHH")







    def __repr__(self) -> str:
        return_str = f"Game {self.game_id} (D: {self.dealer}, C: {self.current_player}, T: {self.trans_player})\n  Number of players: {len(self.player_list)}\n  Game State: {self.game_state}\n  Pot: {self.pot}\n  Pot Hist:\n"
        for pot in self.pot_history:
            return_str += f"    {pot}\n"
        return return_str

