import os, sys, random, math
from .strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.player_action import Player_Action

class GTO_strategy(Strategy): #https://www.pokerprofessor.com/university/how-to-win-at-poker/poker-starting-hands
    def __init__(self) -> None:
        self.group_a = ["AA", "KK", "AKs"]
        self.group_b = ["AK", "QQ"]
        self.group_c = ["JJ", "TT"]
        self.group_d = ["AQ", "AJs", "99", "88"]
        self.group_e = ["AJ", "ATs", "KQs", "77", "66", "55"]
        self.group_f = ["AT", "KQ", "KJs", "QJs", "44", "33", "22"]
        self.group_g = ["A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "KTs", "QTs", "JTs", "J9s", "T9s", "98s"]
        self.group_h = ["KJ", "KT", "QJ", "J8s", "T8s", "87s", "76s"]
        self.groups = [self.group_a, self.group_b, self.group_c, self.group_d, self.group_e, self.group_f, self.group_g, self.group_h]
        self.group_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
        
        self.rank_map = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}

        self.action_map = {"Unraised": 
                             {"Early": 
                               {"Raise": ["A", "B", "C", "D"],
                                "Check": ["E", "F", "G", "H"],
                                "Fold": [],
                                "Call": []},
                              "Mid":
                               {"Raise": ["A", "B", "C", "D", "E"],
                                "Check": ["F", "G", "H"],
                                "Fold": [],
                                "Call": []},
                              "Late": 
                               {"Raise": ["A", "B", "C", "D", "E", "F"],
                                "Check": ["G", "H"],
                                "Fold": [],
                                "Call": []}},
                            "Raised":
                             {"Early": 
                               {"Raise": ["A", "B"],
                                "Call": ["C"],
                                "Fold": ["D", "E", "F", "G", "H"],
                                "Check": []},
                              "Mid":
                               {"Raise": ["A", "B"],
                                "Call": ["C"],
                                "Fold": ["D", "E", "F", "G", "H"],
                                "Check": []},
                              "Late": 
                               {"Raise": ["A", "B"],
                                "Call": ["C", "D"],
                                "Fold": ["E", "F", "G", "H"],
                                "Check": []}},
                            "Blinds":
                             {"Early": 
                               {"Raise": ["A"],
                                "Call": ["B", "C", "D"],
                                "Fold": ["E", "F", "G", "H"],
                                "Check": []},
                              "Mid":
                               {"Raise": ["A", "B", "C"],
                                "Call": ["D", "E"],
                                "Fold": ["F", "G", "H"],
                                "Check": []},
                              "Late": 
                               {"Raise": ["A", "B", "C", "D"],
                                "Call": ["E", "F"],
                                "Fold": ["G", "H"],
                                "Check": []}
                             }
                            }

    def get_player(self, table, player_id):
        return table.seated_players[player_id]

    def format_rank(self, rank):
        if rank > 9:
            return self.rank_map[rank]
        return str(rank)
    
    def get_hand_group(self, player):
        hand = player.hand
        suited = hand[0].current_suit == hand[1].current_suit

        formated_hand_str = ""
        formated_hand_str_2 = ""

        if hand[0].current_rank > hand[1].current_rank:
            formated_hand_str += self.format_rank(hand[0].current_rank)
            formated_hand_str += self.format_rank(hand[1].current_rank)
        else:
            formated_hand_str += self.format_rank(hand[1].current_rank)
            formated_hand_str += self.format_rank(hand[0].current_rank)
        
        res_group = -1
        if suited:
            formated_hand_str_2 = formated_hand_str + "s"
        else:
            formated_hand_str_2 = formated_hand_str
        

        for i, group in enumerate(self.groups):
            if formated_hand_str in group or formated_hand_str_2 in group:
                res_group = i
                break
        
        return res_group
    
    def get_situation(self, table, player_id):
        curr_game = table.current_game
        if not curr_game.somebody_raised: return "Unraised"
        if player_id in curr_game.blinds:
            return "Blinds"
        return "Raised"

    def get_position(self, table, player_id):
        curr_game = table.current_game
        player_amount = len(list(curr_game.active_player_list.keys()))

        if curr_game.game_state == "Pre-flop":
            curr_pos = curr_game.get_next_active_player(curr_game.blinds[1]).player_id
        else:
            curr_pos = curr_game.get_next_active_player(curr_game.dealer).player_id
        
        if player_amount % 3 == 2:
            early_pos = math.ceil(player_amount / 3)
            mid_pos = early_pos * 2
        elif player_amount % 3 == 1:
            early_pos = math.floor(player_amount / 3)
            mid_pos = (early_pos * 2) + 1
        else:
            early_pos = round(player_amount / 3)
            mid_pos = early_pos * 2
        
        counter = 1
        for i in range(len(list(curr_game.active_player_list.keys()))):
            if curr_pos == player_id:
                if counter <= early_pos:
                    return "Early"
                if counter <= mid_pos:
                    return "Mid"
                return "Late"
            curr_pos = curr_game.get_next_active_player(curr_pos).player_id
            counter += 1
        
            


    def compute_action(self, table, player_id: int, max_currently_on_table) -> Player_Action:
        if super().compute_action(table, player_id) == "NoAction": 
            return None
        player = self.get_player(table, player_id)
        group = self.get_hand_group(player)
        if group == -1:
            return Player_Action(table, player_id, "Fold", 0.0)
        situation = self.get_situation(table, player_id)
        position = self.get_position(table, player_id)
        actions = self.action_map[situation][position]
        res_action_str = None
        for action in list(actions.keys()):
            if self.group_map[group] in actions[action]:
                res_action_str = action
        
        if res_action_str == "Raise":
            bet_amount = self.compute_bet_amount(table, position, player.balance, max_currently_on_table - player.current_money_on_table)
        else:
            bet_amount = 0        

        if res_action_str is None:
            return None

        res_action = Player_Action(table, player_id, res_action_str, bet_amount)
        return res_action

    def compute_bet_amount(self, table, position, bal, minimum):
        bb = table.current_game.blinds_amount[1]
        res = 0
        if position == "Early":
            res = round(bb * 4, 2)
        if position == "Mid":
            res = round(bb * 3.5, 2)
        if position == "Late":
            res = round(bb * 3, 2)
        res += minimum
        bet_amount = min(res, bal)
        return bet_amount