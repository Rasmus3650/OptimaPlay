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
            print(player)
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
        for _, player in self.active_player_list.items():
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
        
        for _, player in self.active_player_list.items():
            self.compute_hand(player.hand, self.cards_on_table)
    
    def compute_straight_flush(self, hand, card_on_table, hand_suits: list[str], table_suits: list[str], royal = False) -> tuple[str, int]:
        hand_fitted = len(np.unique(hand_suits)) == 1
        matching_cards = []
        if (hand_fitted and table_suits.count(hand_suits[0]) >= 3) or (table_suits.count(hand_suits[0]) >= 4 or table_suits.count(hand_suits[1]) >= 4) or (len(np.unique(table_suits)) == 1): # Royal Flush
            if hand_fitted:
                matching_cards.append(hand[0])
                matching_cards.append(hand[1])
                for card in card_on_table:
                    if card.current_suit == hand_suits[0]:
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
            if royal:
                if 10 in rank_list and 11 in rank_list and 12 in rank_list and 13 in rank_list and 14 in rank_list:
                    return True, ("Royal Flush", 14)
            else:
                start_rank = None
                straight_list = []
                matching_cards.sort(key=lambda x: x.current_rank)
                
                for card in matching_cards:
                    if start_rank == None or (len(straight_list) > 0 and card.current_rank != straight_list[-1].current_rank + 1):
                        start_rank = card.current_rank
                        straight_list = [card]
                    elif len(straight_list) > 0 and card.current_rank == (straight_list[-1].current_rank) + 1:
                        straight_list.append(card)
                        if len(straight_list) == 5:
                            return True, ("Straight Flush", card.current_rank)

        return False, ("", 0)
    
    def compute_four_of_a_kind(self, hand, card_on_table, hand_ranks: list[str], table_ranks: list[str]):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #(rank, all_ranks.count(rank))

        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        for rank_tup in rank_tuples:
            if rank_tup[1] >= 4:
                return True, ("Four of a kind", rank_tup[0])  # Hvis der er 2 four of a kinds skal vi finde den største
        
        return False, ("", 0)
    
    def compute_full_house(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #(rank, all_ranks.count(rank))

        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        three_of_a_kind = False
        pair = False
        highest_rank = 0

        for rank_tup in rank_tuples:
            if rank_tup[1] >= 3:
                if not three_of_a_kind: 
                    if rank_tup[0] > highest_rank:
                        highest_rank = rank_tup[0]
                    three_of_a_kind = True
                else:
                    pair = True
            elif rank_tup[1] >= 2:
                if rank_tup[0] > highest_rank:
                    highest_rank = rank_tup[0]
                pair = True
        
        if three_of_a_kind and pair:
            return True, ("Full House", highest_rank)
        else:
            return False, ("", 0)
    

    def compute_flush(self, hand, card_on_table, hand_suits, table_suits):
        all_cards = hand + card_on_table

        suits_map = {"Spades": [0, 0], "Hearts": [0, 0], "Clubs": [0, 0], "Diamonds": [0, 0]}

        for suit in list(suits_map.keys()):
            suits_map[suit][0] += sum(c.current_suit == suit for c in all_cards)
            for c in all_cards:
                if c.current_rank > suits_map[c.current_suit][1]:
                    suits_map[c.current_suit][1] = c.current_rank

        

        flush_found = False
        highest_rank = 0

        
        for suit in list(suits_map.keys()):
            amount, highest = suits_map[suit]
            if amount >= 5:
                if not flush_found:
                    flush_found = True
                    highest_rank = highest
                elif highest > highest_rank:
                    highest_rank = highest
        if flush_found:
            return flush_found, ("Flush", highest_rank)
        else:
            return flush_found, ("", highest_rank)


    def compute_straight(self, hand, card_on_table, hand_ranks, table_ranks):
        all_cards = hand + card_on_table
        sorted_cards = sorted(all_cards, key=lambda x: x.current_rank)


        start_rank = None
        straight_list = []

        for card in sorted_cards:
            if start_rank == None or (len(straight_list) > 0 and (card.current_rank != straight_list[-1].current_rank + 1 and card.current_rank != straight_list[-1].current_rank)):
                start_rank = card.current_rank
                straight_list = [card]
            elif len(straight_list) > 0 and card.current_rank == (straight_list[-1].current_rank) + 1:
                straight_list.append(card)
                if len(straight_list) == 5:
                    return True, ("Straight", card.current_rank)

        return False, ("", 0)
    
    def compute_three_of_a_kind(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #list[(rank, all_ranks.count(rank))]
        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        highest = 0
        found_three = False

        for rank_tup in rank_tuples:
            if rank_tup[1] >= 3:
                found_three = True
                if rank_tup[0] > highest:
                    highest = rank_tup[0]
        
        if found_three:
            return True, ("Three of a kind", highest) 
        else:
            return False, ("", 0)
        
    def compute_pairs(self, hand, card_on_table, hand_ranks, table_ranks, amount_of_pairs):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #list[(rank, all_ranks.count(rank))]
        for rank in all_ranks_unique:
            rank_tuples.append([rank, all_ranks.count(rank)])
        
        pairs = 0
        highest = 0

        for rank_tup_idx in range(len(rank_tuples)):
            rank_tup = rank_tuples[rank_tup_idx]
            if rank_tup[1] >= 2:
                pairs += 1
                rank_tup[1] -= 2
                rank_tup_idx -= 1
                if rank_tup[0] > highest:
                    highest = rank_tup[0]

        if amount_of_pairs == 2:
            if pairs >= 2:
                return True, ("Two Pairs", highest)
            else:
                return False, ("", 0)
        if amount_of_pairs == 1:
            if pairs >= 1:
                return True, ("One Pair", highest)
            else:
                return False, ("", 0)
        

    def compute_high_card(self, hand, card_on_table, hand_ranks, table_ranks):
        return True, ("High Card", max(hand_ranks + table_ranks))
        

    def compute_hand(self, hand: list[Card], card_on_table: list[Card]) -> tuple[str, int]: #fx: ("One Pair", 6) Har et par 6
        print(f"Compute hand called")
        
        # TODO Needs Testing

        #Custom hands/table for testing
        #hand = [Card(12, "Hearts"), Card(2, "Clubs")]
        #card_on_table = [Card(9, "Clubs"), Card(13, "Hearts"), Card(7, "Diamonds"), Card(10, "Hearts"), Card(6, "Hearts")]



        hand_suits = [hand[0].current_suit, hand[1].current_suit]
        table_suits = [elem.current_suit for elem in card_on_table]

        hand_ranks = [hand[0].current_rank, hand[1].current_rank]
        table_ranks = [elem.current_rank for elem in card_on_table]
        
        hand_res = None
        royal_flush, royal_flush_res = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=True)
        if royal_flush:
            print(f"ROYAL FLUSH: {royal_flush}:\n  {royal_flush_res}")
            hand_res = royal_flush_res
            pass #Så behøver vi ikke kalde de andre compute - metoder
        
        straight_flush, straight_flush_res = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=False)
        if straight_flush:
            print(f"STRAIGHT FLUSH: {straight_flush}:\n  {straight_flush_res}")
            if hand_res is None:
                hand_res = straight_flush_res
            pass
        
        four_of_a_kind, four_of_a_kind_res = self.compute_four_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if four_of_a_kind:
            print(f"FOUR OF A KIND: {four_of_a_kind}:\n  {four_of_a_kind_res}")
            if hand_res is None:
                hand_res = four_of_a_kind_res
            pass
        
        full_house, full_house_res = self.compute_full_house(hand, card_on_table, hand_ranks, table_ranks)
        if full_house:
            print(f"FULL HOUSE: {full_house}:\n  {full_house_res}")
            if hand_res is None:
                hand_res = full_house_res
            pass

        flush, flush_res = self.compute_flush(hand, card_on_table, hand_suits, table_suits)
        if flush:
            print(f"FLUSH: {flush}:\n  {flush_res}")
            if hand_res is None:
                hand_res = flush_res
            pass
        
        straight, straight_res = self.compute_straight(hand, card_on_table, hand_ranks, table_ranks)
        if straight:
            print(f"STRAIGHT: {straight}:\n  {straight_res}")
            if hand_res is None:
                hand_res = straight_res
            pass
        
        three_of_a_kind, three_of_a_kind_res = self.compute_three_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if three_of_a_kind:
            print(f"THREE OF A KIND: {three_of_a_kind}:\n  {three_of_a_kind_res}")
            if hand_res is None:
                hand_res = three_of_a_kind_res
            pass
        
        two_pairs, two_pairs_res = self.compute_pairs(hand, card_on_table, hand_ranks, table_ranks, amount_of_pairs=2)
        if two_pairs:
            print(f"TWO PAIRS: {two_pairs}:\n  {two_pairs_res}")
            if hand_res is None:
                hand_res = two_pairs_res
            pass
        
        one_pair, one_pairs_res = self.compute_pairs(hand, card_on_table, hand_ranks, table_ranks, amount_of_pairs=1)
        if one_pair:
            print(f"ONE PAIR: {one_pair}:\n  {one_pairs_res}")
            if hand_res is None:
                hand_res = one_pairs_res
            pass

        high_card, high_card_res = self.compute_high_card(hand, card_on_table, hand_ranks, table_ranks)
        if high_card:
            print(f"HIGH CARD: {high_card}:\n  {high_card_res}")
            if hand_res is None:
                hand_res = high_card_res
            pass
        
        if hand_res is None:
            print(f"ERROR: this shouldn't happen")
        
        
        
        print()
        print(f"Hand: {hand}")
        print(f"Table: {card_on_table}")
        print(f"Result: {hand_res}")
                    


    def transition_state(self):
        if 5 > self.all_game_states.index(self.game_state) and self.all_game_states.index(self.game_state) > 0:
                self.pot_history.append(self.pot)
        new_state = self.all_game_states[self.all_game_states.index(self.game_state) + 1 % len(self.all_game_states)]
        hands_map = {}
        
            
        if new_state == "Pre-flop":
            for _, i in self.active_player_list.items():
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
           new_state = "Conclusion" 
            
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
                self.active_player_list[len(self.active_player_list)] = player
                player.set_hand(self.table.deck.draw_cards(2))
    
    def deal_table(self, amount):
        self.cards_on_table += self.table.deck.draw_cards(amount)
    
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

