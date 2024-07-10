import numpy as np
from .card import Card
from .player import Player


class Hand_Evaluator():
    def __init__(self) -> None:
        self.rank_list = {"Royal Flush": 10, "Straight Flush": 9, "Four of a Kind": 8, "Full House": 7, "Flush": 6, "Straight": 5, "Three of a Kind": 4, "Two Pairs": 3, "One Pair": 2, "High Card": 1}
        self.cards_on_table = []

    def set_cards_on_table(self, cards_on_table):
        self.cards_on_table = cards_on_table
    
    def get_hand_result(self, hand):
        return self.compute_hand(hand, self.cards_on_table)
    
    def compare_players(self, player1: Player, player2: Player) -> int:
        return self.compare_hands(player1.hand, player2.hand)

    def compare_hands(self, hand1: list[Card], hand2: list[Card]) -> int:
        (hand1_str, hand1_secondary_rank), kicker1, hand1_primary_secondary_rank = self.compute_hand(hand1, self.cards_on_table)
        (hand2_str, hand2_secondary_rank), kicker2, hand2_primary_secondary_rank = self.compute_hand(hand2, self.cards_on_table)

        return_if_hand1 = -1
        return_if_hand2 = 1
        return_if_equal = 0
        

        if self.rank_list[hand1_str] > self.rank_list[hand2_str]:
            return return_if_hand1
        elif self.rank_list[hand1_str] < self.rank_list[hand2_str]:
            return return_if_hand2
        
        if hand1_secondary_rank > hand2_secondary_rank:
            return return_if_hand1
        elif hand1_secondary_rank < hand2_secondary_rank:
            return return_if_hand2
        
        if hand1_str == "Two Pairs" or hand1_str == "Full House":
            if hand1_primary_secondary_rank > hand2_primary_secondary_rank:
                return return_if_hand1
            elif hand1_primary_secondary_rank < hand2_primary_secondary_rank:
                return return_if_hand2
            
        for k1, k2 in zip(kicker1, kicker2):
            if k1 > k2:
                return return_if_hand1
            elif k1 < k2:
                return return_if_hand2
        
        return return_if_equal
        
    
    def compute_hand(self, hand: list[Card], card_on_table: list[Card]) -> tuple[tuple[str, int], list[int], int]: #fx: '("Two Pairs", 6), [14], 3'      Har et par 6 & et par 3 med kicker 14

        #Custom hands/table for testing
        #hand = [Card(12, "Hearts"), Card(2, "Clubs")]
        #card_on_table = [Card(9, "Clubs"), Card(13, "Hearts"), Card(7, "Diamonds"), Card(10, "Hearts"), Card(6, "Hearts")]

        hand_suits = [hand[0].current_suit, hand[1].current_suit]
        table_suits = [elem.current_suit for elem in card_on_table]

        hand_ranks = [hand[0].current_rank, hand[1].current_rank]
        table_ranks = [elem.current_rank for elem in card_on_table]
        
        royal_flush, royal_flush_res, kicker = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=True)
        if royal_flush:
            return royal_flush_res, kicker, None
            
        straight_flush, straight_flush_res, kicker = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=False)
        if straight_flush:
            return straight_flush_res, kicker, None
        
        four_of_a_kind, four_of_a_kind_res, kicker = self.compute_four_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if four_of_a_kind:
            return four_of_a_kind_res, kicker, None
        
        full_house, full_house_res, kicker, second_rank = self.compute_full_house(hand, card_on_table, hand_ranks, table_ranks)
        if full_house:
            return full_house_res, kicker, second_rank

        flush, flush_res, kicker = self.compute_flush(hand, card_on_table, hand_suits, table_suits)
        if flush:
            return flush_res, kicker, None
        
        straight, straight_res, kicker = self.compute_straight(hand, card_on_table, hand_ranks, table_ranks)
        if straight:
            return straight_res, kicker, None
        
        three_of_a_kind, three_of_a_kind_res, kicker = self.compute_three_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if three_of_a_kind:
            return three_of_a_kind_res, kicker, None
        
        two_pairs, two_pairs_res, kicker, second_rank = self.compute_two_pairs(hand, card_on_table, hand_ranks, table_ranks)
        if two_pairs:
            return two_pairs_res, kicker, second_rank
        
        one_pair, one_pairs_res, kicker = self.compute_one_pair(hand, card_on_table, hand_ranks, table_ranks)
        if one_pair:
            return one_pairs_res, kicker, None

        high_card, high_card_res, kicker = self.compute_high_card(hand, card_on_table, hand_ranks, table_ranks)
        if high_card:
            return high_card_res, kicker, None
        return None

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
                matching_cards.append(hand[0])
                for card in card_on_table:
                    if card.current_suit == hand_suits[0]:
                        matching_cards.append(card)
                        
            elif table_suits.count(hand_suits[1]) >= 4:
                matching_cards.append(hand[1])
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
                    return True, ("Royal Flush", 14), []
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
                            return True, ("Straight Flush", card.current_rank), []

        return False, ("", 0), []

    def compute_four_of_a_kind(self, hand, card_on_table, hand_ranks: list[str], table_ranks: list[str]):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #(rank, all_ranks.count(rank))

        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        for rank_tup in rank_tuples:
            if rank_tup[1] >= 4:
                kicker_amount = 1
                kicker = []
                if rank_tup[1] == 4:
                    remaining_ranks = np.delete(all_ranks_unique, np.where(all_ranks_unique == rank_tup[0])[0])
                else:
                    remaining_ranks = all_ranks_unique
                for rank in sorted(remaining_ranks, reverse=True):
                    if len(kicker) == kicker_amount:
                        break
                    kicker.append(rank)
                return True, ("Four of a Kind", rank_tup[0]), kicker
        
        return False, ("", 0), []
    
    def compute_full_house(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #(rank, all_ranks.count(rank))

        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        three_of_a_kind = False
        pair = False
        highest_rank = 0
        three_of_a_kind_rank = 0
        pair_rank = 0

        for rank_tup in rank_tuples:
            if rank_tup[1] >= 3:
                if not three_of_a_kind: 
                    if rank_tup[0] > highest_rank:
                        highest_rank = rank_tup[0]
                        three_of_a_kind_rank = rank_tup[0]
                    three_of_a_kind = True
                else:
                    pair = True
                    pair_rank = rank_tup[0]
            elif rank_tup[1] >= 2:
                if rank_tup[0] > highest_rank:
                    highest_rank = rank_tup[0]
                    pair_rank = rank_tup[0]
                pair = True
        
        if three_of_a_kind and pair:
            return True, ("Full House", three_of_a_kind_rank), [], pair_rank
        else:
            return False, ("", 0), [], None

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
            return flush_found, ("Flush", highest_rank), []
        else:
            return flush_found, ("", highest_rank), []

    def compute_straight(self, hand, card_on_table, hand_ranks, table_ranks):
        all_cards = hand + card_on_table
        sorted_cards = sorted(all_cards, key=lambda x: x.current_rank, reverse=True)

        for card in sorted_cards:
            if card.current_rank == 14:
                sorted_cards.append(Card(1, card.current_suit))
                break

        
        start_rank = None
        straight_list = []


        for card in sorted_cards:
            if start_rank == None or (len(straight_list) > 0 and (card.current_rank != straight_list[-1].current_rank - 1 and card.current_rank != straight_list[-1].current_rank)):
                start_rank = card.current_rank
                straight_list = [card]
            elif len(straight_list) > 0 and card.current_rank == (straight_list[-1].current_rank) - 1:
                straight_list.append(card)
                if len(straight_list) == 5:
                    return True, ("Straight", start_rank), []

        return False, ("", 0), []

    def compute_three_of_a_kind(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #list[(rank, all_ranks.count(rank))]
        for rank in all_ranks_unique:
            rank_tuples.append((rank, all_ranks.count(rank)))
        
        highest = 0
        found_three = False
        kicker = []

        for rank_tup in rank_tuples:
            if rank_tup[1] >= 3:
                found_three = True
                if rank_tup[0] > highest:
                    kicker_amount = 2
                    kicker = []
                    if rank_tup[1] == 3:
                        remaining_ranks = np.delete(all_ranks_unique, np.where(all_ranks_unique == rank_tup[0])[0])
                    else:
                        remaining_ranks = all_ranks_unique
                    for rank in sorted(remaining_ranks, reverse=True):
                        if len(kicker) == kicker_amount:
                            break
                        kicker.append(rank)
                    highest = rank_tup[0]
        
        if found_three:
            return True, ("Three of a Kind", highest), kicker
        else:
            return False, ("", 0), kicker
        

    def compute_two_pairs(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #list[(rank, all_ranks.count(rank))]
        for rank in all_ranks_unique:
            rank_tuples.append([rank, all_ranks.count(rank)])
        
        pair1_rank = 0
        pair1_amount = 0
        pair2_rank = 0
        pair2_amount = 0

        for rank_tup_idx in range(len(rank_tuples)):
            rank_tup = rank_tuples[rank_tup_idx]
            if rank_tup[1] >= 2:
                if rank_tup[0] > pair1_rank:
                    pair1_rank = rank_tup[0]
                    pair1_amount = rank_tup[1]
        
        for rank_tup_idx in range(len(rank_tuples)):
            rank_tup = rank_tuples[rank_tup_idx]
            if rank_tup[0] == pair1_rank:
                rank_tup[1] -= 2
            if rank_tup[1] >= 2:
                if rank_tup[0] > pair2_rank:
                    pair2_rank = rank_tup[0]
                    pair2_amount = rank_tup[1]

        if pair1_rank == 0 or pair2_rank == 0:
            return False, ("", 0), [], None

        

        if pair1_amount > 2:
            remaining_ranks = all_ranks_unique
        elif pair1_amount == 2:
            remaining_ranks = np.delete(all_ranks_unique, np.where(all_ranks_unique == pair1_rank)[0])
        
        if pair2_amount == 2:
            remaining_ranks = np.delete(remaining_ranks, np.where(remaining_ranks == pair2_rank)[0])
        
        kicker_amount = 1
        kicker = []
        
        for rank in sorted(remaining_ranks, reverse=True):
            if len(kicker) == kicker_amount:
                break
            kicker.append(rank)

        return True, ("Two Pairs", max(pair1_rank, pair2_rank)), kicker, min(pair1_rank, pair2_rank)
    

    def compute_one_pair(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = hand_ranks + table_ranks
        all_ranks_unique = np.unique(all_ranks)
        
        
        rank_tuples = []   #list[(rank, all_ranks.count(rank))]
        for rank in all_ranks_unique:
            rank_tuples.append([rank, all_ranks.count(rank)])
        
        highest = 0
        highest_count = 0

        for rank_tup_idx in range(len(rank_tuples)):
            rank_tup = rank_tuples[rank_tup_idx]
            if rank_tup[1] >= 2:
                if rank_tup[0] > highest:
                    highest = rank_tup[0]
                    highest_count = rank_tup[1]

        if highest == 0:
            return False, ("", 0), []


        if highest_count > 2:
            remaining_ranks = all_ranks_unique
        elif highest_count == 2:
            remaining_ranks = np.delete(all_ranks_unique, np.where(all_ranks_unique == highest)[0])
        
        kicker = []
        kicker_amount = 3

        for rank in sorted(remaining_ranks, reverse=True):
            if len(kicker) == kicker_amount:
                break
            kicker.append(rank)

        return True, ("One Pair", highest), kicker


    def compute_high_card(self, hand, card_on_table, hand_ranks, table_ranks):
        all_ranks = sorted(hand_ranks + table_ranks, reverse=True)

        kicker_amount = 4
        kicker = []

        remaining_ranks = all_ranks[1:]

        for rank in sorted(remaining_ranks, reverse=True):
            if len(kicker) == kicker_amount:
                break
            kicker.append(rank)

        return True, ("High Card", all_ranks[0]), kicker

