import numpy as np
from .card import Card


class Hand_Evaluator():
    def __init__(self) -> None:
        pass
    
    def compute_hand(self, hand: list[Card], card_on_table: list[Card]) -> tuple[str, int]: #fx: ("One Pair", 6) Har et par 6
        
        # TODO Needs Testing

        #Custom hands/table for testing
        #hand = [Card(12, "Hearts"), Card(2, "Clubs")]
        #card_on_table = [Card(9, "Clubs"), Card(13, "Hearts"), Card(7, "Diamonds"), Card(10, "Hearts"), Card(6, "Hearts")]

        hand_suits = [hand[0].current_suit, hand[1].current_suit]
        table_suits = [elem.current_suit for elem in card_on_table]

        hand_ranks = [hand[0].current_rank, hand[1].current_rank]
        table_ranks = [elem.current_rank for elem in card_on_table]
        
        royal_flush, royal_flush_res = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=True)
        if royal_flush:
            #print(f"ROYAL FLUSH: {royal_flush}:\n  {royal_flush_res}")
            return royal_flush_res
            
        straight_flush, straight_flush_res = self.compute_straight_flush(hand, card_on_table, hand_suits, table_suits, royal=False)
        if straight_flush:
            #print(f"STRAIGHT FLUSH: {straight_flush}:\n  {straight_flush_res}")
            return straight_flush_res
        
        four_of_a_kind, four_of_a_kind_res = self.compute_four_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if four_of_a_kind:
            #print(f"FOUR OF A KIND: {four_of_a_kind}:\n  {four_of_a_kind_res}")
            return four_of_a_kind_res
        
        full_house, full_house_res = self.compute_full_house(hand, card_on_table, hand_ranks, table_ranks)
        if full_house:
            #print(f"FULL HOUSE: {full_house}:\n  {full_house_res}")
            return full_house_res

        flush, flush_res = self.compute_flush(hand, card_on_table, hand_suits, table_suits)
        if flush:
            #print(f"FLUSH: {flush}:\n  {flush_res}")
            return flush_res
        
        straight, straight_res = self.compute_straight(hand, card_on_table, hand_ranks, table_ranks)
        if straight:
            #print(f"STRAIGHT: {straight}:\n  {straight_res}")
            return straight_res
        
        three_of_a_kind, three_of_a_kind_res = self.compute_three_of_a_kind(hand, card_on_table, hand_ranks, table_ranks)
        if three_of_a_kind:
            #print(f"THREE OF A KIND: {three_of_a_kind}:\n  {three_of_a_kind_res}")
            return three_of_a_kind_res
        
        two_pairs, two_pairs_res = self.compute_pairs(hand, card_on_table, hand_ranks, table_ranks, amount_of_pairs=2)
        if two_pairs:
            #print(f"TWO PAIRS: {two_pairs}:\n  {two_pairs_res}")
            return two_pairs_res
        
        one_pair, one_pairs_res = self.compute_pairs(hand, card_on_table, hand_ranks, table_ranks, amount_of_pairs=1)
        if one_pair:
            #print(f"ONE PAIR: {one_pair}:\n  {one_pairs_res}")
            return one_pairs_res

        high_card, high_card_res = self.compute_high_card(hand, card_on_table, hand_ranks, table_ranks)
        if high_card:
            #print(f"HIGH CARD: {high_card}:\n  {high_card_res}")
            return high_card_res
        
        print(f"ERROR: this shouldn't happen")
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
                return True, ("Four of a Kind", rank_tup[0])  # Hvis der er 2 four of a kinds skal vi finde den største
        
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
            return True, ("Three of a Kind", highest) 
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

