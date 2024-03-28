class PokerStatistics():
    def __init__(self):
        self.true_count = 0
        self.pov_count = []             # The count as observed by each player
        self.pov_blockers = [[]]        # Blockers are the cards in your hand that reduce the chance of your opponents holding a specific combination of cards.
                                        # We do it in a 2D-array since we need a list of cards for each player
    

    def update_count(self, card_rank):
        if card_rank >= 2 and card_rank <= 6:
            self.true_count += 1
        elif card_rank >= 10 and card_rank <= 14:
            self.true_count -= 1
        # The else case here would change the count by 0 so we dont need to implement it

    def reset_count(self):
        self.true_count = 0
        self.pov_count = []