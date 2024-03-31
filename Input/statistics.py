import numpy as np
class PokerStatistics():
    def __init__(self, num_of_players: int):
        self.true_count = 0
        self.pov_count = np.zeros(6)
        self.pov_blockers = [[] for _ in range(num_of_players)]        # Blockers are the cards in your hand that reduce the chance of your opponents holding a specific combination of cards.
                                        # We do it in a 2D-array since we need a list of cards for each player
    
    def update_pov_count(self, card_rank, player_id):
        print(len(self.pov_count))
        if card_rank >= 2 and card_rank <= 6:
            self.pov_count[player_id] += 1
        elif card_rank >= 10 and card_rank <= 14:
            self.pov_count[player_id] -= 1


    def update_true_count(self, card_rank):
        if card_rank >= 2 and card_rank <= 6:
            self.true_count += 1
        elif card_rank >= 10 and card_rank <= 14:
            self.true_count -= 1
        # The else case here would change the count by 0 so we dont need to implement it

    def reset_count(self):
        self.true_count = 0
        self.pov_count = []

    def print_stats(self):
        pov_count_str = "\n".join([f"Player {i}: {count}" for i, count in enumerate(self.pov_count)])
        blockers_str = "\n".join([f"Player {i}: {blockers}" for i, blockers in enumerate(self.pov_blockers)])
        res = f"True count: {self.true_count}\nPOV Count:\n{pov_count_str}\nBlockers:\n{blockers_str}"
        print(res)