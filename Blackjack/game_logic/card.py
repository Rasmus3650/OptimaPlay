class Card():
    def __init__(self, rank: int, suit: str):
        self.current_rank = rank
        self.current_suit = suit
        self.current_value = self.get_value()
    
    def get_value(self):
        all_ranks = [2,3,4,5,6,7,8,9,10,10,10,10,11]
        all_values = [2,3,4,5,6,7,8,9,10,11,12,13,1]
        return all_values[all_ranks.index(self.current_rank)]

    def __repr__(self) -> str:
        return f'Card({self.current_rank}, "{self.current_suit}")'