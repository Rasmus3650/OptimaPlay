class Card():
    def __init__(self, rank: int, suit: str):
        self.current_rank = rank
        self.current_suit = suit
    
    def __repr__(self) -> str:
        return f'Card({self.current_rank}, "{self.current_suit}")'