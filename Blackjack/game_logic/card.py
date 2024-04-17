class Card():
    def __init__(self, rank: int, suit: str, value: int):
        self.current_rank = rank
        self.current_suit = suit
        self.current_value = value
    
    def __repr__(self) -> str:
        return f'Card({self.current_rank}, "{self.current_suit}")'