from .card import Card

class Deck():
    def __init__(self, set_of_cards: int = 1):
        self.all_suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.all_ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        self.undiscovered_cards = []
        self.discovered_cards = []
        for _ in range(set_of_cards):
            for rank in self.all_ranks:
                for suit in self.all_suits:
                    self.undiscovered_cards.append(Card(rank, suit))
                    
    def draw_card(self, in_card: Card):
        if in_card != None:
            for card in self.undiscovered_cards:
                if in_card == card:
                    self.discovered_cards.append(card)
                    self.undiscovered_cards.remove(card)

    def print_discovered_cards(self):
        result = f"Discovered Cards:\n"
        for card in self.discovered_cards:
            result += f"  {card}\n"
        print(result)

    def print_undiscovered_cards(self):
        result = f"Undiscovered Cards: \n"
        for card in self.discovered_cards:
            result += f"  {card}\n"
        print(result)

    def __repr__(self) -> str:
        return_str = f"Deck: \nCards played {len(self.discovered_cards)} / {len(self.undiscovered_cards)}\n(More details: 'print_discovered_cards()' and 'print_undiscovered_cards()')\n"
        