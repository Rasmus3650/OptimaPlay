from Blackjack.game_logic.card import Card
import random

class Deck():
    def __init__(self, set_of_cards: int = 1, reset = 0.5):
        self.soc = set_of_cards
        self.reset = reset
        self.all_suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.all_ranks = [2,3,4,5,6,7,8,9,10,10,10,10,11]
        self.undiscovered_cards = []
        self.discovered_cards = []
        for _ in range(set_of_cards):
            for rank in self.all_ranks:
                for suit in self.all_suits:
                    self.undiscovered_cards.append(Card(rank, suit))
        self.total_len = len(self.undiscovered_cards)
        self.reset_at = round(self.total_len * reset)
                    
    def draw_cards(self, amount):
        res_list = []
        for _ in range(amount):
            card = random.choice(self.undiscovered_cards)
            res_list.append(card)
            self.discovered_cards.append(card)
            self.undiscovered_cards.remove(card)

        if len(self.undiscovered_cards) <= self.reset_at:
            self.reset_deck(self.soc, self.reset)

        return res_list
    
    def reset_deck(self, set_of_cards: int = 1, reset = 0.5):
        self.__init__(set_of_cards, reset)
    
    def is_empty(self):
        return len(self.undiscovered_cards) == 0

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
        