from .game import Game
from .deck import Deck
from .player import Player

class Table():
    def __init__(self, start_balance: float, side: int) -> None:
        self.balance = start_balance
        self.game_history: list[Game] = []
        self.seated_players = []
        self.deck = Deck()
        self.current_game = None
        self.curr_id = 0
        self.curr_pos = 0
        self.side = side
        self.corner_points = [[249,325 + (side*960)], [249, 388 + (side*960)+ 1], [249, 453 + (side*960)+1], [249, 517 + (side*960)], [249,582 + (side*960)]]
    
    def start_game(self):
        self.current_game = Game(len(self.game_history), self.seated_players, self.end_game, self.balance)

    def end_game(self):
        self.game_history.append(self.current_game)


    def player_joined(self, balance: float = None):
        self.seated_players.append(Player(len(self.seated_players), len(self.seated_players) == 0, balance))

    def player_left(self, player_id):
        self.seated_players.pop(player_id)
    
    def __repr__(self):
        return f"Table:\nSeated players: {len(self.seated_players)}\n{self.current_game}\n{self.deck}\n"

