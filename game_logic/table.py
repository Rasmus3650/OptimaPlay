from .game import Game
from .deck import Deck
from .player import Player

class Table():
    def __init__(self, start_balance: float, side: int) -> None:
        self.game_history: list[Game] = []
        self.seated_players = []
        self.deck = Deck()
        self.current_game = None
        self.curr_id = 0
        self.curr_pos = 0
        self.side = side
        self.corner_points = [[249,325 + (side*960)], [249, 388 + (side*960)+ 1], [249, 453 + (side*960)+1], [249, 517 + (side*960)], [249,582 + (side*960)]]
    
    def check_if_all_folded(self):
        for player in self.current_game.player_list:
            if not player.folded: return False
        return True
    
    def start_game(self):
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self)
        print(self.current_game)
        while not self.current_game.game_ended:
        #for _ in range(3):
            action = self.current_game.player_performed_action()
            print(action)
            print(self.current_game)
        

    def update_players(self):
        for player in self.seated_players:
            if player.balance <= 0.01:
                self.seated_players.remove(player)

    def end_game(self):
        print(f"RETURNREUTUNEURUERNEURNUERNU")
        self.game_history.append(self.current_game)
        self.update_players()
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self)
    
    def get_id(self):
        return self.side
    
    def get_game_id(self):
        return self.current_game.game_id

    def player_joined(self, balance: float = 1.6): #TESTING
        self.seated_players.append(Player(len(self.seated_players), len(self.seated_players) == 0, balance, table=self))

    def player_left(self, player_id):
        self.seated_players.pop(player_id)
    
    def __repr__(self):
        return f"Table {self.side}:\n  Seated players: {len(self.seated_players)}\n  {self.current_game}\n  Undiscovered Cards: {len(self.deck.undiscovered_cards)}\n  Discovered Cards: {len(self.deck.discovered_cards)}"
    
