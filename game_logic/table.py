from .game import Game
from .deck import Deck
from .player import Player
import os

class Table():
    def __init__(self, start_balance: float, side: int, save_table = True, record_folder_path = "recorded_tables/") -> None:
        self.game_history: list[Game] = []
        self.seated_players = []
        self.deck = Deck()
        self.current_game = None
        self.record_folder_path = record_folder_path
        self.table_id = self.get_table_id()
        self.curr_pos = 0
        self.side = side
        self.corner_points = [[249,325 + (side*960)], [249, 388 + (side*960)+ 1], [249, 453 + (side*960)+1], [249, 517 + (side*960)], [249,582 + (side*960)]]
        self.save_table = save_table
        

    def get_table_id(self):
        if not os.path.exists(self.record_folder_path): return 0
        sub_dirs = [x[0] for x in os.walk(self.record_folder_path)][1:]
        return len(sub_dirs)
    
    def get_table_folder(self):
        path = os.path.join(self.record_folder_path, f"table_{self.table_id}")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_game_folder(self, table_folder_path, game_id):
        path = os.path.join(table_folder_path, f"Game_{game_id}")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    
    def check_if_all_folded(self):
        for player in self.current_game.player_list:
            if not player.folded: return False
        return True
    
    def start_game(self):
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self)
        while not self.current_game.game_ended:
            action = self.current_game.player_performed_action()
            

        if self.save_table:
            save_path = self.get_game_folder(self.get_table_folder(), self.current_game.game_id)
            print(f"SAVE_PATH: {save_path}")
            self.current_game.record_game(save_path)

    def update_players(self):
        for player in self.seated_players:
            if player.balance <= 0.01:
                self.seated_players.remove(player)

    def end_game(self):
        print(f"RETURNREUTUNEURUERNEURNUERNU")
        self.game_history.append(self.current_game)
        self.update_players()
        if len(self.seated_players) > 1:
            #self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self)
            self.start_game()
    
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
    
