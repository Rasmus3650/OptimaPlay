import os
from Backgammon.game_logic.player import Player
from Backgammon.game_logic.game import Game
from Backgammon.Strategies.random_strategy import Random_strategy

class Table():
    def __init__(self, table_id:int, save_table = True, record_folder_path = "Backgammon/recorded_tables/", consumer_thread = None) -> None:
        self.game_history: list[Game] = []
        self.table_id = table_id
        self.save_table = save_table
        self.record_folder_path = record_folder_path
        self.current_game = None
        self.seated_players = {}
        self.strategy_map = {"random": Random_strategy}

        self.consumer_thread = consumer_thread
    
    def get_table_folder(self):
        if not os.path.exists(self.record_folder_path):
            os.mkdir(self.record_folder_path)
        path = os.path.join(self.record_folder_path, f"table_{self.table_id}")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_game_folder(self, table_folder_path, game_id):
        path = os.path.join(table_folder_path, f"Game_{game_id}")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def player_joined(self, strategy, p = None):
        id = len(list(self.seated_players.keys()))
        if p is None: 
            self.seated_players[id] = Player(id, len(self.seated_players) == 0, strategy=self.strategy_map[strategy.lower()], table=self)
        else:
            self.seated_players[id] = p

    def start_game(self):
        if self.save_table:
            game_folder = self.get_game_folder(self.get_table_folder(), len(self.game_history))
        else:
            game_folder = None
        
        self.seated_players[0].set_color(0)
        self.seated_players[1].set_color(1)
        
        
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.game_ended, game_folder=game_folder, consumer_thread=self.consumer_thread, save_game=self.save_table)

        while not self.current_game.game_ended:
            action = self.current_game.perform_player_action()
    
    def game_ended(self):
        self.game_history.append(self.current_game)