from .game import Game
from .deck import Deck
from .player import Player
import os


class Table():
    def __init__(self, start_balance: float, save_table = True, record_folder_path = "Blackjack/recorded_tables/", play_untill_1_winner = True, reset = 0.5, set_of_cards = 4) -> None:
        self.game_history: list[Game] = []
        self.seated_players = {}
        self.start_balance = start_balance
        self.deck = Deck(set_of_cards=set_of_cards, reset=reset)
        self.current_game = None
        self.record_folder_path = record_folder_path
        self.play_untill_1_winner = play_untill_1_winner
        self.table_id = self.get_table_id()
        self.curr_pos = 0
        self.past_players = {}
        self.save_table = save_table
        self.reset = reset

    def get_table_id(self):
        if not os.path.exists(self.record_folder_path): return 1
        res = len([name for name in os.listdir(self.record_folder_path)
            if os.path.isdir(os.path.join(self.record_folder_path, name))])
        return res + 1
    
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
    
    def start_game(self):

        if self.save_table:
            game_folder = self.get_game_folder(self.get_table_folder(), len(self.game_history))
        else:
            game_folder = None


        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self, save_game=self.save_table, game_folder=game_folder)
        print(f"\nSTARTING GAME {self.current_game.game_id}...")
        while not self.current_game.game_ended:
            action = self.current_game.player_performed_action()
        print(f"GAME {self.current_game.game_id} ENDED\n")
        
        if self.play_untill_1_winner and len(list(self.seated_players.keys())) > 1:
            self.start_game()
    
    def player_joined(self): 
        id = len(list(self.seated_players.keys()))
        #if id == 0:
        #    self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, table=self, strategy=GTO_strategy)
        #else:
        self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, table=self)

    def player_left(self, player_id):
        self.seated_players.pop(player_id)

    def end_game(self):
        print(f"Game has ended")
        self.game_history.append(self.current_game)
        self.update_players()

    def update_players(self):
        for player_id in list(self.seated_players.keys()):
            player = self.seated_players[player_id]
            if player.balance <= 0.01:
                removed_player = self.seated_players.pop(player_id)
                self.past_players[removed_player.player_id] = removed_player
    
    def __repr__(self):
        return f"Table: {self.table_id}\n{self.current_game}\n  Undiscovered Cards: {len(self.deck.undiscovered_cards)}\n  Discovered Cards: {len(self.deck.discovered_cards)}"
    