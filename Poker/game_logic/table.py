from .game import Game
from .deck import Deck
from .player import Player
import os
from Strategies.GTO_strategy import GTO_strategy

class Table():
    def __init__(self, start_balance: float, side: int, save_table = True, record_folder_path = "Poker/recorded_tables/", play_untill_1_winner = True) -> None:
        self.game_history: list[Game] = []
        self.seated_players = {}
        self.start_balance = start_balance
        self.deck = Deck()
        self.current_game = None
        self.record_folder_path = record_folder_path
        self.play_untill_1_winner = play_untill_1_winner
        self.table_id = self.get_table_id()
        self.curr_pos = 0
        self.side = side
        self.corner_points = [[249,325 + (side*960)], [249, 388 + (side*960)+ 1], [249, 453 + (side*960)+1], [249, 517 + (side*960)], [249,582 + (side*960)]]
        self.save_table = save_table
        self.past_players = {}

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

    
    def check_if_all_folded(self):
        for player in self.current_game.player_list:
            if not player.folded: return False
        return True
    
    def start_game(self):
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self)
        while not self.current_game.game_ended:
            action = self.current_game.player_performed_action()
        
        #input(f"Game {self.current_game.game_id} ended...")
        
        self.deck.reset_deck()
            
        if self.play_untill_1_winner and len(list(self.seated_players.keys())) > 1:
            self.start_game()
        else:
            print(f"Run completed in {self.current_game.game_id} games")
            winner_id = list(self.seated_players.keys())[0]
            winner = self.seated_players[winner_id]
            print(f"Balances: ")
            print(f"  {winner_id}: {winner.balance} $ [WINNER]     {type(winner.balance)}")
            for p_id in list(self.past_players.keys()):
                print(f"  {p_id}: {self.past_players[p_id].balance} $     {type(self.past_players[p_id].balance)}")
        

    def update_players(self):
        for player_id in list(self.seated_players.keys()):
            player = self.seated_players[player_id]
            if player.balance <= 0.01:
                removed_player = self.seated_players.pop(player_id)
                self.past_players[removed_player.player_id] = removed_player



    def end_game(self):
        if self.save_table:
            save_path = self.get_game_folder(self.get_table_folder(), self.current_game.game_id)
            self.current_game.record_game(save_path)
        self.game_history.append(self.current_game)
        self.update_players()

    def get_side(self):
        return self.side
    
    def get_game_id(self):
        return self.current_game.game_id

    def player_joined(self): 
        id = len(list(self.seated_players.keys()))
        if id == 0:
            self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, table=self, strategy=GTO_strategy)
        else:
            self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, table=self)


    def player_left(self, player_id):
        self.seated_players.pop(player_id)
    
    def __repr__(self):
        return f"Table {self.side}:\n  Seated players: {len(self.seated_players)}\n  {self.current_game}\n  Undiscovered Cards: {len(self.deck.undiscovered_cards)}\n  Discovered Cards: {len(self.deck.discovered_cards)}"
    