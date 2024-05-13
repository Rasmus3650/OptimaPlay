from Blackjack.game_logic.game import Game
from Blackjack.game_logic.deck import Deck
from Blackjack.game_logic.player import Player
import os
from Blackjack.Strategies.GTO_strategy import GTO_strategy
from Blackjack.Strategies.random_strategy import Random_strategy


class Table():
    def __init__(self, start_balance: float,table_id:int, save_table = True, record_folder_path = "Blackjack/recorded_tables/", play_untill_1_winner = True, reset = 0.5, set_of_cards = 4, consumer_thread = None) -> None:
        self.game_history: list[Game] = []
        self.seated_players = {}
        self.start_balance = start_balance
        self.deck = Deck(set_of_cards=set_of_cards, reset=reset)
        self.current_game = None
        self.record_folder_path = record_folder_path
        self.play_untill_1_winner = play_untill_1_winner
        self.curr_pos = 0
        self.past_players = {}
        self.save_table = save_table
        self.reset = reset
        self.consumer_thread = consumer_thread
        self.strategy_map = {"random": Random_strategy, "gto": GTO_strategy}
        self.table_id = table_id


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
        self.current_game = Game(len(self.game_history), self.seated_players, return_function=self.end_game, table=self, save_game=self.save_table, game_folder=game_folder, consumer_thread=self.consumer_thread)
        while not self.current_game.game_ended:
            action = self.current_game.player_performed_action()
        
        if self.play_untill_1_winner and len(list(self.seated_players.keys())) > 1:
            self.start_game()
    

    def clear_player_hands(self):
        for p_id in list(self.seated_players.keys()):
            self.seated_players[p_id].clear_hand()

    def player_joined(self, strategy): 
        id = len(list(self.seated_players.keys()))
        #if id == 0:
        #    self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, table=self, strategy=GTO_strategy)
        #else:
        self.seated_players[id] = Player(id, len(self.seated_players) == 0, self.start_balance, strategy= self.strategy_map[strategy.lower()],table=self)

    def player_left(self, player_id):
        self.seated_players.pop(player_id)

    def end_game(self):
        self.game_history.append(self.current_game)
        self.clear_player_hands()
        self.update_players()

    def update_players(self):
        for player_id in list(self.seated_players.keys()):
            player = self.seated_players[player_id]
            if player.balance <= 0.01:
                removed_player = self.seated_players.pop(player_id)
                self.past_players[removed_player.player_id] = removed_player
    
    def __repr__(self):
        return f"Table: {self.table_id}\n{self.current_game}\n  Undiscovered Cards: {len(self.deck.undiscovered_cards)}\n  Discovered Cards: {len(self.deck.discovered_cards)}"
    