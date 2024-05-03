
import sys, os, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Blackjack.game_logic.card import Card
from Blackjack.game_logic.table import Table
from Blackjack.game_logic.deck import Deck
from Blackjack.game_logic.player import Player
from Blackjack.game_logic.game import Game

class BlackjackTraining():
    def __init__(self, number_of_tables: int = 1, consumer_thread = None, folder_path:str = "Blackjack/recorded_tables") -> None:
        self.folder_path = folder_path
        start_num = self.get_latest_table()
        
        self.table_list = {}
        for i in range(start_num, start_num+number_of_tables):
            self.table_list[i]=Table(random.randint(5,1000),table_id=i, consumer_thread=consumer_thread)
        
        for table in list(self.table_list.keys()):
            for i in range(6):
                self.table_list[table].player_joined()
            self.table_list[table].start_game()
            
        consumer_thread.enqueue_data({"stop": True}) 


    def get_latest_table(self):
        if not os.path.exists(self.folder_path): return 1
        res = len([name for name in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, name))])
        return res + 1

    def get_cards_on_hand(self, table_id: int, player_id: int = 0) -> list[Card]: #Return list of length 2
        return self.table_list[table_id].seated_players[player_id].hand
    
    def get_cards_on_table(self, table_id):
        return self.table_list[table_id].current_game.cards_on_table

    def get_player_balance(self, table_id, player_id):
        return self.table_list[table_id].seated_playes[player_id].balance