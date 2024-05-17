
import sys, os, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.card import Card
from game_logic.table import Table
from game_logic.deck import Deck
from game_logic.player import Player
from game_logic.game import Game

class PokerTraining():
    def __init__(self, number_of_tables: int = 1, consumer_thread = None, folder_path:str = "Poker/recorded_tables", strategies = []) -> None:
        self.folder_path = folder_path
        start_num = self.get_latest_table()
        
        if len(strategies) < 6:
            for i in range(len(strategies), 6):
                strategies.append("random")

        self.table_list = {}
        for i in range(start_num, start_num+number_of_tables):
            self.table_list[i]=Table(random.randint(5,1000), table_id=i, consumer_thread=consumer_thread, record_folder_path=self.folder_path)
        
        for table in list(self.table_list.keys()):
            for i in range(6):
                self.table_list[table].player_joined(strategies[i])
            self.table_list[table].start_game()

        #consumer_thread.enqueue_data({"stop": True}) 
        




        #for i, table in enumerate(self.table_list):
        #    while len(table.seated_players)> 1:
        #        table.start_game()

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