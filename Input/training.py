
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.card import Card
from game_logic.table import Table
from game_logic.deck import Deck
from game_logic.player import Player
from game_logic.game import Game
from game_logic.game import Game

class Training():
    def __init__(self, number_of_tables: int = 1) -> None:
        self.table_list = {}
        for i in range(number_of_tables):
            self.table_list[i]=Table(1.6, i)
        for table in list(self.table_list.keys()):
            for i in range(6):
                self.table_list[table].player_joined()
            self.table_list[table].start_game()
            
        




        #for i, table in enumerate(self.table_list):
        #    while len(table.seated_players)> 1:
        #        table.start_game()

    def get_cards_on_hand(self, table_id: int, player_id: int = 0) -> list[Card]: #Return list of length 2
        return self.table_list[table_id].seated_players[player_id].hand
    
    def get_cards_on_table(self, table_id):
        return self.table_list[table_id].current_game.cards_on_table

    def get_player_balance(self, table_id, player_id):
        return self.table_list[table_id].seated_playes[player_id].balance