import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.table import Table
from game_logic.deck import Deck

def first_test():
    print("AAAAAAAAAAA")
    table = Table(100)
    table.start_game()
    table.end_game()
    table.start_game()
    table.end_game()
    for t in table.game_history:
        print(t.game_id)
    #print(table.game_history)


