import os, sys, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Backgammon.game_logic.table import Table




class BackgammonTraining():
    def __init__(self, number_of_tables: int = 1, consumer_thread = None, folder_path:str = "Backgammon/recorded_tables", strategies = []) -> None:
        self.folder_path = folder_path
        start_num = self.get_latest_table()
        if len(strategies) < 6:
            for i in range(len(strategies), 6):
                strategies.append("random")
        self.table_list = {}
        for i in range(start_num, start_num+number_of_tables):
            self.table_list[i]=Table(table_id=i, consumer_thread=consumer_thread)
        
        for table in list(self.table_list.keys()):
            for i in range(2):
                self.table_list[table].player_joined(strategies[i])
            self.table_list[table].start_game()
            
        consumer_thread.enqueue_data({"stop": True}) 



    def get_latest_table(self):
        if not os.path.exists(self.folder_path): return 1
        res = len([name for name in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, name))])
        return res + 1