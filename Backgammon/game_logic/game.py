from .player import Player
from .board import Board
import numpy as np, os, json

class Game():
    def __init__(self, game_id, player_list: dict[int, Player], return_function, game_folder = None, consumer_thread = None, save_game = False) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.board = Board()
        self.game_ended = False
        self.return_function = return_function
        self.current_player = self.player_list[np.random.choice(list(self.player_list.keys()))]
        self.winner = None
        self.consumer_thread = consumer_thread
        self.game_folder = game_folder
        self.save_game = save_game
        self.all_actions = []
    
    def roll_dice(self):
        return np.random.randint(1, 7), np.random.randint(1, 7)
    
    def perform_player_action(self):
        print(f"###################################################")
        self.board.print_board()
        print(f"Player {self.current_player.backgammon_color} 's turn")
        roll1, roll2 = self.roll_dice()
        if roll1 == roll2:
            available_dice = [roll1, roll1, roll2, roll2]
        else:
            available_dice = [roll1, roll2]
        print(f"    Available_dice: {available_dice}")
        while len(available_dice) > 0:
            moves = self.board.get_moves(self.current_player.backgammon_color, available_dice)
            print(f"    Available Moves: {moves}")
            move = self.current_player.get_action(moves)
            print(f"    Move: {move}")
            if move is not None:
                self.board.perform_move(self.current_player.backgammon_color, move)
                self.all_actions.append([len(self.all_actions), self.current_player.backgammon_color, self.board.board, move])
                available_dice.remove(abs(move[2]))
                self.check_winner()
            else:
                break
            if self.game_ended:
                break
        self.current_player = self.get_player_by_color((self.current_player.backgammon_color + 1) % 2)
        self.board.print_board()
        print(f"###################################################")
        if self.game_ended:
            self.game_over()
    
    def get_player_by_color(self, color):
        for p_id in list(self.player_list.keys()):
            if self.player_list[p_id].backgammon_color == color:
                return self.player_list[p_id]
            
    def check_winner(self):
        has_won = [True, True]
        for field in self.board.board:
            for player in [0, 1]:
                if player in field:
                    has_won[player] = False

        if has_won[0]:
            self.winner = self.get_player_by_color(0)
        if has_won[1]:
            self.winner = self.get_player_by_color(1)

        if has_won[0] or has_won[1]:
            self.game_ended = True

    def game_over(self):
        if self.save_game:
            self.record_game()
        self.return_function()

    
    def record_game(self):
        game_data = {}
        game_data['actions'] = self.all_actions
        #print(game_data)
        if self.consumer_thread == None:
            with open(os.path.join(self.game_folder, f"game_data.json"), "w") as json_file:
                json.dump(game_data, json_file)
        else:
            self.consumer_thread.enqueue_data(game_data, self.game_folder)
