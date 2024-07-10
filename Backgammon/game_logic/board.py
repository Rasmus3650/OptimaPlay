


class Board():
    def __init__(self) -> None:
        self.board = []
        self.init_board()
    
    def init_board(self):
        self.board = [[1, 1], [], [], [], [], [0, 0, 0, 0, 0], [], [0, 0, 0], [], [], [], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [], [], [], [1, 1, 1], [], [1, 1, 1, 1, 1], [], [], [], [], [0, 0]]
        self.white_home = [] #White = 0
        self.black_home = [] #Black = 1
        self.homes = [self.white_home, self.black_home]
        self.bar = []

    def is_blocked(self, player, field_n):
        opponent = (player + 1) % 2
        if field_n >= len(self.board) or field_n < 0:
            return True
        return opponent in self.board[field_n] and len(self.board[field_n]) > 1

    def get_special_home_moves(self, player, available_dice):
        if player == 0:
            goal = -1
        elif player == 1:
            goal = 24
        else:
            print(f"Player {player} SHOULD BE EITHER 0 OR 1...")
            input(f"The fuck?")
        extra_moves = []
        max_tile = 0
        max_tile_pos = -1
        if player == 0:
            available_dice = [-x for x in available_dice]
            for i in range(6):
                if player in self.board[i]:
                    max_tile = i
                    max_tile_pos = i
        if player == 1:
            for i in range(23, 17, -1):
                if player in self.board[i]:
                    max_tile = 24 - i
                    max_tile_pos = i
        
        for dice in available_dice:
            if dice > max_tile and [max_tile_pos, goal, dice] not in extra_moves:
                extra_moves.append([max_tile_pos, goal, dice])
        return extra_moves
    
    def get_bar_moves(self, player, available_dice):
        moves = []
        if player == 0:
            for dice in available_dice:
                if not self.is_blocked(player, 24 - dice):
                    moves.append([30, 24 - dice, dice])
        if player == 1:
            for dice in available_dice:
                if not self.is_blocked(player, dice - 1):
                    moves.append([30, dice - 1, dice])
        return moves


    def get_moves(self, player, available_dice):
        if player in self.bar:
            return self.get_bar_moves(player, available_dice)
        can_move_home = True
        if player == 0:
            goal = -1
            for i in range(6, len(self.board), 1):
                if player in self.board[i]: can_move_home = False
        if player == 1:
            goal = 24
            for i in range(18):
                if player in self.board[i]: can_move_home = False
        moves = []
        if player == 0:
            available_dice = [-x for x in available_dice]
        for i, field in enumerate(self.board):
            if player in field:
                for dice in available_dice:
                    if i + dice == goal and can_move_home:
                        if [i, i + dice, dice] not in moves:
                            moves.append([i, i + dice, dice])
                    if i + dice < len(self.board) and i + dice >= 0 and i + dice >= 0:
                        if not self.is_blocked(player, i + dice):
                            if [i, i + dice, dice] not in moves:
                                moves.append([i, i + dice, dice])
        if can_move_home:
            moves = moves + self.get_special_home_moves(player, available_dice)
        return moves
                
    def perform_move(self, player, move):
        start, stop, steps = move
        if start == 30:
            chip = player
            self.bar.remove(player)
        else:
            chip = self.board[start].pop()
        if stop == -1 or stop == 24:
            self.homes[player].append(chip)
        else:
            opponent = (player + 1) % 2
            
            if opponent in self.board[stop]:
                self.bar.append(self.board[stop].pop())

            self.board[stop].append(chip)
    
    def print_board(self):
        max_top_len = 0
        max_bot_len = 0
        for i in range(12):
            if len(self.board[i]) > max_top_len: max_top_len = len(self.board[i])
            if len(self.board[i + 12]) > max_bot_len: max_bot_len = len(self.board[i + 12])
        #for field in self.board: print(field)
        print_str = f""
        for i in range(max(max_top_len, max_bot_len)):
            for j in range(11, -1, -1):
                if len(self.board[j]) > i:
                    print_str += f"{self.board[j][i]} | "
                else:
                    print_str += f"  | "
                if j == 6:
                    print_str = print_str[:-2] + "||| "
            print_str = print_str[:-2] + "\n"
        print_str += "\n"
        for i in range(max(max_top_len, max_bot_len)):
            for j in range(12, 24, 1):
                if len(self.board[j]) > i:
                    print_str += f"{self.board[j][i]} | "
                else:
                    print_str += f"  | "
                if j == 17:
                    print_str = print_str[:-2] + "||| "
            print_str = print_str[:-2] + "\n"
        print(print_str)
        