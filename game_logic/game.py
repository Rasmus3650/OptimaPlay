from .player import Player

class Game():
    def __init__(self, game_id, player_list: list[Player], return_function, start_balance: int = None) -> None:
        self.game_id = game_id
        self.player_list = player_list
        self.pot = 0
        self.pot_history = []
        self.all_game_states = ["Pre-round", "Pre-flop", "Flop", "Turn", "River", "Showdown", "Conclusion"]
        self.game_state = ["Pre-round"]
        self.return_function = return_function


    def player_performed_action(self, player_id: int, action: str, amount: int = None):
        self.player_list[player_id].perform_action(action, amount)
        
    def game_over(self):
        self.return_function()
    
    def __repr__(self) -> str:
        return_str = f"Game {self.game_id}\nNumber of players: {len(self.player_list)}\nGame State: {self.game_state}\nPot\n"
        for pot in self.pot_history:
            return_str += f"  {pot}\n"
        return return_str
