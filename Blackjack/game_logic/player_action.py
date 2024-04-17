
class Player_Action():
    def __init__(self, table, player_id: int, action_str: str, new_card) -> None:
        self.table = table
        self.player_id = player_id
        self.action_str = action_str
        self.new_card = new_card
    
    def __repr__(self) -> str:
        return_str = f"Table 'side {self.table.table_id}':\n  Player ID: {self.player_id} did '{self.action_str}'"
        if self.action_str == "Bet" or self.action_str == "Raise" or self.action_str == "Call": return_str += f"\n  Bet Amount: {self.bet_amount}"
        return return_str
