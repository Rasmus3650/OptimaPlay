import json

class RewardCalculator():
    def __init__(self):
        super().__init__()
        self.reward_json = json.load(open("Backgammon/Models/rewards.json"))
    
    def calculate_reward(self, board, moves, action, bar, homes, current_player):
        reward = 0
        if action is None:
            for chip in bar:
                if chip == current_player:
                    reward += self.reward_json["selfKnockout"]
                else:
                    reward += self.reward_json["enemyKnockout"]
            return reward
        
        reward += self.reward_json["movePenalty"]
        other_player = (current_player + 1) % 2
        if len(homes[current_player]) == 15:
            reward += self.reward_json["win"]
        elif len(homes[other_player]) == 15:
            reward += self.reward_json["lose"]
        return reward