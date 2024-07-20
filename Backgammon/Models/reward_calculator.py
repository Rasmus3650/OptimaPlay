import json

class RewardCalculator():
    def __init__(self):
        super().__init__()
        self.reward_json = json.load(open("rewards.json"))
    
    def calculate_reward(self, board, moves, action, bar, homes, current_player):
        reward = 0
        if action is None:
            for chip in bar:
                if chip == current_player:
                    reward += self.reward_json["selfKnockout"]
                else:
                    reward += self.reward_json["enemyKnockout"]
            
            return reward
        white_home, black_home = homes[0], homes[1]
        


