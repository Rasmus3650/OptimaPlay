class PokerAgent():
    def __init__(self):
        self.states = []  # The reward agent does not just depend on the current state, but the entire history of states. Take the last element for current state
        self.reward = 0
        self.table = None   # Setup the table for the agent

    def compute_action(self, state):
        """
        Compute the action based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            The computed action.
        """
        pass

    def compute_bet_amount(self, state):
        """
        Compute the bet amount based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            The computed bet amount.
        """
        pass
