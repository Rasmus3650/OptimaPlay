class PokerAgent():
    def __init__(self):
        self.states = []  # The reward agent does not just depend on the current state, but the entire history of states. Take the last element for current state
        self.reward = 0
        self.table = None   # Setup the table for the agent


    # Start the game
    def start_game(self):
        pass

    def get_state(self):
        # Get the game state and append to self.states
        pass

    def encode_state(self, encoding_scheme):
        # Encode a game state, to be suitable as input for a NN, given an encoding scheme
        pass

    def update_reward(self):
        # Update the reward depending on the reward_function passed along from the models object.
        pass

    def update_states(self, state):
        self.states.append(state)

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