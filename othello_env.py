import gym
from gym import spaces
import numpy as np
from othello import Othello

class OthelloEnv(gym.Env):
    """
    OpenAI Gym environment for an Othello (Reversi) game.
    Assumes that the Othello class maintains the current board state in
    a `board` attribute and the current player in a `current_player` attribute.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(OthelloEnv, self).__init__()
        self.game = Othello()
        self.board_size = Othello.BOARD_SIZE
        
        # Define the action space:
        # The game board is BOARD_SIZE x BOARD_SIZE and the special NOOP_ACTION uses a row index of BOARD_SIZE.
        # Thus, actions are represented as a pair [i, j] where i in {0, 1, ..., BOARD_SIZE} and j in {0, 1, ..., BOARD_SIZE-1}.
        self.action_space = spaces.MultiDiscrete([self.board_size + 1, self.board_size])
        
        # Define the observation space:
        # The board is represented as an 8x8 grid (or BOARD_SIZE x BOARD_SIZE) with cell values -1 (BLACK), 0 (EMPTY), or 1 (WHITE).
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.board_size, self.board_size),
                                            dtype=np.int8)

    def reset(self):
        """
        Reset the game to its initial state.
        Returns:
            observation (np.ndarray): The initial board configuration.
            info (dict): Additional info (the action_mask that determines the valid moves from the returned state).
        """
        self.game.reset()
        # Retrieve the initial board configuration.
        board = self.game.get_initial_board()
        # Assume that the game stores its board state in a `board` attribute.
        self.game.board = board

        info = {'action_mask': self.create_flattened_action_mask_for_current_player()}
        return np.array(board, dtype=np.int8), info

    def create_flattened_action_mask_for_current_player(self):
        board_size = Othello.BOARD_SIZE
        total_actions = board_size * board_size + 1
        # Initialize the mask with zeros.
        mask = np.zeros(total_actions, dtype=np.float32)

        legal_moves = self.game.get_legal_actions(self.game.player)

        for action in legal_moves:
            row, col = action
            if row < board_size:
                index = row * board_size + col
            else:
                # For NOOP action (row == board_size)
                index = board_size * board_size
            mask[index] = 1.0

        return mask

    def sample_random_action(self, action_mask):
        """
        Sample a random valid action based on the provided action mask.
        
        Args:
            action_mask (np.ndarray): A flattened action mask with shape (BOARD_SIZE*BOARD_SIZE + 1,)
                where valid actions have a value of 1.0.
        
        Returns:
            List[int]: A randomly selected valid action in [row, col] format, or NOOP_ACTION.
        """
        # Get all valid indices where the action mask is 1.0.
        valid_indices = np.flatnonzero(action_mask)
        # Randomly select one valid index.
        chosen_idx = np.random.choice(valid_indices)
        board_size = Othello.BOARD_SIZE
        if chosen_idx < board_size * board_size:
            row = chosen_idx // board_size
            col = chosen_idx % board_size
            return [row, col]
        else:
            # This corresponds to the NOOP action.
            return Othello.NOOP_ACTION

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (List[int]): The action to perform, represented as [i, j].

        Returns:
            observation (np.ndarray): The updated board state.
            reward (int): The reward obtained after the move.
            terminated (bool): Whether the game has completed.
            truncated (bool): Whether the game timed out. (Always False in this environment.)
            info (dict): Additional info (the action_mask that determines the valid moves from the returned state).
        """
        # Convert action to list if it comes as a numpy array.
        if isinstance(action, np.ndarray):
            action = action.tolist()

        # Retrieve the current player from the game instance.
        # (Assumes the Othello class has a current_player attribute.)
        current_player = self.game.player

        # Get the legal actions for the current player.
        legal_actions = self.game.get_legal_actions(current_player)
        # If the provided action is not legal, default to the NOOP action.
        assert action in legal_actions, "Invalid action: {}. Must be one of: {}".format(action, legal_actions)

        # Execute the move; the step method returns True if the game is finished.
        terminated = self.game.step(action)

        # Retrieve the updated board state.
        board = self.game.board

        # Calculate the reward.
        # Here we set reward=0 for non-terminal states.
        # When the game is terminated, we compute the reward as the score difference.
        if terminated:
            # score_white = self.game.get_score(Othello.WHITE)
            # score_black = self.game.get_score(Othello.BLACK)
            # reward = score_white - score_black
            reward = 1
        else:
            reward = 0

        info = {'action_mask': self.create_flattened_action_mask_for_current_player()}
        truncated = False
        return np.array(board, dtype=np.int8), reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the current board state.
        For the 'human' mode, prints the board to the console using:
        - 'W' for white pieces,
        - 'B' for black pieces,
        - '.' for empty cells.
        The method also prints the column indices at the top and bottom,
        and row indices along the left side.
        """
        board = self.game.board
        board_size = len(board)

        # Create a string with column indices
        col_indices = "    " + " ".join(str(i) for i in range(board_size))
        print(col_indices)
        
        # Print each row with its row index
        for idx, row in enumerate(board):
            row_str = " ".join("W" if cell == Othello.WHITE 
                            else "B" if cell == Othello.BLACK 
                            else "." for cell in row)
            print(f"{idx:2d}  {row_str}")

        print()

    def close(self):
        """
        Perform any necessary cleanup (not needed for this simple environment).
        """
        pass
