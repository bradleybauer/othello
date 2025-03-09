import gym
import torch
from gym import spaces
import numpy as np
from othello import Othello
import othello
from numba import jit

@jit
def create_flattened_action_mask_for_current_player(legal_moves):
    board_size = othello.BOARD_SIZE
    total_actions = board_size * board_size + 1
    # Initialize the mask with zeros.
    mask = np.zeros(total_actions, dtype=np.int8)

    for action in legal_moves:
        row, col = action
        if row < board_size:
            index = row * board_size + col
        else:
            # For NOOP action (row == board_size)
            index = board_size * board_size
        mask[index] = 1

    return mask

@jit
def inflate_action(flat_action):
    """
    Convert a flattened action index to a pair of row and column indices.
    
    Args:
        flat_action (int): The flattened action index.
    
    Returns:
        List[int]: The corresponding row and column indices.
    """
    board_size = othello.BOARD_SIZE
    if flat_action == board_size * board_size:
        return othello.NOOP_ACTION
    row = flat_action // board_size
    col = flat_action % board_size
    return row, col

class OthelloEnv(gym.Env):
    """
    OpenAI Gym environment for an othello (Reversi) game.
    Assumes that the othello class maintains the current board state in
    a `board` attribute and the current player in a `current_player` attribute.
    """
    metadata = {"render.modes": ["human"]}

    # TODO assumes player is BLACK and opponent is WHITE
    def __init__(self, opponent):
        super(OthelloEnv, self).__init__()
        self.opponent = opponent

        self.game = Othello()
        self.board_size = othello.BOARD_SIZE
        
        # Define the action space:
        # The game board is BOARD_SIZE x BOARD_SIZE and the special NOOP_ACTION uses a row index of BOARD_SIZE.
        # Thus, actions are represented as a pair [i, j] where i in {0, 1, ..., BOARD_SIZE} and j in {0, 1, ..., BOARD_SIZE-1}.
        self.action_space = spaces.MultiDiscrete([self.board_size + 1, self.board_size])
        
        # Define the observation space:
        # The board is represented as an 8x8 grid (or BOARD_SIZE x BOARD_SIZE) with cell values -1 (BLACK), 0 (EMPTY), or 1 (WHITE).
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.board_size, self.board_size),
                                            dtype=np.int8)

    def get_info(self):
        info = {'action_mask': create_flattened_action_mask_for_current_player(self.game.get_legal_actions(self.game.player))}#, 'current_player': self.game.player, 'remaining_pieces_for_current_player'}
        return info

    def reset(self):
        """
        Reset the game to its initial state.
        Returns:
            observation (np.ndarray): The initial board configuration.
            info (dict): Additional info (the action_mask that determines the valid moves from the returned state).
        """
        self.game.reset()
        board = self.game.board

        # Initial self.game.player is random to simulate that the agent under optimization does not always
        # go first. (agent is always BLACK)
        if self.game.player == othello.WHITE:
            action_mask = self.get_info()['action_mask']
            with torch.no_grad():
                state_tensor = torch.from_numpy(-board).reshape(1,-1).float()
                flat_action = self.opponent.select_action(state_tensor, torch.from_numpy(action_mask))
            action = inflate_action(flat_action.item())
            if isinstance(action, np.ndarray):
                action = tuple(action.tolist())
            self.game.step(action)
            board = self.game.board

        return np.array(board, dtype=np.int8), self.get_info()

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (List[int]): The action to perform, represented as [i, j].

        Returns:
            observation (np.ndarray): The updated board state.
            reward (int): The reward obtained after the move.
            done (bool): Whether the game has completed.
            truncated (bool): Whether the game timed out. (Always False in this environment.)
            info (dict): Additional info (the action_mask that determines the valid moves from the returned state).
        """
        # Convert action to list if it comes as a numpy array.
        if isinstance(action, np.ndarray):
            action = tuple(action.tolist())

        assert(self.game.player == othello.BLACK)

        # Execute the move; the step method returns True if the game is finished.
        done = self.game.step(action)

        # Retrieve the updated board state.
        state = self.game.board
        info = self.get_info()

        assert(self.game.player == othello.WHITE)

        if done:
            score_white = self.game.get_score(othello.WHITE)
            score_black = self.game.get_score(othello.BLACK)
            reward = 1 if score_black > score_white else 0 if score_black == score_white else -1
        else: # step opponent
            action_mask = info['action_mask']
            with torch.no_grad():
                # the model always plays as BLACK
                state_tensor = torch.from_numpy(-state).reshape(1,-1).float()
                flat_action = self.opponent.select_action(state_tensor, torch.from_numpy(action_mask))
                action = inflate_action(flat_action.item())
            done = self.game.step(action)
            state = self.game.board
            info = self.get_info()
            if done:
                score_white = self.game.get_score(othello.WHITE)
                score_black = self.game.get_score(othello.BLACK)
                reward = 1 if score_black > score_white else 0 if score_black == score_white else -1
            else:
                reward = 0

        truncated = False
        return state, reward, done, truncated, info
