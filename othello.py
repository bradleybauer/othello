import random
import numpy as np
from numba import int8, boolean
from numba.experimental import jitclass
from numba.typed import List

# Global constants (used inside the jitclass)
BOARD_SIZE = 8
BLACK = 1
WHITE = -1
EMPTY = 0
NUM_STARTING_PIECES = (BOARD_SIZE**2) // 2 - 2
NOOP_ACTION = (BOARD_SIZE, 0)  # a tuple to represent the no–move action

# Specify the types for the jitclass attributes
spec = [
    ('board', int8[:, :]),         # 2D NumPy array for the board
    ('player', int8),              # current player
    ('previous_player_skipped', boolean),
    ('black_player_num_pieces', int8),
    ('white_player_num_pieces', int8)
]

@jitclass(spec)
class Othello:
    def __init__(self):
        # Initialize the board as an 8x8 NumPy array
        self.board = self.get_empty_board()
        self.player = BLACK
        self.previous_player_skipped = False
        self.black_player_num_pieces = NUM_STARTING_PIECES
        self.white_player_num_pieces = NUM_STARTING_PIECES
        self.reset()

    def reset(self):
        # Reset the game state
        self.board = self.get_initial_board()
        self.player = BLACK
        self.previous_player_skipped = False
        self.black_player_num_pieces = NUM_STARTING_PIECES
        self.white_player_num_pieces = NUM_STARTING_PIECES

    def get_empty_board(self):
        # Create an empty board filled with EMPTY
        return np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)

    def get_initial_board(self):
        board = self.get_empty_board()
        mid = BOARD_SIZE // 2
        # randomize the initial board configuration
        if random.random() < 0.5:
            board[mid - 1, mid - 1] = WHITE
            board[mid, mid] = WHITE
            board[mid - 1, mid] = BLACK
            board[mid, mid - 1] = BLACK
        else:
            board[mid - 1, mid - 1] = BLACK
            board[mid, mid] = BLACK
            board[mid - 1, mid] = WHITE
            board[mid, mid - 1] = WHITE
        return board

    def get_score(self, which_player):
        return np.where(self.board == which_player, 1, 0).sum()

    def coordinate_in_bounds(self, i, j):
        return (i >= 0) and (i < BOARD_SIZE) and (j >= 0) and (j < BOARD_SIZE)

    def opposite_player(self, player):
        return -player
        # if player == BLACK:
        #     return WHITE
        # else:
        #     return BLACK

    def board_is_full(self):
        return np.all(self.board != EMPTY)

    def get_legal_actions(self, for_player):
        # Use a typed list to accumulate legal actions (each as a tuple of (i, j))
        legal_actions = List()
        # If the player has no remaining pieces, the only move is NOOP_ACTION
        if for_player == BLACK:
            if self.black_player_num_pieces == 0:
                legal_actions.append(NOOP_ACTION)
                return legal_actions
        else:
            if self.white_player_num_pieces == 0:
                legal_actions.append(NOOP_ACTION)
                return legal_actions

        # Find open positions that are adjacent to an enemy piece.
        open_positions = List()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] == EMPTY:
                    is_available = False
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = j + dj
                            if self.coordinate_in_bounds(ni, nj):
                                if self.board[ni, nj] == self.opposite_player(for_player):
                                    is_available = True
                                    break
                        if is_available:
                            break
                    if is_available:
                        open_positions.append((i, j))

        enemy = self.opposite_player(for_player)
        # Validate each open position by checking in all eight directions.
        for pos in open_positions:
            i, j = pos
            is_valid_play = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    for k in range(1, BOARD_SIZE):
                        ip = i + k * di
                        jp = j + k * dj
                        if not self.coordinate_in_bounds(ip, jp):
                            break
                        if k >= 2 and self.board[ip, jp] == for_player:
                            is_valid_play = True
                            break
                        if self.board[ip, jp] != enemy:
                            break
                    if is_valid_play:
                        break
            if is_valid_play:
                legal_actions.append((i, j))
        if len(legal_actions) == 0:
            legal_actions.append(NOOP_ACTION)
        return legal_actions

    def step(self, action):
        done = False
        # Check if the action is the NOOP_ACTION.
        if not (action[0] == NOOP_ACTION[0] and action[1] == NOOP_ACTION[1]):
            if self.player == BLACK:
                self.black_player_num_pieces -= 1
            else:
                self.white_player_num_pieces -= 1

            i = action[0]
            j = action[1]
            self.board[i, j] = self.player

            # Flip enemy discs in all eight directions.
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    for k in range(1, BOARD_SIZE):
                        ip = i + k * di
                        jp = j + k * dj
                        if not self.coordinate_in_bounds(ip, jp):
                            break
                        if self.board[ip, jp] == EMPTY:
                            break
                        if self.board[ip, jp] == self.player:
                            for kp in range(1, k):
                                ip2 = i + kp * di
                                jp2 = j + kp * dj
                                self.board[ip2, jp2] = self.player
                            break

            self.previous_player_skipped = False
        else:
            if self.previous_player_skipped:
                done = True
            self.previous_player_skipped = True

        if self.board_is_full():
            done = True

        self.player = self.opposite_player(self.player)
        return done
