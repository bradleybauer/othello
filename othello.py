from typing import List

class Othello:
    """
    A class representing an Othello (Reversi) game.
    
    Attributes:
        BOARD_SIZE (int): The size of the game board (BOARD_SIZE x BOARD_SIZE).
        WHITE (int): The identifier for the white player.
        BLACK (int): The identifier for the black player.
        EMPTY (int): The value representing an empty cell.
        NUM_STARTING_PIECES (int): The total number of pieces placed on the board at the start.
        NOOP_ACTION (List[int]): The action used to indicate a pass turn.
    """
    BOARD_SIZE: int = 8
    BLACK: int = 1
    WHITE: int = -1
    EMPTY: int = 0
    NUM_STARTING_PIECES: int = 32
    NOOP_ACTION: List[int] = [BOARD_SIZE, 0]

    def __init__(self) -> None:
        """
        Initialize a new Othello game instance.
        
        Sets up the initial state with an empty board, sets the starting player to WHITE,
        and initializes piece counters and the previous player move flag.
        """
        self.player: int = Othello.BLACK
        self.board: List[List[int]] = []
        self.previous_player_skipped: bool = False
        self.black_player_num_pieces: int = 0
        self.white_player_num_pieces: int = 0
        self.reset()

    def reset(self) -> None:
        """
        Reset the game to its initial state.
        
        This method sets up the initial board configuration with the four starting pieces,
        resets the current player to WHITE, clears the previous skip flag, and initializes
        the piece counters for both players.
        """
        self.board = self.get_initial_board()
        self.player = Othello.BLACK
        self.previous_player_skipped = False
        self.black_player_num_pieces = Othello.NUM_STARTING_PIECES - 2
        self.white_player_num_pieces = Othello.NUM_STARTING_PIECES - 2

    def get_initial_board(self) -> List[List[int]]:
        """
        Create and return the initial board configuration for Othello.
        
        The board is an 8x8 grid with the four central squares initialized:
        - Two WHITE pieces in the top-left and bottom-right of the center.
        - Two BLACK pieces in the top-right and bottom-left of the center.
        
        Returns:
            List[List[int]]: The initialized game board.
        """
        board: List[List[int]] = self.get_empty_board()
        mid: int = Othello.BOARD_SIZE // 2
        board[mid - 1][mid - 1] = Othello.WHITE
        board[mid][mid] = Othello.WHITE
        board[mid - 1][mid] = Othello.BLACK
        board[mid][mid - 1] = Othello.BLACK
        return board

    def get_empty_board(self) -> List[List[int]]:
        """
        Generate an empty board.
        
        Creates an 8x8 board (list of lists) where each cell is initialized to EMPTY.
        
        Returns:
            List[List[int]]: An empty game board.
        """
        return [[Othello.EMPTY for _ in range(Othello.BOARD_SIZE)] for _ in range(Othello.BOARD_SIZE)]

    def get_score(self, which_player: int) -> int:
        """
        Calculate the current score for a given player.
        
        The score is determined by counting the number of cells on the board that match
        the player's identifier.
        
        Args:
            which_player (int): The player identifier (WHITE or BLACK).
        
        Returns:
            int: The total count of the player's pieces on the board.
        """
        return sum(cell == which_player for row in self.board for cell in row)

    def get_legal_actions(self, for_player: int) -> List[List[int]]:
        """
        Determine all legal actions for the given player.
        
        This method scans the board to find empty positions adjacent to the opponent's pieces.
        It then validates each open position by checking in all eight directions to see if
        placing a piece there would capture enemy pieces.
        
        Args:
            for_player (int): The player for whom to compute legal moves.
        
        Returns:
            List[List[int]]: A list of legal actions represented by [i, j] coordinates.
                             If no legal moves are available, returns a list with the NOOP_ACTION.
        """
        if self.player == Othello.BLACK and self.black_player_num_pieces == 0:
            return [Othello.NOOP_ACTION]
        elif self.player == Othello.WHITE and self.white_player_num_pieces == 0:
            return [Othello.NOOP_ACTION]

        open_positions: List[List[int]] = []
        for i in range(Othello.BOARD_SIZE):
            for j in range(Othello.BOARD_SIZE):
                if self.board[i][j] == Othello.EMPTY:
                    is_available: bool = False
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            if self.coordinate_in_bounds(i + di, j + dj):
                                is_available = self.board[i + di][j + dj] == self.opposite_player(for_player)
                            if is_available:
                                break
                        if is_available:
                            break
                    if is_available:
                        open_positions.append([i, j])

        enemy: int = self.opposite_player(for_player)
        actions: List[List[int]] = []
        for position in open_positions:
            i, j = position
            is_valid_play: bool = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    for k in range(1, Othello.BOARD_SIZE):
                        ip, jp = i + k * di, j + k * dj
                        if not self.coordinate_in_bounds(ip, jp):
                            break
                        if k >= 2 and self.board[ip][jp] == for_player:
                            is_valid_play = True
                            break
                        if self.board[ip][jp] != enemy:
                            break
                    if is_valid_play:
                        break
            if is_valid_play:
                actions.append([i, j])
    
        if len(actions) == 0:
            actions.append(Othello.NOOP_ACTION)

        return actions

    def step(self, action: List[int]) -> bool:
        """
        Execute a move for the current player.
        
        If the action is not a NOOP_ACTION, place the piece on the board,
        update the player's piece count, and flip the opponent's discs as appropriate.
        If the action is a NOOP_ACTION and the previous player already skipped,
        the game is marked as done.
        
        Args:
            action (List[int]): The action to perform represented as [i, j] coordinates.
        
        Returns:
            bool: True if the game has ended (both players skipped consecutively or all positions on the board are filled), False otherwise.
        """
        done: bool = False
        if action != Othello.NOOP_ACTION:
            if self.player == Othello.BLACK:
                self.black_player_num_pieces -= 1
            else:
                self.white_player_num_pieces -= 1

            assert self.black_player_num_pieces >= 0 and self.white_player_num_pieces >= 0, "Player piece count cannot be negative. black {}, white {}".format(self.black_player_num_pieces, self.white_player_num_pieces)

            i, j = action
            self.board[i][j] = self.player

            # Flip enemy discs in all eight directions
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    for k in range(1, Othello.BOARD_SIZE):
                        ip, jp = i + k * di, j + k * dj
                        if not self.coordinate_in_bounds(ip, jp):
                            break
                        if self.board[ip][jp] == Othello.EMPTY:
                            break
                        if self.board[ip][jp] == self.player:
                            for kp in range(1, k):
                                ip, jp = i + kp * di, j + kp * dj
                                self.board[ip][jp] = self.player
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

    def board_is_full(self) -> bool:
        """
        Check if the game board is completely filled.
        
        Returns:
            bool: True if the board is full, False otherwise.
        """
        # return if no cell is empty
        return all(cell != Othello.EMPTY for row in self.board for cell in row)

    def opposite_player(self, player: int) -> int:
        """
        Get the opposite player.
        
        Args:
            player (int): The current player (WHITE or BLACK).
        
        Returns:
            int: The opposing player's identifier.
        """
        return Othello.WHITE if player == Othello.BLACK else Othello.BLACK

    def coordinate_in_bounds(self, i: int, j: int) -> bool:
        """
        Check if a coordinate is within the bounds of the board.
        
        Args:
            i (int): The row index.
            j (int): The column index.
        
        Returns:
            bool: True if the coordinates are within the board, False otherwise.
        """
        return 0 <= i < Othello.BOARD_SIZE and 0 <= j < Othello.BOARD_SIZE
