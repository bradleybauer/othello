#!/usr/bin/env python3
import random
from othello import Othello
import othello

def check_board_dimensions(game):
    """Verify that the board has the proper dimensions."""
    assert len(game.board) == othello.BOARD_SIZE, "Board row count mismatch"
    for row in game.board:
        assert len(row) == othello.BOARD_SIZE, "Board column count mismatch"

def check_cell_values(game):
    """Verify that each cell contains a valid value."""
    allowed = {othello.EMPTY, othello.WHITE, othello.BLACK}
    for row in game.board:
        for cell in row:
            assert cell in allowed, "Invalid cell value detected"

def check_piece_placement(prev_board, game, action, prev_player):
    """
    Verify that when a move is made (not a NOOP), a piece is placed at the designated location,
    and that the cell was empty prior to the move.
    """
    if action != othello.NOOP_ACTION:
        i, j = action
        # Ensure that the cell was empty before the move.
        assert prev_board[i][j] == othello.EMPTY, f"Cell ({i},{j}) was not empty before move."
        # Ensure that the piece is now correctly placed.
        assert game.board[i][j] == prev_player, "Piece not placed correctly on the board."

def verify_piece_increase(prev_total, game, action):
    """
    Verify that when a piece is played (action is not NOOP), the total number
    of pieces on the board increases by 1.
    """
    current_total = game.get_score(othello.WHITE) + game.get_score(othello.BLACK)
    if action != othello.NOOP_ACTION:
        assert current_total == prev_total + 1, (
            f"Expected total pieces to increase by 1, but increased from {prev_total} to {current_total}"
        )
    else:
        assert current_total == prev_total, (
            f"Expected total pieces to remain the same on NOOP, but changed from {prev_total} to {current_total}"
        )

def check_flips(prev_board, game, action, prev_player):
    """
    For each direction from the played cell, verify that if there is a valid chain of enemy pieces
    terminated by a piece of prev_player, then those enemy pieces have been flipped.
    """
    if action == othello.NOOP_ACTION:
        return

    i, j = action
    enemy = game.opposite_player(prev_player)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            chain = []
            k = 1
            while True:
                ip = i + k * di
                jp = j + k * dj
                if not game.coordinate_in_bounds(ip, jp):
                    # Out-of-bound: if a chain was built, none should have been flipped.
                    for (ci, cj) in chain:
                        assert game.board[ci][cj] == enemy, (
                            f"Cell ({ci},{cj}) incorrectly flipped (chain terminated out-of-bound)."
                        )
                    break
                cell_prev = prev_board[ip][jp]
                if cell_prev == enemy:
                    chain.append((ip, jp))
                    k += 1
                    continue
                elif cell_prev == prev_player:
                    if chain:
                        for (ci, cj) in chain:
                            assert game.board[ci][cj] == prev_player, (
                                f"Cell ({ci},{cj}) in valid chain not flipped to {prev_player}."
                            )
                    break
                else:  # cell_prev is EMPTY
                    for (ci, cj) in chain:
                        assert game.board[ci][cj] == enemy, (
                            f"Cell ({ci},{cj}) incorrectly flipped (chain terminated with empty cell)."
                        )
                    break

def check_player_switch(prev_player, game):
    """Verify that after the move, the current player is the opposite of the one who just moved."""
    expected = game.opposite_player(prev_player)
    assert game.player == expected, f"Player did not switch correctly: expected {expected}, got {game.player}"

def check_noop_termination(prev_skipped, action, done):
    """
    If the action is NOOP and the previous move was also a NOOP, then the game should be terminated.
    """
    if action == othello.NOOP_ACTION and prev_skipped:
        assert done, "Game did not terminate after consecutive NOOP actions."

def check_done_only_if_double_noop_or_game_board_full(done, action, prev_skipped, board_full):
    """
    The game should be done only if the action is NOOP and the previous move was also a NOOP,
    or if the board is full.
    """
    if done:
        assert board_full or (action == othello.NOOP_ACTION and prev_skipped), "Game done without double NOOP or full board"

def check_noop_only_if_no_other_actions(legal_actions):
    """
    If the action is NOOP and the previous move was also a NOOP, then the game should be terminated.
    """
    if othello.NOOP_ACTION in legal_actions:
        assert len(legal_actions) == 1, "Only possible to play NOOP if there is no other possible move"

def simulate_random_game():
    """
    Run one random game of othello with early exit if the game is done.
    Self-consistency checks are performed before and after each move.
    """
    game = Othello()
    turn_count = 0
    max_turns = 64 * 2
    done = False

    while not done and turn_count <= max_turns:
        # Pre-move checks.
        check_board_dimensions(game)
        check_cell_values(game)
        prev_total = game.get_score(othello.WHITE) + game.get_score(othello.BLACK)

        legal_actions = game.get_legal_actions(game.player)
        prev_skipped = game.previous_player_skipped
        action = random.choice(legal_actions)

        # Snapshot the board and player before making the move.
        prev_board = [row.copy() for row in game.board]
        prev_player = game.player

        done = game.step(action)

        # Post-move checks.
        check_done_only_if_double_noop_or_game_board_full(done, action, prev_skipped, game.board_is_full())
        check_noop_termination(prev_skipped, action, done)
        if action != othello.NOOP_ACTION:
            check_piece_placement(prev_board, game, action, prev_player)
        verify_piece_increase(prev_total, game, action)
        check_player_switch(prev_player, game)
        check_flips(prev_board, game, action, prev_player)

        turn_count += 1

    # Ensure the game did not simply end because of hitting the turn limit.
    assert turn_count <= max_turns, "Game simulation terminated due to reaching the maximum turn count."

###############################################################################
# Fuzz Tests - deliberately corrupt game states to ensure verification checks work.
###############################################################################

def fuzz_test_cell_values():
    game = Othello()
    # Insert an invalid value.
    game.board[0][0] = 999
    try:
        check_cell_values(game)
    except AssertionError as e:
        pass
    else:
        print("Fuzz cell values test FAILED: no assertion error detected.")

def fuzz_test_piece_placement():
    game = Othello()
    legal_actions = game.get_legal_actions(game.player)
    action = None
    for a in legal_actions:
        if a != othello.NOOP_ACTION:
            action = a
            break
    if action is None:
        print("Fuzz piece placement test skipped: no valid move found.")
        return
    prev_board = [row.copy() for row in game.board]
    prev_player = game.player
    # Instead of executing a proper move, manually corrupt the board.
    i, j = action
    game.board[i][j] = game.opposite_player(prev_player)  # Wrong piece placed.
    try:
        check_piece_placement(prev_board, game, action, prev_player)
    except AssertionError as e:
        pass
    else:
        print("Fuzz piece placement test FAILED: no assertion error detected.")

def fuzz_test_piece_increase():
    game = Othello()
    prev_total = game.get_score(othello.WHITE) + game.get_score(othello.BLACK)
    legal_actions = game.get_legal_actions(game.player)
    action = None
    for a in legal_actions:
        if a != othello.NOOP_ACTION:
            action = a
            break
    if action is None:
        print("Fuzz piece increase test skipped: no valid move found.")
        return
    # Execute the move normally.
    game.step(action)
    # Now corrupt the board: remove the newly placed piece.
    i, j = action
    game.board[i][j] = othello.EMPTY
    try:
        verify_piece_increase(prev_total, game, action)
    except AssertionError as e:
        pass
    else:
        print("Fuzz verify piece increase test FAILED: no assertion error detected.")

def fuzz_test_flips():
    game = Othello()
    legal_actions = game.get_legal_actions(game.player)
    action = None
    for a in legal_actions:
        if a != othello.NOOP_ACTION:
            action = a
            break
    if action is None:
        print("Fuzz check flips test skipped: no valid move found.")
        return
    prev_board = [row.copy() for row in game.board]
    prev_player = game.player
    game.step(action)
    i, j = action
    enemy = game.opposite_player(prev_player)
    chain_found = False
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            chain = []
            k = 1
            while True:
                ip = i + k * di
                jp = j + k * dj
                if not game.coordinate_in_bounds(ip, jp):
                    break
                cell_prev = prev_board[ip][jp]
                if cell_prev == enemy:
                    chain.append((ip, jp))
                    k += 1
                    continue
                elif cell_prev == prev_player and chain:
                    chain_found = True
                    # Corrupt the board: revert the flip in the middle of the chain.
                    ci, cj = chain[(len(chain))// 2]
                    game.board[ci][cj] = enemy
                    try:
                        check_flips(prev_board, game, action, prev_player)
                    except AssertionError as e:
                        pass
                    else:
                        print("Fuzz check flips test FAILED: no assertion error detected.")
                    break
                else:
                    break
            if chain_found:
                break
        if chain_found:
            break
    if not chain_found:
        print("Fuzz check flips test skipped: no valid flip chain found.")

def fuzz_test_player_switch():
    game = Othello()
    legal_actions = game.get_legal_actions(game.player)
    action = None
    for a in legal_actions:
        if a != othello.NOOP_ACTION:
            action = a
            break
    if action is None:
        print("Fuzz player switch test skipped: no valid move found.")
        return
    prev_player = game.player
    game.step(action)
    # Fuzz: set player back to prev_player instead of switching.
    game.player = prev_player
    try:
        check_player_switch(prev_player, game)
    except AssertionError as e:
        pass
    else:
        print("Fuzz player switch test FAILED: no assertion error detected.")

def fuzz_test_noop_termination():
    # Simulate a case where action is NOOP, previous move was skipped, but done is False.
    prev_skipped = True
    action = othello.NOOP_ACTION
    done = False
    try:
        check_noop_termination(prev_skipped, action, done)
    except AssertionError as e:
        pass
    else:
        print("Fuzz NOOP termination test FAILED: no assertion error detected.")

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    NUM_GAMES = 11111
    for game_num in range(1, NUM_GAMES + 1):
        try:
            simulate_random_game()
        except AssertionError as e:
            print(f"Game {game_num} failed consistency check: {e}")
            break

    fuzz_test_cell_values()
    fuzz_test_piece_placement()
    fuzz_test_piece_increase()
    fuzz_test_flips()
    fuzz_test_player_switch()
    fuzz_test_noop_termination()
