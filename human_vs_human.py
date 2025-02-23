import random
from policy import Policy
from othello_env import OthelloEnv  # Assumes the environment is defined in othello_env.py
from othello import Othello         # To access Othello.WHITE, Othello.BLACK, etc.
import torch

def main():
    env = OthelloEnv()
    obs, info = env.reset()
    done = False

    while not done:
        # Identify the current player.
        current_player = env.game.player
        player_name = "White" if current_player == Othello.WHITE else "Black"
        print(f"\n{player_name}'s turn.")

        p = Policy(env.board_size)

        # Get legal moves for the current player.
        legal_moves = env.game.get_legal_actions(current_player)
        
        # Check if the only legal move is the NOOP action.
        if legal_moves == [Othello.NOOP_ACTION]:
            print("No legal moves available. Turn will be skipped.\n")
            action = Othello.NOOP_ACTION
            # Render the current board state.
            env.render()

            obs_torch = torch.from_numpy(obs).float().unsqueeze(0)
            mask = torch.from_numpy(info['action_mask']).float().unsqueeze(0)
            act, logps = p.select_action(obs_torch, mask)
            print(act, logps, done)
            input()
        else:
            # Display the legal moves.
            print("Legal moves (row col):")
            for move in legal_moves:
                print(f"  {move[0]} {move[1]}")
            print()
            # Render the current board state.
            env.render()

            obs_torch = torch.from_numpy(obs).float().unsqueeze(0)
            mask = torch.from_numpy(info['action_mask']).float().unsqueeze(0)
            act, logps = p.select_action(obs_torch, mask)
            print(act)

            # Prompt the user for input until a valid move is provided.
            valid_input = False
            while not valid_input:
                move_input = input("Enter your move as 'row col': ").strip()
                try:
                    row, col = map(int, move_input.split())
                    action = [row, col]
                    if action in legal_moves:
                        valid_input = True
                    else:
                        print("That move is not legal. Please choose one of the listed moves.")
                except Exception:
                    print("Invalid input. Please enter two integers separated by a space.")
            # action = random.choice(legal_moves)

        # Take the action in the environment.
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("\n" + "-" * 40 + "\n")

    # Game over; display the final board and score.
    print("Game over!")
    env.render()
    white_score = env.game.get_score(Othello.WHITE)
    black_score = env.game.get_score(Othello.BLACK)
    print(f"\nFinal Score:\n  White: {white_score}\n  Black: {black_score}")
    
    if white_score > black_score:
        print("White wins!")
    elif black_score > white_score:
        print("Black wins!")
    else:
        print("It's a tie!")

if __name__ == '__main__':
    main()
