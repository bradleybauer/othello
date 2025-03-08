import argparse
import torch
import numpy as np
import random
import torch.multiprocessing as mp

import othello
from othello_env import OthelloEnv
from policy_function import Policy

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint

def play_game(current_policy, opponent_policy):
    """
    Plays one game using the given policies.
    Returns:
        1 if current_policy wins,
        0 if draw,
       -1 if current_policy loses.
    """
    current_policy.eval()
    opponent_policy.eval()
    
    env = OthelloEnv(opponent=opponent_policy)
    state, info = env.reset()
    done = False

    while not done:
        state_tensor = torch.from_numpy(state).reshape(1, -1).float()
        mask = torch.from_numpy(info['action_mask'])
        flat_action = current_policy.select_action(state_tensor, mask)
        action = env.inflate_action(flat_action.item())
        state, reward, done, _, info = env.step(action)
    
    if reward > 0:
        return 1   # win
    elif reward == 0:
        return 0   # draw
    else:
        return -1  # loss

def evaluate_policy_worker(chunk_size, policy_state, opponent_state):
    """
    Worker function that plays a given number of games (chunk_size)
    and returns the number of wins, draws, and losses.
    """
    wins, draws, losses = 0, 0, 0
    # Create local policy objects and load the states.
    policy = Policy(othello.BOARD_SIZE**2)
    opponent = Policy(othello.BOARD_SIZE**2)
    policy.load_state_dict(policy_state)
    opponent.load_state_dict(opponent_state)
    
    for _ in range(chunk_size):
        outcome = play_game(policy, opponent)
        if outcome == 1:
            wins += 1
        elif outcome == 0:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses

def evaluate_policy(policy_state, opponent_state, num_games=50):
    """
    Splits the total number of games into 16 chunks and processes them in parallel.
    Aggregates and returns the overall wins, draws, and losses.
    """
    num_chunks = 16
    # Determine chunk sizes (distribute remainder across the first few chunks)
    chunk_sizes = [num_games // num_chunks] * num_chunks
    for i in range(num_games % num_chunks):
        chunk_sizes[i] += 1

    # Launch a process pool to evaluate games in parallel.
    with mp.Pool(processes=num_chunks) as pool:
        # starmap passes each tuple (chunk_size, policy_state, opponent_state) to the worker.
        results = pool.starmap(evaluate_policy_worker, 
                               [(chunk, policy_state, opponent_state) for chunk in chunk_sizes])
    
    wins_total = sum(result[0] for result in results)
    draws_total = sum(result[1] for result in results)
    losses_total = sum(result[2] for result in results)
    return wins_total, draws_total, losses_total

def main():
    parser = argparse.ArgumentParser(
        description="Compare the best policies from two training checkpoints by head-to-head evaluation."
    )
    parser.add_argument("checkpoint1", type=str, help="Path to first checkpoint")
    parser.add_argument("checkpoint2", type=str, help="Path to second checkpoint")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games to play per direction (default: 50)")
    args = parser.parse_args()

    # Load both checkpoints.
    cp1 = load_checkpoint(args.checkpoint1)
    cp2 = load_checkpoint(args.checkpoint2)

    # Extract best policy states from each checkpoint.
    best_policy_state1 = cp1["best_policy_state"]
    best_policy_state2 = cp2["best_policy_state"]

    print("Evaluating")
    wins1, draws1, losses1 = evaluate_policy(best_policy_state1, best_policy_state2, num_games=args.games)
    print(f"Results: {wins1} wins, {draws1} draws, {losses1} losses.")

    total_games = args.games
    overall_wins1 = wins1         # current policy wins
    overall_wins2 = losses1       # opponent wins (since each opponent win is a loss for current)

    print("\nOverall win rates:")
    print(f"Policy from {args.checkpoint1}: {overall_wins1}/{total_games} wins ({(overall_wins1/total_games)*100:.2f}%)")
    print(f"Policy from {args.checkpoint2}: {overall_wins2}/{total_games} wins ({(overall_wins2/total_games)*100:.2f}%)")

if __name__ == "__main__":
    # Optionally set seeds for reproducibility during evaluation.
    random.seed(124132)
    np.random.seed(124132)
    torch.manual_seed(124132)
    main()
