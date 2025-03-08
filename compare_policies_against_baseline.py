import argparse
import torch
import numpy as np
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from train_ppo import EloManager

import othello
from othello_env import OthelloEnv
from policy_function import Policy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def evaluate_policy(policy_state, opponent_state, num_games=10):
    num_chunks = 16
    chunk_sizes = [num_games // num_chunks] * num_chunks
    for i in range(num_games % num_chunks):
        chunk_sizes[i] += 1

    wins_total = 0
    draws_total = 0
    losses_total = 0
    for chunk in chunk_sizes:
        wins, draws, losses = evaluate_policy_worker(chunk, policy_state, opponent_state)
        wins_total += wins
        draws_total += draws
        losses_total += losses
    return wins_total, draws_total, losses_total

def evaluate_candidate(args):
    idx, candidate_policy_state, baseline_policy_state, num_games = args
    wins, draws, losses = evaluate_policy(candidate_policy_state, baseline_policy_state, num_games=num_games)
    win_rate = (wins / num_games) * 100
    print(f"Policy {idx}: {wins} wins, {draws} draws, {losses} losses (Win rate: {win_rate:.2f}%)")
    return idx, win_rate, wins, draws, losses

def main():
    parser = argparse.ArgumentParser(
        description="Compare each 100th policy from the policy pool against the baseline (index 0) and plot performance."
    )
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint containing the policy pool (elo_manager)")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play per evaluation (default: 10)")
    args = parser.parse_args()

    # Load checkpoint and initialize EloManager.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    em = EloManager()
    em.load_state_dict(checkpoint["elo_manager"])
    
    policy_pool = em.pool
    if not policy_pool:
        print("Policy pool is empty.")
        return

    # Baseline policy is the policy at index 0.
    baseline_policy_state = policy_pool[0]["policy_params"]

    candidate_args = []
    # Prepare evaluation arguments for every 100th candidate policy.
    for idx in range(10, len(policy_pool), 10):
        candidate_policy_state = policy_pool[idx]["policy_params"]
        candidate_args.append((idx, candidate_policy_state, baseline_policy_state, args.games))
    
    # Use one process pool to evaluate candidate policies concurrently.
    with mp.Pool() as pool:
        results = pool.map(evaluate_candidate, candidate_args)

    results.sort(key=lambda x: x[0])
    indices = [r[0] for r in results]
    win_rates = [r[1] for r in results]

    # Plot the results.
    plt.figure(figsize=(10, 6))
    plt.plot(indices, win_rates, marker="o", linestyle='-')
    plt.xlabel("Policy Index")
    plt.ylabel("Baseline Win Rate (%)")
    plt.title("Baseline Performance vs. Candidate Policy Index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set seeds for reproducibility.
    random.seed(124132)
    np.random.seed(124132)
    torch.manual_seed(124132)
    main()
