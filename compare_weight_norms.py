import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from train_ppo import EloManager
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def compute_weight_norm(policy_state):
    """Compute the L2 norm of the policy weights."""
    norm_sq = 0.0
    for key in policy_state:
        norm_sq += torch.sum(policy_state[key] ** 2)
    return torch.sqrt(norm_sq).item()

def compute_avg_abs_max_weight(policy_state):
    """
    Compute the average of the maximum absolute weight value for each parameter tensor.
    For each key in the policy state, the maximum absolute value is computed,
    and then these values are averaged over all parameters.
    """
    max_values = []
    for key in policy_state:
        max_val = torch.max(torch.abs(policy_state[key])).item()
        max_values.append(max_val)
    return sum(max_values) / len(max_values) if max_values else 0.0

def main():
    parser = argparse.ArgumentParser(
        description="Compute the L2 weight norm and average absolute max weight for all policies in the pool."
    )
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint containing the policy pool (elo_manager)")
    args = parser.parse_args()

    # Load checkpoint and initialize EloManager.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    em = EloManager()
    em.load_state_dict(checkpoint["elo_manager"])
    
    policy_pool = em.pool
    if not policy_pool:
        print("Policy pool is empty.")
        return

    indices = []
    weight_norms = []
    avg_abs_max_weights = []
    
    # Compute metrics for every policy in the pool.
    for idx, policy_info in enumerate(policy_pool):
        policy_state = policy_info["policy_params"]
        norm = compute_weight_norm(policy_state)
        avg_max = compute_avg_abs_max_weight(policy_state)
        indices.append(idx)
        weight_norms.append(norm)
        avg_abs_max_weights.append(avg_max)
        print(f"Policy {idx}: Weight L2 Norm = {norm:.4f}, Avg Abs Max Weight = {avg_max:.4f}")

    # Plot the weight norms and average absolute max weights.
    plt.figure(figsize=(12, 10))
    
    # Subplot for weight norms.
    plt.subplot(2, 1, 1)
    plt.plot(indices, weight_norms, marker="o", linestyle='-')
    plt.xlabel("Policy Index")
    plt.ylabel("Weight Norm (L2)")
    plt.title("L2 Norm of Policy Weights for All Policies")
    plt.grid(True)
    
    # Subplot for average absolute max weights.
    plt.subplot(2, 1, 2)
    plt.plot(indices, avg_abs_max_weights, marker="o", linestyle='-')
    plt.xlabel("Policy Index")
    plt.ylabel("Average Absolute Max Weight")
    plt.title("Average Absolute Max Value of Policy Weights for All Policies")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set seeds for reproducibility.
    random.seed(124132)
    np.random.seed(124132)
    torch.manual_seed(124132)
    main()
