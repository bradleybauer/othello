import copy
import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
from policy_function import Policy
from value_function import Value
import torch.multiprocessing as mp

# Import TensorBoard’s SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# -------------------------------
# Utility Saving Function (with ONNX Exporting)
# -------------------------------
def save_models(policy_state, value_state, base_name: str):
    """
    Save the policy and value state dictionaries to PyTorch checkpoints and export them to ONNX.
    
    Args:
        policy_state (dict): State dictionary of the policy model.
        value_state (dict): State dictionary of the value model.
        base_name (str): Base file name to use (e.g., "latest" or "latest_max_elo").
    """
    # Save PyTorch checkpoints.
    torch.save(policy_state, f"{base_name}_policy.pth")
    torch.save(value_state, f"{base_name}_value.pth")
    
    # For ONNX export, create new model instances, load the state dicts, and export.
    dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
    
    # Export policy model.
    policy_model = Policy(othello.BOARD_SIZE**2)
    policy_model.load_state_dict(policy_state)
    torch.onnx.export(
        policy_model,
        dummy_state,
        f"{base_name}_policy.onnx",
        input_names=["state"],
        output_names=["logits"],
        opset_version=11
    )
    
    # Export value model.
    value_model = Value(othello.BOARD_SIZE**2)
    value_model.load_state_dict(value_state)
    torch.onnx.export(
        value_model,
        dummy_state,
        f"{base_name}_value.onnx",
        input_names=["state"],
        output_names=["value"],
        opset_version=11
    )


# -------------------------------
# Tournament Evaluation Functions (Parallel Version)
# -------------------------------

def play_game(controlled_policy_state, opponent_policy_state):
    """
    Simulate one complete game where the controlled policy makes all decisions.
    The opponent policy is passed into the environment via OthelloEnv(opponent=...).
    
    Returns:
      1 if the controlled agent wins,
      0 if the controlled agent loses,
      0.5 in case of a draw.
    """
    controlled_policy = Policy(othello.BOARD_SIZE**2)
    controlled_policy.load_state_dict(controlled_policy_state)
    opponent_policy = Policy(othello.BOARD_SIZE**2)
    opponent_policy.load_state_dict(opponent_policy_state)
    
    env = OthelloEnv(opponent=opponent_policy)
    state, info = env.reset()
    done = False
    while not done:
        state_tensor = torch.from_numpy(state).reshape(1, -1).float()
        mask = torch.from_numpy(info['action_mask'])
        action = controlled_policy.select_action(state_tensor, mask)
        action_inflated = env.inflate_action(action.item())
        state, reward, done, _, info = env.step(action_inflated)
    if reward > 0:
        return 1
    elif reward < 0:
        return 0
    else:
        return 0.5

def simulate_match(policy1_state, policy2_state, num_games=4):
    """
    Simulate a match between two policies by playing an equal number of games with:
      - policy1 as the controlled agent (and policy2 as opponent)
      - policy2 as the controlled agent (and policy1 as opponent)
    
    Returns the win fraction for policy1.
    """
    wins = 0.0
    half_games = num_games // 2
    for _ in range(half_games):
        wins += play_game(policy1_state, policy2_state)
    for _ in range(half_games):
        wins += (1 - play_game(policy2_state, policy1_state))
    return wins / num_games

def update_elo(elo1, elo2, score1, K=32):
    """
    Update Elo ratings for two policies based on the match result.
    score1 is 1 for a win by policy1, 0 for a loss, and 0.5 for a draw.
    """
    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 - expected1
    new_elo1 = elo1 + K * (score1 - expected1)
    new_elo2 = elo2 + K * ((1 - score1) - expected2)
    return new_elo1, new_elo2

def simulate_pair_match(args):
    """
    Worker function for a pair match.
    args is a tuple: (i, j, policy_i_state, policy_j_state, elo_i, elo_j, num_games, K, seed)
    Returns (i, j, delta_i, delta_j) where delta_i and delta_j are Elo adjustments.
    """
    i, j, policy_i_state, policy_j_state, elo_i, elo_j, num_games, K, seed = args
    random.seed(seed)
    np.random.seed(seed)
    score1 = simulate_match(policy_i_state, policy_j_state, num_games)
    new_elo_i, new_elo_j = update_elo(elo_i, elo_j, score1, K)
    delta_i = new_elo_i - elo_i
    delta_j = new_elo_j - elo_j
    return (i, j, delta_i, delta_j)

def evaluate_new_policy_parallel(pool, new_policy_index, seed, num_games=8, K=32):
    """
    Evaluate the new policy (at index new_policy_index) against all previous policies
    (indices 0 to new_policy_index-1) in parallel. Updates the Elo ratings of both the new policy
    and the older policies based solely on their head-to-head matches.
    """
    pair_args = []
    for i in range(new_policy_index):
        pair_args.append((
            i,
            new_policy_index,
            pool[i]['params'],
            pool[new_policy_index]['params'],
            pool[i]['elo'],
            pool[new_policy_index]['elo'],
            num_games,
            K,
            seed + i
        ))
    with mp.Pool() as p:
        results = p.map(simulate_pair_match, pair_args)
    
    adjustments = [0.0] * len(pool)
    for (i, j, delta_i, delta_j) in results:
        adjustments[i] += delta_i
        adjustments[j] += delta_j

    for i in range(new_policy_index):
        pool[i]['elo'] += adjustments[i]
    pool[new_policy_index]['elo'] += adjustments[new_policy_index]
    return pool

# -------------------------------
# Experience Generation and Training Functions
# -------------------------------

def discount_cumsum(x, discount):
    """
    Computes discounted cumulative sums of vectors (from rllab).

    input: 
        vector x = [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    result = torch.zeros_like(x)
    running_sum = 0
    for i in range(len(x) - 1, -1, -1):
        running_sum = x[i] + discount * running_sum
        result[i] = running_sum
    return result

def generate_experience(policy_params, value_params, pool_policy_params, gamma, lam, num_rollouts, seed):
    """
    Worker function to generate experience.
    For each rollout, a random opponent is sampled from the historical pool.
    Returns a tuple: (states, actions, masks, returns, advantages, wins).
    """
    with torch.no_grad():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        policy_model = Policy(othello.BOARD_SIZE**2)
        policy_model.load_state_dict(policy_params)
        value_model = Value(othello.BOARD_SIZE**2)
        value_model.load_state_dict(value_params)

        wins = 0
        states, actions, masks, returns, advantages = [], [], [], [], []
        for _ in range(num_rollouts):
            rollout_states, rollout_actions, rollout_masks, rollout_rewards = [], [], [], []
            sampled_opponent = random.choice(pool_policy_params)
            opponent_policy = Policy(othello.BOARD_SIZE**2)
            opponent_policy.load_state_dict(sampled_opponent['params'])
            env = OthelloEnv(opponent=opponent_policy)
            state, info = env.reset()
            done = False
            while not done:
                state_tensor = torch.from_numpy(state).reshape(1, -1).float()
                mask = torch.from_numpy(info['action_mask'])
                rollout_states.append(state_tensor)
                rollout_masks.append(mask)
                flat_action = policy_model.select_action(state_tensor, mask)
                rollout_actions.append(flat_action)
                action = env.inflate_action(flat_action.item())
                state, reward, done, _, info = env.step(action)
                rollout_rewards.append(reward)
            rollout_states = torch.concatenate(rollout_states, axis=0)
            rollout_actions = torch.stack(rollout_actions)
            rollout_masks = torch.stack(rollout_masks)
            rollout_rewards = torch.tensor(rollout_rewards).float()
            rollout_values = value_model(rollout_states).squeeze()
            # GAE-Lambda advantage calculation
            rollout_returns = discount_cumsum(rollout_rewards, gamma)
            deltas = rollout_rewards + gamma * torch.cat([rollout_values[1:], torch.zeros(1)]) - rollout_values
            rollout_advantages = discount_cumsum(deltas, gamma * lam)
            wins += 1 if reward > 0 else 0

            states.append(rollout_states)
            actions.append(rollout_actions)
            masks.append(rollout_masks)
            advantages.append(rollout_advantages)
            returns.append(rollout_returns)

        states = torch.vstack(states)
        actions = torch.vstack(actions)
        masks = torch.vstack(masks)
        advantages = torch.hstack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.hstack(returns)
        return (states, actions, masks, returns, advantages, wins)

# -------------------------------
# Main Training Loop
# -------------------------------

def main():
    writer = SummaryWriter(log_dir="runs/othello_experiment")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_model.parameters(), lr=1e-3)

    gamma = 0.99
    lam = 0.95
    num_iterations = 3000
    num_workers = 16
    rollouts_per_worker = 512 // num_workers
    total_rollouts = num_workers * rollouts_per_worker

    pool_size = 50000
    initial_elo = 1200
    policy_params_pool = []
    initial_policies = 5
    for i in range(initial_policies):
        policy_params_pool.append({
            'name': f"policy_{i}",
            'params': copy.deepcopy(policy_model.state_dict()),
            'value_params': copy.deepcopy(value_model.state_dict()),
            'elo': initial_elo
        })
    policy_counter = initial_policies

    saturation_counter = 0
    saturation_threshold = 20
    win_threshold = 0.51

    pool = mp.Pool(processes=num_workers)
    for iteration in range(num_iterations):
        value_model.cpu()
        args = [
            (
                copy.deepcopy(policy_model.state_dict()),
                copy.deepcopy(value_model.state_dict()),
                policy_params_pool,
                gamma,
                lam,
                rollouts_per_worker,
                seed + i
            )
            for i in range(num_workers)
        ]
        seed += num_workers
        wins = 0
        policy_loss = 0
        value_loss = 0
        returns_list = []
        entropy_total = 0.0

        results = pool.starmap(generate_experience, args)
        policy_model.to(device)
        value_model.to(device)
        for states, actions, masks, returns, advantages, wins_ in results:
            states = states.to(device)
            masks = masks.to(device)
            returns_list.append(returns)
            log_probs, entropy = policy_model.log_probs(states, actions.to(device), masks)
            policy_loss += -torch.sum(log_probs * advantages.to(device))
            value_loss += torch.sum((value_model(states).squeeze() - returns.to(device)) ** 2)
            wins += wins_
            entropy_total += entropy
        
        policy_loss /= total_rollouts
        value_loss /= total_rollouts
        avg_return = torch.cat(returns_list, dim=0).mean()
        avg_entropy = entropy_total / len(results)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        policy_model.cpu()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        value_model.cpu()

        policy_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in policy_model.parameters() if p.grad is not None) ** 0.5
        value_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in value_model.parameters() if p.grad is not None) ** 0.5
        win_percentage = wins / total_rollouts

        writer.add_scalar("Training/PolicyLoss", policy_loss.item(), iteration)
        writer.add_scalar("Training/ValueLoss", value_loss.item(), iteration)
        writer.add_scalar("Training/WinPercentage", win_percentage, iteration)
        writer.add_scalar("Training/AverageReturn", avg_return.item(), iteration)
        writer.add_scalar("Training/PolicyGradNorm", policy_grad_norm, iteration)
        writer.add_scalar("Training/ValueGradNorm", value_grad_norm, iteration)
        writer.add_scalar("Training/PolicyEntropy", avg_entropy, iteration)

        print(f"Iteration {iteration}: PLoss = {policy_loss.item():.3f}, VLoss = {value_loss.item():.3f}, " +
              f"Train win% = {win_percentage:.3f}, Pool size = {len(policy_params_pool)}.")

        # Save the latest model.
        save_models(policy_model.state_dict(), value_model.state_dict(), base_name="latest")

        if win_percentage >= win_threshold:
            saturation_counter += 1
        else:
            saturation_counter = 0

        if saturation_counter >= saturation_threshold:
            new_policy_name = f"policy_{policy_counter}"
            previous_policy_name = f"policy_{policy_counter - 1}"
            policy_counter += 1
            previous_policy_elo = initial_elo
            for pol in policy_params_pool:
                if pol['name'] == previous_policy_name:
                    previous_policy_elo = pol['elo']
                    break
            print(f"Adding current policy as {new_policy_name} to historical pool.")
            if len(policy_params_pool) >= pool_size:
                lowest_index = min(range(len(policy_params_pool)), key=lambda i: policy_params_pool[i]['elo'])
                removed_policy = policy_params_pool.pop(lowest_index)
                print(f"Removed {removed_policy['name']} with Elo {removed_policy['elo']:.1f}")
            policy_params_pool.append({
                'name': new_policy_name,
                'params': copy.deepcopy(policy_model.state_dict()),
                'value_params': copy.deepcopy(value_model.state_dict()),
                'elo': previous_policy_elo
            })
            saturation_counter = 0

            with torch.no_grad():
                policy_params_pool = evaluate_new_policy_parallel(
                    policy_params_pool,
                    new_policy_index=len(policy_params_pool) - 1,
                    seed=seed
                )
            seed += len(policy_params_pool)
            writer.add_scalar("Elo/LatestPolicy", policy_params_pool[-1]['elo'], iteration)
            for entry in policy_params_pool[:20]:
                writer.add_scalar(f"Elo/{entry['name']}", entry['elo'], iteration)
            print("New policy tournament complete. Updated Elo ratings:")
            for entry in policy_params_pool:
                print(f"  {entry['name']}: Elo = {entry['elo']:.1f}")
            max_policy = max(policy_params_pool, key=lambda entry: entry['elo'])
            save_models(max_policy['params'], max_policy['value_params'], base_name="latest_max_elo")

        if iteration % 100 == 0:
            for name, param in policy_model.named_parameters():
                writer.add_histogram(f"Policy/Params/{name}", param, iteration)
                if param.grad is not None:
                    writer.add_histogram(f"Policy/Grads/{name}", param.grad, iteration)
            for name, param in value_model.named_parameters():
                writer.add_histogram(f"Value/Params/{name}", param, iteration)
                if param.grad is not None:
                    writer.add_histogram(f"Value/Grads/{name}", param.grad, iteration)

    pool.close()
    pool.join()
    writer.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
