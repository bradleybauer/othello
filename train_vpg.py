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
    args is a tuple: (i, j, policy_i_state, policy_j_state, elo_i, elo_j, num_games, K)
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

def evaluate_historical_pool_parallel(pool, seed, num_games=4, K=32, min_elo_threshold=10):
    """
    Run a round-robin tournament among all policies in the pool in parallel.
    Each unique pair is evaluated concurrently; the Elo adjustments are then aggregated and applied.
    Policies with Elo below min_elo_threshold are removed.
    
    Run this function under a torch.no_grad() context.
    """
    n = len(pool)
    orig_elos = [entry['elo'] for entry in pool]
    adjustments = [0.0] * n
    pair_args = []
    for i in range(n):
        for j in range(i+1, n):
            pair_args.append((
                i,
                j,
                pool[i]['params'],
                pool[j]['params'],
                orig_elos[i],
                orig_elos[j],
                num_games,
                K,
                seed + i * n + j
            ))
    with mp.Pool() as p:
        results = p.map(simulate_pair_match, pair_args)
    for (i, j, delta_i, delta_j) in results:
        adjustments[i] += delta_i
        adjustments[j] += delta_j
    for i in range(n):
        pool[i]['elo'] = orig_elos[i] + adjustments[i]
    pool = [entry for entry in pool if entry['elo'] >= min_elo_threshold]
    return pool

# -------------------------------
# Experience Generation and Training Functions
# -------------------------------

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x = [x0, 
                    x1, 
                    x2]

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
    Returns a tuple: (states, actions, masks, rewards, wins).
    """
    with torch.no_grad():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        policy = Policy(othello.BOARD_SIZE**2)
        policy.load_state_dict(policy_params)
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
                flat_action = policy.select_action(state_tensor, mask)
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages
        returns = torch.hstack(returns)
        return (states, actions, masks, returns, advantages, wins)

# -------------------------------
# Saving Functions
# -------------------------------

def save_latest_model(policy_model, value_model):
    """
    Save the latest training model (policy and value) to separate weight files and ONNX exports.
    """
    torch.save(policy_model.state_dict(), "latest_model.pth")
    torch.save(value_model.state_dict(), "latest_value_model.pth")
    with torch.no_grad():
        dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
        latest_policy = Policy(othello.BOARD_SIZE**2)
        latest_policy.load_state_dict(policy_model.state_dict())
        torch.onnx.export(
            latest_policy,
            (dummy_state,),
            "latest_model.onnx",
            input_names=["state"],
            output_names=["logits"],
            opset_version=11
        )
        latest_value = Value(othello.BOARD_SIZE**2)
        latest_value.load_state_dict(value_model.state_dict())
        torch.onnx.export(
            latest_value,
            (dummy_state,),
            "latest_value.onnx",
            input_names=["state"],
            output_names=["value"],
            opset_version=11
        )

def save_latest_max_elo_model(max_policy):
    """
    Save the latest max Elo model from the pool to separate weight files and ONNX exports.
    """
    torch.save(max_policy['params'], "latest_max_elo_model.pth")
    torch.save(max_policy['value_params'], "latest_max_elo_value_model.pth")
    with torch.no_grad():
        dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
        model = Policy(othello.BOARD_SIZE**2)
        model.load_state_dict(max_policy['params'])
        torch.onnx.export(
            model,
            (dummy_state,),
            "latest_max_elo_model.onnx",
            input_names=["state"],
            output_names=["logits"],
            opset_version=11
        )
        value = Value(othello.BOARD_SIZE**2)
        value.load_state_dict(max_policy['value_params'])
        torch.onnx.export(
            value,
            (dummy_state,),
            "latest_max_elo_value_model.onnx",
            input_names=["state"],
            output_names=["value"],
            opset_version=11
        )

def save_best_elo_model(max_policy):
    """
    Save the best Elo ever seen model to separate weight files and ONNX exports.
    """
    torch.save(max_policy['params'], "best_elo_model.pth")
    torch.save(max_policy['value_params'], "best_elo_value_model.pth")
    with torch.no_grad():
        dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
        model = Policy(othello.BOARD_SIZE**2)
        model.load_state_dict(max_policy['params'])
        torch.onnx.export(
            model,
            (dummy_state,),
            "best_elo_model.onnx",
            input_names=["state"],
            output_names=["logits"],
            opset_version=11
        )
        value = Value(othello.BOARD_SIZE**2)
        value.load_state_dict(max_policy['value_params'])
        torch.onnx.export(
            value,
            (dummy_state,),
            "best_elo_value_model.onnx",
            input_names=["state"],
            output_names=["value"],
            opset_version=11
        )

# -------------------------------
# Main Training Loop
# -------------------------------

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)
    value_model.to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_model.parameters(), lr=1e-3)

    gamma = .99
    lam = .95

    num_iterations = 3000
    num_workers = 16
    rollouts_per_worker = 512 // num_workers
    total_rollouts = num_workers * rollouts_per_worker

    best_elo = -float('inf')
    
    # Historical pool: each entry has a unique name, policy params, value function params, and an Elo.
    pool_size = 50000
    initial_elo = 1200
    policy_params_pool = []
    initial_policies = 2
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
    win_threshold = 0.63  # Not used directly for saving weights.

    pool = mp.Pool(processes=num_workers)
    for iteration in range(num_iterations):
        # Run training rollouts in parallel.
        value_model.cpu()
        args = [
            (copy.deepcopy(policy_model.state_dict()), copy.deepcopy(value_model.state_dict()), policy_params_pool, gamma, lam, rollouts_per_worker, seed+i)
            for i in range(num_workers)
        ]
        seed += num_workers
        value_model.to(device)
        wins = 0
        policy_loss = 0
        value_loss = 0
        results = pool.starmap(generate_experience, args)
        policy_model.to(device)
        for states, actions, masks, returns, advantages, wins_ in results:
            states = states.to(device)
            log_probs = policy_model.log_probs(states, actions.to(device), masks.to(device))
            policy_loss += -torch.sum(log_probs * advantages.to(device))
            value_loss += torch.sum((value_model(states).squeeze() - returns.to(device)) ** 2)
            wins += wins_
        
        policy_loss /= total_rollouts
        value_loss /= total_rollouts

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        policy_model.cpu()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        win_percentage = wins / total_rollouts
        print(f"Iteration {iteration}: PLoss = {policy_loss.item():.3f}, "
              f"VLoss = {value_loss.item():.3f}, Train win% = {win_percentage:.3f}, "
              f"Pool size = {len(policy_params_pool)}.")

        # Always save the current training model as the latest model.
        save_latest_model(policy_model, value_model)

        if win_percentage >= win_threshold:
            saturation_counter += 1
        else:
            saturation_counter = 0

        if saturation_counter >= saturation_threshold:
            new_policy_name = f"policy_{policy_counter}"
            policy_counter += 1
            print(f"Adding current policy as {new_policy_name} to historical pool.")
            if len(policy_params_pool) >= pool_size:
                lowest_index = min(range(len(policy_params_pool)), key=lambda i: policy_params_pool[i]['elo'])
                removed_policy = policy_params_pool.pop(lowest_index)
                print(f"Removed {removed_policy['name']} with Elo {removed_policy['elo']:.1f}")
            value_model.cpu()
            policy_params_pool.append({
                'name': new_policy_name,
                'params': copy.deepcopy(policy_model.state_dict()),
                'value_params': copy.deepcopy(value_model.state_dict()),
                'elo': initial_elo
            })
            value_model.to(device)
            saturation_counter = 0

            with torch.no_grad():
                policy_params_pool = evaluate_historical_pool_parallel(policy_params_pool, seed)
            seed += len(policy_params_pool)**2
            print("Historical pool tournament complete. Elo ratings:")
            for entry in policy_params_pool:
                print(f"  {entry['name']}: Elo = {entry['elo']:.1f}")

            max_policy = max(policy_params_pool, key=lambda entry: entry['elo'])
            # Save the latest max Elo model regardless.
            save_latest_max_elo_model(max_policy)
            if max_policy['elo'] > best_elo:
                best_elo = max_policy['elo']
                print(f"New best policy is {max_policy['name']} with Elo {max_policy['elo']:.1f}. Saving best Elo weights.")
                save_best_elo_model(max_policy)

        # if iteration % 100 == 0 and len(policy_params_pool) > 1:
        #     with torch.no_grad():
        #         policy_params_pool = evaluate_historical_pool_parallel(policy_params_pool, seed)
        #     seed += len(policy_params_pool)**2
        #     print("Periodic historical pool tournament complete. Elo ratings:")
        #     for entry in policy_params_pool:
        #         print(f"  {entry['name']}: Elo = {entry['elo']:.1f}")
        #     max_policy = max(policy_params_pool, key=lambda entry: entry['elo'])
        #     save_latest_max_elo_model(max_policy)
        #     if max_policy['elo'] > best_elo:
        #         best_elo = max_policy['elo']
        #         print(f"New best policy is {max_policy['name']} with Elo {max_policy['elo']:.1f}. Saving best Elo weights.")
        #         save_best_elo_model(max_policy)

    pool.close()
    pool.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
