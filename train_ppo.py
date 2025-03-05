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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def update_elo(elo1, elo2, score1, K=32):
    """
    Update Elo ratings for two policies based on the match result.
    score1 is 1 for a win by policy1, 0 for a loss, and 0.5 for a draw.
    Returns new elo values: (new_elo1, new_elo2)
    """
    expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected2 = 1 - expected1
    new_elo1 = elo1 + K * (score1 - expected1)
    new_elo2 = elo2 + K * ((1 - score1) - expected2)
    return new_elo1, new_elo2

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
    Now also counts draws and losses.
    Returns a tuple:
      (states, actions, masks, returns, advantages, wins, draws, losses,
       wins_vector, draws_vector, losses_vector, plays_vector)
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
        draws = 0
        losses = 0
        wins_vector = torch.zeros(len(pool_policy_params), dtype=int)
        draws_vector = torch.zeros(len(pool_policy_params), dtype=int)
        losses_vector = torch.zeros(len(pool_policy_params), dtype=int)
        plays_vector = torch.zeros(len(pool_policy_params), dtype=int)

        states, actions, masks, returns, advantages = [], [], [], [], []
        for _ in range(num_rollouts):
            rollout_states, rollout_actions, rollout_masks, rollout_rewards = [], [], [], []
            # Sample an opponent by index so that we know its position.
            opponent_index = random.randint(0, len(pool_policy_params) - 1)
            sampled_opponent = pool_policy_params[opponent_index]
            opponent_policy = Policy(othello.BOARD_SIZE**2)
            opponent_policy.load_state_dict(sampled_opponent['policy_params'])
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
            
            # Record outcome for the sampled opponent.
            if reward > 0:
                wins += 1
                wins_vector[opponent_index] += 1
            elif reward == 0:
                draws += 1
                draws_vector[opponent_index] += 1
            else:
                losses += 1
                losses_vector[opponent_index] += 1
            plays_vector[opponent_index] += 1

            states.append(rollout_states)
            actions.append(rollout_actions)
            masks.append(rollout_masks)
            advantages.append(rollout_advantages)
            returns.append(rollout_returns)

        states = torch.vstack(states)
        actions = torch.vstack(actions)
        masks = torch.vstack(masks)
        advantages = torch.hstack(advantages)
        returns = torch.hstack(returns)
        return (states, actions, masks, returns, advantages,
                wins, draws, losses,
                wins_vector, draws_vector, losses_vector, plays_vector)

def ppo_clip_loss(policy_model, states: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
                  advantages: torch.Tensor, log_probs_old: torch.Tensor, clip_param: float):
    log_probs, entropy = policy_model.log_probs(states, actions, masks)
    ratio = torch.exp(log_probs - log_probs_old)
    clipped = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    loss = -(torch.min(ratio * advantages, clipped)).mean()
    approximate_kl = (log_probs_old - log_probs).mean().item()
    return loss, approximate_kl, entropy

# -------------------------------
# Main Training Loop
# -------------------------------

def main():
    writer = SummaryWriter(log_dir="runs/othello_experiment_ppo2")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=.0003)
    value_optimizer = optim.Adam(value_model.parameters(), lr=.0005)

    gamma = 0.99
    lam = 0.95
    clip_param = .2
    target_kl = .05
    num_policy_steps = 80
    num_value_steps = 80

    num_iterations = 300000000
    num_workers = 16
    rollouts_per_worker = 1024 // num_workers
    total_rollouts = num_workers * rollouts_per_worker

    initial_elo = 1200
    # The current policy's elo is maintained separately.
    current_policy_elo = initial_elo

    policy_params_pool = []
    initial_policies = 5
    for i in range(initial_policies):
        policy_params_pool.append({
            'name': f"policy_{i}",
            'policy_params': copy.deepcopy(policy_model.state_dict()),
            'value_params': copy.deepcopy(value_model.state_dict()),
            'elo': initial_elo
        })
    policy_counter = initial_policies

    saturation_counter = 0
    saturation_threshold = 20
    win_threshold = 0.51

    # Initialize accumulators for wins, draws, losses and plays; track current pool size.
    prev_pool_size = len(policy_params_pool)
    accum_wins_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_draws_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_losses_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_plays_vector = torch.zeros(prev_pool_size, dtype=int)

    pool = mp.Pool(processes=num_workers)
    for iteration in range(num_iterations):
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

        states_, actions_, masks_, returns_, advantages_ = [], [], [], [], []
        # Create iteration-specific accumulators.
        iteration_wins_vector = torch.zeros(len(policy_params_pool), dtype=int)
        iteration_draws_vector = torch.zeros(len(policy_params_pool), dtype=int)
        iteration_losses_vector = torch.zeros(len(policy_params_pool), dtype=int)
        iteration_plays_vector = torch.zeros(len(policy_params_pool), dtype=int)
        wins_total = 0
        draws_total = 0
        losses_total = 0

        results = pool.starmap(generate_experience, args)
        for (states, actions, masks, returns, advantages,
             wins_, draws_, losses_,
             wins_vector_, draws_vector_, losses_vector_, plays_vector_) in results:
            states_.append(states)
            actions_.append(actions)
            masks_.append(masks)
            returns_.append(returns)
            advantages_.append(advantages)
            wins_total += wins_
            draws_total += draws_
            losses_total += losses_
            iteration_wins_vector += wins_vector_
            iteration_draws_vector += draws_vector_
            iteration_losses_vector += losses_vector_
            iteration_plays_vector += plays_vector_

        states = torch.cat(states_, dim=0).to(device)
        actions = torch.cat(actions_, dim=0).to(device)
        masks = torch.cat(masks_, dim=0).to(device)
        returns = torch.cat(returns_, dim=0).to(device)
        advantages = torch.cat(advantages_, dim=0).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_model.to(device)
        value_model.to(device)

        with torch.no_grad():
            log_probs_old, _ = policy_model.log_probs(states, actions, masks)

        policy_loss = 0
        policy_grad_norm = 0
        for _ in range(num_policy_steps):
            policy_optimizer.zero_grad()
            new_policy_loss, kl, mean_policy_entropy = ppo_clip_loss(policy_model, states, actions, masks, advantages, log_probs_old, clip_param)
            if kl > 1.5 * target_kl:
                break
            new_policy_loss.backward()
            policy_optimizer.step()
            policy_loss += new_policy_loss.item()
            policy_grad_norm += sum(p.grad.data.norm(2).item() ** 2 for p in policy_model.parameters() if p.grad is not None) ** 0.5

        value_loss = 0
        value_grad_norm = 0
        for _ in range(num_value_steps):
            value_optimizer.zero_grad()
            new_value_loss = torch.mean((value_model(states).squeeze() - returns) ** 2)
            new_value_loss.backward()
            value_optimizer.step()
            value_loss += new_value_loss.item()
            value_grad_norm += sum(p.grad.data.norm(2).item() ** 2 for p in value_model.parameters() if p.grad is not None) ** 0.5

        policy_model.cpu()
        value_model.cpu()

        win_percentage = wins_total / total_rollouts

        writer.add_scalar("Training/PolicyLoss", policy_loss, iteration)
        writer.add_scalar("Training/ValueLoss", value_loss, iteration)
        writer.add_scalar("Training/PolicyKL", kl, iteration)
        writer.add_scalar("Training/WinPercentage", win_percentage, iteration)
        writer.add_scalar("Training/AverageReturn", returns.mean().item(), iteration)
        writer.add_scalar("Training/PolicyGradNorm", policy_grad_norm, iteration)
        writer.add_scalar("Training/ValueGradNorm", value_grad_norm, iteration)
        writer.add_scalar("Training/PolicyEntropy", mean_policy_entropy, iteration)

        # Reset accumulators if pool size increased.
        if len(policy_params_pool) != prev_pool_size:
            accum_wins_vector = torch.zeros(len(policy_params_pool), dtype=int)
            accum_draws_vector = torch.zeros(len(policy_params_pool), dtype=int)
            accum_losses_vector = torch.zeros(len(policy_params_pool), dtype=int)
            accum_plays_vector = torch.zeros(len(policy_params_pool), dtype=int)
            prev_pool_size = len(policy_params_pool)
        # Accumulate the iteration counts.
        accum_wins_vector += iteration_wins_vector
        accum_draws_vector += iteration_draws_vector
        accum_losses_vector += iteration_losses_vector
        accum_plays_vector += iteration_plays_vector

        # Generate the histogram of win fraction.
        ratios = []
        for i in range(len(accum_wins_vector)):
            plays = accum_plays_vector[i].item()
            win_ratio = accum_wins_vector[i].item() / plays if plays > 0 else 0
            ratios.append(win_ratio)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(ratios)), ratios)
        plt.xlabel('Policy Index')
        plt.ylabel('Win Fraction')
        plt.title('Accumulated Win/Play Ratio for Historical Policies')
        plt.savefig("wins_histogram.png")  # Overwrite each iteration
        plt.close()

        # Update Elo ratings using the iteration outcomes.
        # For each opponent in the pool, if any games were played, compute average score and update Elo.
        for i in range(len(policy_params_pool)):
            plays = iteration_plays_vector[i].item()
            if plays > 0:
                avg_score = (iteration_wins_vector[i].item() + 0.5 * iteration_draws_vector[i].item()) / plays
                current_policy_elo, updated_opp_elo = update_elo(current_policy_elo, policy_params_pool[i]['elo'], avg_score)
                policy_params_pool[i]['elo'] = updated_opp_elo

        print(f"Iteration {iteration}: PLoss = {policy_loss:.3f}, VLoss = {value_loss:.3f}, " +
              f"Train win% = {win_percentage:.3f}, Pool size = {len(policy_params_pool)}, " +
              f"Current policy Elo: {current_policy_elo:.1f}")

        writer.add_scalar("Training/PolicyElo", current_policy_elo, iteration)

        if win_percentage >= win_threshold:
            saturation_counter += 1
        else:
            saturation_counter = 0

        if saturation_counter >= saturation_threshold:
            save_models(policy_model.state_dict(), value_model.state_dict(), base_name="latest")
            new_policy_name = f"policy_{policy_counter}"
            policy_counter += 1
            # Use current_policy_elo for the new policy's elo.
            print(f"Adding current policy as {new_policy_name} to historical pool with Elo {current_policy_elo:.1f}.")
            policy_params_pool.append({
                'name': new_policy_name,
                'policy_params': copy.deepcopy(policy_model.state_dict()),
                'value_params': copy.deepcopy(value_model.state_dict()),
                'elo': current_policy_elo
            })
            saturation_counter = 0

            for entry in policy_params_pool:
                print(f"  {entry['name']}: Elo = {entry['elo']:.1f}")
            max_policy = max(policy_params_pool, key=lambda entry: entry['elo'])
            save_models(max_policy['policy_params'], max_policy['value_params'], base_name="latest_max_elo")

        if iteration % 100 == 0:
            for name, param in policy_model.named_parameters():
                writer.add_histogram(f"Policy/Params/{name}", param, iteration)
                if param.grad is not None:
                    writer.add_histogram(f"Policy/Grads/{name}", param.grad, iteration)
            for name, param in value_model.named_parameters():
                writer.add_histogram(f"Value/Params/{name}", param, iteration)
                if param.grad is not None:
                    writer.add_histogram(f"Value/Grads/{name}", param.grad, iteration)

    writer.close()
    pool.close()
    pool.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
