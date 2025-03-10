import os
import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv, inflate_action
from policy_function import Policy
from value_function import Value
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(path, iteration, policy_state, value_state,
                    policy_optimizer_state, value_optimizer_state, elo_manager_state,
                    best_elo, best_policy_state, best_value_state):
    checkpoint = {
        "iteration": iteration,
        "policy_state": policy_state,
        "value_state": value_state,
        "policy_optimizer_state": policy_optimizer_state,
        "value_optimizer_state": value_optimizer_state,
        "elo_manager": elo_manager_state,
        "best_elo": best_elo,
        "best_policy_state": best_policy_state,
        "best_value_state": best_value_state,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    return (checkpoint["iteration"],
            checkpoint["policy_state"],
            checkpoint["value_state"],
            checkpoint["policy_optimizer_state"],
            checkpoint["value_optimizer_state"],
            checkpoint["elo_manager"],
            checkpoint["best_elo"],
            checkpoint["best_policy_state"],
            checkpoint["best_value_state"])

class EloManager:
    def __init__(self, initial_elo=1200, smoothing_coef=0.95):
        self.current_policy_elo = initial_elo
        self.smooth_current_policy_elo = initial_elo
        self.smoothing_coef = smoothing_coef
        self.pool = []
        self.policy_counter = 0

    def add_initial_policy(self, name, policy, value):
        entry = {
            'name': name,
            'policy_params': get_cpu_state(policy),
            'value_params': get_cpu_state(value),
            'elo': self.current_policy_elo,
        }
        self.pool.append(entry)
        self.policy_counter += 1

    def update_ratings(self, iteration_wins_vector, iteration_draws_vector, iteration_plays_vector):
        K = 32
        R_current = self.current_policy_elo
        total_delta_current = 0.0

        for i, entry in enumerate(self.pool):
            plays = iteration_plays_vector[i].item()
            if plays > 0:
                s = (iteration_wins_vector[i].item() + 0.5 * iteration_draws_vector[i].item()) / plays
                R_opp = entry['elo']
                E_current = 1 / (1 + 10 ** ((R_opp - R_current) / 400))
                delta_current = K * (s - E_current)
                total_delta_current += delta_current
        self.current_policy_elo = R_current + total_delta_current
        self.smooth_current_policy_elo = (self.smoothing_coef * self.smooth_current_policy_elo +
                                          (1 - self.smoothing_coef) * self.current_policy_elo)

    def add_new_policy(self, policy_state, value_state):
        new_policy_name = f"policy_{self.policy_counter}"
        entry = {
            'name': new_policy_name,
            'policy_params': policy_state,
            'value_params': value_state,
            'elo': self.current_policy_elo,
        }
        self.pool.append(entry)
        self.policy_counter += 1
        print(f"Adding current policy as {new_policy_name} to pool with Elo {self.current_policy_elo:.1f}.")
        self.print_pool_ratings()

    def print_pool_ratings(self):
        for entry in self.pool:
            print(f"  {entry['name']}: Elo = {entry['elo']:.1f}")

    def state_dict(self):
        return {
            "pool": self.pool,
            "current_policy_elo": self.current_policy_elo,
            "smooth_current_policy_elo": self.smooth_current_policy_elo,
            "policy_counter": self.policy_counter,
        }
    
    def load_state_dict(self, state):
        self.pool = state["pool"]
        self.current_policy_elo = state["current_policy_elo"]
        self.smooth_current_policy_elo = state.get("smooth_current_policy_elo", self.current_policy_elo)
        self.policy_counter = state["policy_counter"]

def discount_cumsum(x, discount):
    n = x.shape[0]
    device = x.device
    discount_factors = discount ** torch.arange(n, dtype=x.dtype, device=device)
    discounted_x = x * discount_factors
    discounted_cumsum = torch.flip(torch.cumsum(torch.flip(discounted_x, dims=[0]), dim=0), dims=[0])
    return discounted_cumsum / discount_factors

def get_cpu_state(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def generate_experience(policy_params, value_params, cached_opponent_policies, gamma, lam, num_rollouts, opponent_probs):
    with torch.no_grad():
        policy_model = Policy(othello.BOARD_SIZE**2)
        policy_model.load_state_dict(policy_params)
        value_model = Value(othello.BOARD_SIZE**2)
        value_model.load_state_dict(value_params)

        num_opponents = len(cached_opponent_policies)
        wins = 0
        draws = 0
        wins_vector = torch.zeros(num_opponents, dtype=int)
        draws_vector = torch.zeros(num_opponents, dtype=int)
        plays_vector = torch.zeros(num_opponents, dtype=int)

        states, actions, masks, returns, advantages = [], [], [], [], []

        # Sample opponent indices based on the provided opponent_probs.
        if opponent_probs is None:
            # Fall back to uniform sampling if no probabilities provided.
            opponent_indices = []
            full_rounds = num_rollouts // num_opponents
            remainder = num_rollouts % num_opponents
            for _ in range(full_rounds):
                opponent_indices.extend(range(num_opponents))
            if remainder > 0:
                opponent_indices.extend(random.sample(range(num_opponents), remainder))
        else:
            # Sample with replacement using the probabilities.
            opponent_indices = np.random.choice(num_opponents, size=num_rollouts, replace=True, p=opponent_probs)

        for opponent_index in opponent_indices:
            rollout_states, rollout_actions, rollout_masks, rollout_rewards = [], [], [], []
            opponent_policy = cached_opponent_policies[opponent_index]
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
                action = inflate_action(flat_action.item())
                state, reward, done, _, info = env.step(action)
                rollout_rewards.append(reward)
            rollout_states = torch.concatenate(rollout_states, axis=0)
            rollout_actions = torch.stack(rollout_actions)
            rollout_masks = torch.stack(rollout_masks)
            rollout_rewards = torch.tensor(rollout_rewards).float()
            rollout_values = value_model(rollout_states).squeeze()
            rollout_returns = discount_cumsum(rollout_rewards, gamma)
            deltas = rollout_rewards + gamma * torch.cat([rollout_values[1:], torch.zeros(1)]) - rollout_values
            rollout_advantages = discount_cumsum(deltas, gamma * lam)

            if reward > 0:
                wins += 1
                wins_vector[opponent_index] += 1
            elif reward == 0:
                draws += 1
                draws_vector[opponent_index] += 1
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
                wins, draws,
                wins_vector, draws_vector, plays_vector)


def worker_process(worker_id, task_queue, result_queue, initial_cached_policies):
    cached_opponent_policies = [Policy(othello.BOARD_SIZE**2) for _ in initial_cached_policies]
    for policy, params in zip(cached_opponent_policies, initial_cached_policies):
        policy.load_state_dict(params)
    torch.set_num_threads(1)
    while True:
        task = task_queue.get()
        if task is None:
            break
        # Unpack the new opponent_probs from the task tuple.
        policy_state, value_state, gamma, lam, num_rollouts, new_opponent_policy, opponent_probs = task
        if new_opponent_policy:
            new_policy = Policy(othello.BOARD_SIZE**2)
            new_policy.load_state_dict(new_opponent_policy)
            cached_opponent_policies.append(new_policy)
        result = generate_experience(policy_state, value_state, cached_opponent_policies, gamma, lam, num_rollouts, opponent_probs)
        result_queue.put(result)

def compute_opponent_weights(smoothed_rates, threshold=0.84, steepness=30):
    # Base weight is 1 - win_rate.
    base_weights = 1 - smoothed_rates
    # Apply a logistic penalty: when smoothed_rate > threshold, the penalty factor quickly drops.
    penalty = 1 / (1 + torch.exp(steepness * (smoothed_rates - threshold)))
    weights = base_weights * penalty
    return weights

def ppo_clip_loss(policy_model, states: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
                  advantages: torch.Tensor, log_probs_old: torch.Tensor, clip_param: float, entropy_coeff: float):
    log_probs, entropy = policy_model.log_probs(states, actions, masks)
    mean_entropy = entropy.mean()
    ratio = torch.exp(log_probs - log_probs_old)
    clipped = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    loss = -(torch.min(ratio * advantages, clipped)).mean() - entropy_coeff * mean_entropy
    approximate_kl = (log_probs_old - log_probs).mean().item()
    clipped = ratio.gt(1+clip_param) | ratio.lt(1-clip_param)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    return loss, approximate_kl, mean_entropy, clipfrac

def main():
    checkpoint_path = "checkpoint.pth"
    initial_elo = 400
    start_iteration = 0

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)

    policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.0003)
    value_optimizer = optim.Adam(value_model.parameters(), lr=0.0005)

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        (start_iteration,
         policy_state,
         value_state,
         policy_opt_state,
         value_opt_state,
         elo_manager_state,
         best_elo,
         best_policy_state,
         best_value_state) = load_checkpoint(checkpoint_path)
        policy_model.load_state_dict(policy_state)
        value_model.load_state_dict(value_state)
        policy_model.to(device)
        value_model.to(device)
        policy_optimizer.load_state_dict(policy_opt_state)
        value_optimizer.load_state_dict(value_opt_state)
        elo_manager = EloManager(initial_elo=initial_elo)
        elo_manager.load_state_dict(elo_manager_state)
        print(f"Resuming from iteration {start_iteration}.")
    else:
        policy_model.to(device)
        value_model.to(device)
        elo_manager = EloManager(initial_elo=initial_elo)
        best_elo = initial_elo
        best_policy_state = get_cpu_state(policy_model)
        best_value_state = get_cpu_state(value_model)

    writer = SummaryWriter(log_dir="runs/ppo_entropy_reg")
    gamma = 0.99
    lam = 0.95
    entropy_coeff = .01
    clip_param = 0.2
    target_kl = 0.03
    num_policy_steps = 80
    num_value_steps = 80

    num_iterations = 300000000
    num_workers = 16
    rollouts_per_worker = 256 // num_workers
    total_rollouts = num_workers * rollouts_per_worker

    if len(elo_manager.pool) == 0:
        elo_manager.add_initial_policy("policy_0", policy_model, value_model)

    initial_cached_policies = [entry['policy_params'] for entry in elo_manager.pool]
    new_opponent_policy = None

    saturation_counter = 0
    saturation_threshold = 20
    win_threshold = 0.51
    smoothed_rates = None

    prev_pool_size = len(elo_manager.pool)
    accum_wins_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_draws_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_plays_vector = torch.zeros(prev_pool_size, dtype=int)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_process, args=(worker_id, task_queue, result_queue, initial_cached_policies))
        p.start()
        workers.append(p)

    checkpoint_interval = 200

    policy_cpu_state = get_cpu_state(policy_model)
    value_cpu_state = get_cpu_state(value_model)
    for iteration in range(start_iteration, num_iterations):
        if smoothed_rates is None:
            opponent_probs = None
        else:
            weights = compute_opponent_weights(smoothed_rates, threshold=0.8, steepness=30)
            weights = torch.clamp(weights, min=0)
            total = weights.sum().item()
            if total > 0:
                opponent_probs = (weights / total).tolist()
                opponent_probs = np.array(opponent_probs, dtype=np.float64)
                opponent_probs = (opponent_probs / opponent_probs.sum()).tolist()
            else:
                opponent_probs = [1.0 / len(weights)] * len(weights)

        for i in range(num_workers):
            task_queue.put((
                policy_cpu_state,
                value_cpu_state,
                gamma,
                lam,
                rollouts_per_worker,
                new_opponent_policy,
                opponent_probs,
            ))
        new_opponent_policy = None

        states_, actions_, masks_, returns_, advantages_ = [], [], [], [], []
        iteration_wins_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        iteration_draws_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        iteration_plays_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        wins_total = 0
        draws_total = 0
        results = [result_queue.get() for _ in range(num_workers)]
        for (states, actions, masks, returns, advantages,
             wins_, draws_,
             wins_vector_, draws_vector_, plays_vector_) in results:
            states_.append(states)
            actions_.append(actions)
            masks_.append(masks)
            returns_.append(returns)
            advantages_.append(advantages)
            wins_total += wins_
            draws_total += draws_
            iteration_wins_vector += wins_vector_
            iteration_draws_vector += draws_vector_
            iteration_plays_vector += plays_vector_

        states = torch.cat(states_, dim=0).to(device)
        actions = torch.cat(actions_, dim=0).to(device)
        masks = torch.cat(masks_, dim=0).to(device)
        returns = torch.cat(returns_, dim=0).to(device)
        advantages = torch.cat(advantages_, dim=0).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            log_probs_old, _ = policy_model.log_probs(states, actions, masks)

        avgclipfrac = 0

        policy_loss = 0
        policy_grad_norm = 0
        for i in range(num_policy_steps):
            policy_optimizer.zero_grad()
            new_policy_loss, kl, policy_entropy, clipfrac = ppo_clip_loss(policy_model, states, actions, masks, advantages, log_probs_old, clip_param, entropy_coeff)
            if kl > 1.5 * target_kl:
                break
            new_policy_loss.backward()
            policy_optimizer.step()
            avgclipfrac += clipfrac
            policy_loss += new_policy_loss.item()
            # policy_grad_norm += sum(p.grad.data.norm(2).item() ** 2 for p in policy_model.parameters() if p.grad is not None) ** 0.5
        policy_grad_norm /= num_policy_steps

        avgclipfrac /= i

        value_loss = 0
        value_grad_norm = 0
        for _ in range(num_value_steps):
            value_optimizer.zero_grad()
            new_value_loss = torch.mean((value_model(states).squeeze() - returns) ** 2)
            new_value_loss.backward()
            value_optimizer.step()
            value_loss += new_value_loss.item()
            # value_grad_norm += sum(p.grad.data.norm(2).item() ** 2 for p in value_model.parameters() if p.grad is not None) ** 0.5
        value_grad_norm /= num_value_steps

        with torch.no_grad():

            policy_cpu_state = get_cpu_state(policy_model)
            value_cpu_state = get_cpu_state(value_model)

            # policy_weight_norms_sqd = 0.0
            # for _,param in policy_cpu_state.items():
            #     policy_weight_norms_sqd += torch.sum(param ** 2)
            # policy_weight_norms = torch.sqrt(policy_weight_norms_sqd).item()

            # value_weight_norms_sqd = 0.0
            # for _,param in value_cpu_state.items():
            #     value_weight_norms_sqd += torch.sum(param ** 2)
            # value_weight_norms = torch.sqrt(value_weight_norms_sqd).item()

            win_percentage = wins_total / total_rollouts

            writer.add_scalar("Training/PolicyLoss", policy_loss, iteration)
            writer.add_scalar("Training/ValueLoss", value_loss, iteration)
            writer.add_scalar("Training/PolicyKL", kl, iteration)
            writer.add_scalar("Training/WinPercentage", win_percentage, iteration)
            # writer.add_scalar("Training/AverageReturn", returns.mean().item(), iteration)
            # writer.add_scalar("Training/PolicyGradNorm", policy_grad_norm, iteration)
            # writer.add_scalar("Training/ValueGradNorm", value_grad_norm, iteration)
            writer.add_scalar("Training/PolicyEntropy", policy_entropy, iteration)
            # writer.add_scalar("Training/PolicyParamNorm", policy_weight_norms, iteration)
            # writer.add_scalar("Training/ValueParamNorm", value_weight_norms, iteration)
            if len(elo_manager.pool) != prev_pool_size:
                accum_wins_vector = torch.zeros(len(elo_manager.pool), dtype=int)
                accum_draws_vector = torch.zeros(len(elo_manager.pool), dtype=int)
                accum_plays_vector = torch.zeros(len(elo_manager.pool), dtype=int)
                prev_pool_size = len(elo_manager.pool)
            accum_wins_vector += iteration_wins_vector
            accum_draws_vector += iteration_draws_vector
            accum_plays_vector += iteration_plays_vector

            elo_manager.update_ratings(iteration_wins_vector, iteration_draws_vector, iteration_plays_vector)
            writer.add_scalar("Training/PolicyElo", elo_manager.current_policy_elo, iteration)
            print(f"Iteration {iteration}: PLoss = {policy_loss:.3f}, VLoss = {value_loss:.3f}, Train win% = {win_percentage:.3f}, Pool size = {len(elo_manager.pool)}, Current Elo: {elo_manager.current_policy_elo:.1f}, Smoothed Elo: {elo_manager.smooth_current_policy_elo:.1f}, Steps: {i+1}, CF: {avgclipfrac:.2f}")

            if win_percentage >= win_threshold:
                saturation_counter += 1
            else:
                saturation_counter = 0

            alpha = .975
            new_rates = torch.zeros_like(iteration_wins_vector, dtype=torch.float)
            played_mask = iteration_plays_vector > 0
            new_rates[played_mask] = iteration_wins_vector[played_mask].float() / iteration_plays_vector[played_mask].float()
            if smoothed_rates is None:
                smoothed_rates = new_rates
            else:
                smoothed_rates[played_mask] = alpha * smoothed_rates[played_mask] + (1 - alpha) * new_rates[played_mask]
            smoothed_rates = torch.min(smoothed_rates, torch.full_like(smoothed_rates,.93))
            N = min(int(.9 * prev_pool_size + 1), prev_pool_size - 2)
            ready = (accum_plays_vector > 0).all() and (smoothed_rates[:N] > 0.8).all()

            with open("winrate.txt", "w") as f:
                for i in range(len(accum_wins_vector)):
                    f.write(f"{smoothed_rates[i]:.4f} {opponent_probs[i] if opponent_probs else 1/len(accum_wins_vector):.4f}\n")

            if saturation_counter >= saturation_threshold and ready:
                if elo_manager.smooth_current_policy_elo > best_elo:
                    best_elo = elo_manager.smooth_current_policy_elo
                    best_policy_state = policy_cpu_state
                    best_value_state = value_cpu_state

                smoothed_rates = torch.cat([smoothed_rates, torch.tensor([.5])], dim=0)
                elo_manager.add_new_policy(policy_cpu_state, value_cpu_state)
                new_opponent_policy = policy_cpu_state
                saturation_counter = 0

            if iteration % checkpoint_interval == 0 and iteration > start_iteration:
                save_checkpoint(
                    checkpoint_path,
                    iteration,
                    policy_cpu_state,
                    value_cpu_state,
                    policy_optimizer.state_dict(),
                    value_optimizer.state_dict(),
                    elo_manager.state_dict(),
                    best_elo,
                    best_policy_state,
                    best_value_state)

    writer.close()
    for _ in range(num_workers):
        task_queue.put(None)
    for p in workers:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
