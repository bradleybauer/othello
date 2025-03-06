import os
import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
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
    def __init__(self, initial_elo=1200):
        self.current_policy_elo = initial_elo
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
            "policy_counter": self.policy_counter,
        }
    
    def load_state_dict(self, state):
        self.pool = state["pool"]
        self.current_policy_elo = state["current_policy_elo"]
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

def generate_experience(policy_params, value_params, cached_opponent_policies, gamma, lam, num_rollouts, seed):
    with torch.no_grad():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        policy_model = Policy(othello.BOARD_SIZE**2)
        policy_model.load_state_dict(policy_params)
        value_model = Value(othello.BOARD_SIZE**2)
        value_model.load_state_dict(value_params)

        num_opponents = len(cached_opponent_policies)
        wins = 0
        draws = 0
        losses = 0
        wins_vector = torch.zeros(num_opponents, dtype=int)
        draws_vector = torch.zeros(num_opponents, dtype=int)
        losses_vector = torch.zeros(num_opponents, dtype=int)
        plays_vector = torch.zeros(num_opponents, dtype=int)

        states, actions, masks, returns, advantages = [], [], [], [], []

        opponent_indices = []
        full_rounds = num_rollouts // num_opponents
        remainder = num_rollouts % num_opponents
        for _ in range(full_rounds):
            opponent_indices.extend(range(num_opponents))
        if remainder > 0:
            opponent_indices.extend(random.sample(range(num_opponents), remainder))

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
                action = env.inflate_action(flat_action.item())
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

def worker_process(worker_id, task_queue, result_queue, initial_cached_policies):
    cached_opponent_policies = [Policy(othello.BOARD_SIZE**2) for _ in initial_cached_policies]
    for policy, params in zip(cached_opponent_policies, initial_cached_policies):
        policy.load_state_dict(params)
    while True:
        task = task_queue.get()
        if task is None:
            break
        policy_state, value_state, gamma, lam, num_rollouts, seed, new_opponent_policy = task
        if new_opponent_policy:
            new_policy = Policy(othello.BOARD_SIZE**2)
            new_policy.load_state_dict(new_opponent_policy)
            cached_opponent_policies.append(new_policy)
        result = generate_experience(policy_state, value_state, cached_opponent_policies, gamma, lam, num_rollouts, seed)
        result_queue.put(result)

def ppo_clip_loss(policy_model, states: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
                  advantages: torch.Tensor, log_probs_old: torch.Tensor, clip_param: float):
    log_probs, entropy = policy_model.log_probs(states, actions, masks)
    ratio = torch.exp(log_probs - log_probs_old)
    clipped = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
    loss = -(torch.min(ratio * advantages, clipped)).mean()
    approximate_kl = (log_probs_old - log_probs).mean().item()
    return loss, approximate_kl, entropy

def main():
    writer = SummaryWriter(log_dir="runs/othello_experiment_ppo2")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)

    policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.0003)
    value_optimizer = optim.Adam(value_model.parameters(), lr=0.0005)

    checkpoint_path = "checkpoint.pth"
    initial_elo = 1200
    start_iteration = 0
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

    gamma = 0.99
    lam = 0.95
    clip_param = 0.2
    target_kl = 0.05
    num_policy_steps = 80
    num_value_steps = 80

    num_iterations = 300000000
    num_workers = 16
    rollouts_per_worker = 1024 // num_workers
    total_rollouts = num_workers * rollouts_per_worker

    if len(elo_manager.pool) == 0:
        elo_manager.add_initial_policy("policy_0", policy_model, value_model)

    initial_cached_policies = [entry['policy_params'] for entry in elo_manager.pool]
    new_opponent_policy = None

    saturation_counter = 0
    saturation_threshold = 20
    win_threshold = 0.51

    prev_pool_size = len(elo_manager.pool)
    accum_wins_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_draws_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_losses_vector = torch.zeros(prev_pool_size, dtype=int)
    accum_plays_vector = torch.zeros(prev_pool_size, dtype=int)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_process, args=(worker_id, task_queue, result_queue, initial_cached_policies))
        p.start()
        workers.append(p)

    checkpoint_interval = 20

    for iteration in range(start_iteration, num_iterations):
        for i in range(num_workers):
            task_queue.put((
                get_cpu_state(policy_model),
                get_cpu_state(value_model),
                gamma,
                lam,
                rollouts_per_worker,
                seed + i,
                new_opponent_policy,
            ))
        seed += num_workers
        new_opponent_policy = None

        states_, actions_, masks_, returns_, advantages_ = [], [], [], [], []
        iteration_wins_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        iteration_draws_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        iteration_losses_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        iteration_plays_vector = torch.zeros(len(elo_manager.pool), dtype=int)
        wins_total = 0
        draws_total = 0
        losses_total = 0
        results = [result_queue.get() for _ in range(num_workers)]
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

        win_percentage = wins_total / total_rollouts

        writer.add_scalar("Training/PolicyLoss", policy_loss, iteration)
        writer.add_scalar("Training/ValueLoss", value_loss, iteration)
        writer.add_scalar("Training/PolicyKL", kl, iteration)
        writer.add_scalar("Training/WinPercentage", win_percentage, iteration)
        writer.add_scalar("Training/AverageReturn", returns.mean().item(), iteration)
        writer.add_scalar("Training/PolicyGradNorm", policy_grad_norm, iteration)
        writer.add_scalar("Training/ValueGradNorm", value_grad_norm, iteration)
        writer.add_scalar("Training/PolicyEntropy", mean_policy_entropy, iteration)

        if len(elo_manager.pool) != prev_pool_size:
            accum_wins_vector = torch.zeros(len(elo_manager.pool), dtype=int)
            accum_draws_vector = torch.zeros(len(elo_manager.pool), dtype=int)
            accum_losses_vector = torch.zeros(len(elo_manager.pool), dtype=int)
            accum_plays_vector = torch.zeros(len(elo_manager.pool), dtype=int)
            prev_pool_size = len(elo_manager.pool)
        accum_wins_vector += iteration_wins_vector
        accum_draws_vector += iteration_draws_vector
        accum_losses_vector += iteration_losses_vector
        accum_plays_vector += iteration_plays_vector

        with open("win_rate_data.txt", "w") as f:
            f.write("Policy Index, Wins, Plays, Win Fraction\n")
            for i in range(len(accum_wins_vector)):
                plays = accum_plays_vector[i].item()
                wins = accum_wins_vector[i].item()
                win_fraction = wins / plays if plays > 0 else 0
                f.write(f"{i}, {wins}, {plays}, {win_fraction:.3f}\n")

        elo_manager.update_ratings(iteration_wins_vector, iteration_draws_vector, iteration_plays_vector)
        writer.add_scalar("Training/PolicyElo", elo_manager.current_policy_elo, iteration)
        print(f"Iteration {iteration}: PLoss = {policy_loss:.3f}, VLoss = {value_loss:.3f}, Train win% = {win_percentage:.3f}, Pool size = {len(elo_manager.pool)}, Current policy Elo: {elo_manager.current_policy_elo:.1f}")

        if elo_manager.current_policy_elo > best_elo:
            best_elo = elo_manager.current_policy_elo
            best_policy_state = get_cpu_state(policy_model)
            best_value_state = get_cpu_state(value_model)

        if win_percentage >= win_threshold:
            saturation_counter += 1
        else:
            saturation_counter = 0

        if saturation_counter >= saturation_threshold:
            policy_cpu_state = get_cpu_state(policy_model)
            value_cpu_state = get_cpu_state(value_model)
            elo_manager.add_new_policy(policy_cpu_state, value_cpu_state)
            new_opponent_policy = policy_cpu_state
            saturation_counter = 0

        if iteration % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                iteration,
                get_cpu_state(policy_model),
                get_cpu_state(value_model),
                policy_optimizer.state_dict(),
                value_optimizer.state_dict(),
                elo_manager.state_dict(),
                best_elo,
                best_policy_state,
                best_value_state
            )

    writer.close()
    for _ in range(num_workers):
        task_queue.put(None)
    for p in workers:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
