import copy
import os
import time
import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
from policy_function import Policy
from value_function import Value
import torch.multiprocessing as mp

def generate_experience(policy_params, pool_policy_params, num_rollouts):
    """
    Worker function to generate experience.
    For each rollout, a random opponent is sampled from the historical pool.
    Returns a list of tuples: (states, actions, masks, rewards) for each trajectory.
    """
    with torch.no_grad():
        # Create a unique seed for each worker.
        seed = int(time.time() * 1000) % (2**32 - 1) + os.getpid()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Instantiate the current training policy.
        policy = Policy(othello.BOARD_SIZE**2)
        policy.load_state_dict(policy_params)

        states = []
        actions = []
        masks = []
        rewards = []
        wins = 0
        for _ in range(num_rollouts):
            # Randomly sample an opponent from the pool for this trajectory.
            sampled_opponent_params = random.choice(pool_policy_params)
            opponent_policy = Policy(othello.BOARD_SIZE**2)
            opponent_policy.load_state_dict(sampled_opponent_params)
            env = OthelloEnv(opponent=opponent_policy)

            rollout_states = []
            rollout_actions = []
            rollout_masks = []
            rollout_rewards = []

            state, info = env.reset()
            done = False
            while not done:
                state = torch.from_numpy(state).reshape(1, -1).float()
                mask = torch.from_numpy(info['action_mask'])

                rollout_states.append(state)
                rollout_masks.append(mask)

                flat_action = policy.select_action(state, mask)

                rollout_actions.append(flat_action)
                action = env.inflate_action(flat_action.item())
                state, reward, done, _, info = env.step(action)

            rollout_states = torch.concatenate(rollout_states, axis=0)
            rollout_actions = torch.stack(rollout_actions)
            rollout_masks = torch.stack(rollout_masks)
            rollout_rewards = torch.ones_like(rollout_actions) * reward
            wins += 1 if reward > 0 else 0

            states.append(rollout_states)
            actions.append(rollout_actions)
            masks.append(rollout_masks)
            rewards.append(rollout_rewards)

        states = torch.vstack(states)
        actions = torch.vstack(actions)
        masks = torch.vstack(masks)
        rewards = torch.vstack(rewards)
        return (states, actions, masks, rewards, wins)

def main():
    seed = 42  # You can also use a dynamic seed if desired.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    value_model = Value(othello.BOARD_SIZE**2)
    value_model.to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_model.parameters(), lr=1e-3)

    num_iterations = 1000
    num_workers = 16
    rollouts_per_worker = 1024 // num_workers
    total_rollouts = num_workers * rollouts_per_worker
    best_num_wins = -1

    # Set the maximum size of the historical pool.
    pool_size = 30
    # Initialize the pool with the starting policy.
    policy_params_pool = [copy.deepcopy(policy_model.state_dict())]*3
    
    # Saturation parameters: if win percentage is above threshold for these many iterations,
    # add the current policy to the pool.
    saturation_counter = 0
    saturation_threshold = 10
    win_threshold = 0.6

    # Create a multiprocessing pool using torch.multiprocessing.
    pool = mp.Pool(processes=num_workers)
    for iteration in range(num_iterations):
        args = [
            (copy.deepcopy(policy_model.state_dict()), policy_params_pool, rollouts_per_worker)
            for _ in range(num_workers)
        ]
        wins = 0
        policy_loss = 0
        value_loss = 0
        results = pool.starmap(generate_experience, args)
        policy_model.to(device)
        for states, actions, masks, rewards, wins_ in results:
            states = states.to(device)
            rewards = rewards.to(device)

            log_probs = policy_model.log_probs(states, actions.to(device), masks.to(device))
            value = value_model(states)
            with torch.no_grad():
                phi = rewards - value
            policy_loss += -torch.mean(log_probs * phi)
            value_loss += torch.mean((value - rewards)**2)
            wins += wins_

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        policy_model.cpu()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        win_percentage = wins / total_rollouts
        print(f"Iteration {iteration}: PLoss = {policy_loss.item():.3f}, VLoss = {value_loss.item():.3f}, Train win% = {win_percentage:.3f}, Policy Pool size = {len(policy_params_pool)}.")

        if wins > best_num_wins:
            best_num_wins = wins
            torch.save(policy_model.state_dict(), "best_policy_model.pth")
            print(f"New best model with wins% = {win_percentage:.3f}.")
            with torch.no_grad():
                dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
                torch.onnx.export(
                    policy_model,
                    (dummy_state,),
                    "best_policy_model.onnx",
                    input_names=["state"],
                    output_names=["logits"],
                    opset_version=11
                )

        # Check if win percentage has been high for consecutive iterations.
        if win_percentage >= win_threshold:
            saturation_counter += 1
        else:
            saturation_counter = 0

        # Add current policy to the pool when performance saturates.
        if saturation_counter >= saturation_threshold:
            if len(policy_params_pool) >= pool_size:
                policy_params_pool.pop(random.randint(0, len(policy_params_pool) - 1))
            policy_params_pool.append(copy.deepcopy(policy_model.state_dict()))
            saturation_counter = 0
            print("Current policy added to historical pool.")
            torch.save(policy_model.state_dict(), "latest_pool_addition.pth")
            with torch.no_grad():
                dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
                torch.onnx.export(
                    policy_model,
                    (dummy_state,),
                    "latest_pool_addition.onnx",
                    input_names=["state"],
                    output_names=["logits"],
                    opset_version=11
                )

    pool.close()
    pool.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
