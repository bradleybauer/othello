import copy
import os
import time
import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
from policy import Policy
import torch.multiprocessing as mp

def generate_rollouts(policy_params, pool_policy_params, num_rollouts):
    """
    Worker function to generate multiple rollouts.
    For each rollout, a random opponent is sampled from the historical pool.
    Returns a list of tuples: (states, actions, masks, final_reward) for each trajectory.
    """
    # Create a unique seed for each worker.
    seed = int(time.time() * 1000) % (2**32 - 1) + os.getpid()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Instantiate the current training policy.
    policy = Policy(othello.BOARD_SIZE**2)
    policy.load_state_dict(policy_params)

    rollouts = []
    for _ in range(num_rollouts):
        # Randomly sample an opponent from the pool for this trajectory.
        sampled_opponent_params = random.choice(pool_policy_params)
        opponent_policy = Policy(othello.BOARD_SIZE**2)
        opponent_policy.load_state_dict(sampled_opponent_params)
        env = OthelloEnv(opponent=opponent_policy)

        states = []
        actions = []
        masks = []

        state, info = env.reset()
        done = False
        while not done:
            states.append(state)
            masks.append(info['action_mask'])
            
            state_tensor = torch.from_numpy(state).float()
            action_mask = torch.from_numpy(info['action_mask'])
            with torch.no_grad():
                flat_action = policy.select_action(state_tensor.reshape(1, -1), action_mask)

            actions.append(flat_action.item())
            action = env.inflate_action(flat_action.item())
            state, reward, done, _, info = env.step(action)

        states = np.stack(states).reshape(len(states), -1)
        actions = np.array(actions)
        masks = np.stack(masks)

        rollouts.append((states, actions, masks, reward))
    return rollouts

def main():
    seed = 42  # You can also use a dynamic seed if desired.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    policy_model = Policy(othello.BOARD_SIZE**2)
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

    num_iterations = 1000
    num_workers = 16
    rollouts_per_worker = 1024 // num_workers
    total_rollouts = num_workers * rollouts_per_worker
    best_num_wins = -1

    # Set the maximum size of the historical pool.
    pool_size = 10
    # Initialize the pool with the starting policy.
    policy_params_pool = [copy.deepcopy(policy_model.state_dict())]
    
    # Saturation parameters: if win percentage is above threshold for these many iterations,
    # add the current policy to the pool.
    saturation_counter = 0
    saturation_threshold = 20  # e.g., 10 consecutive iterations.
    win_threshold = 0.8

    # Create a multiprocessing pool using torch.multiprocessing.
    pool = mp.Pool(processes=num_workers)
    for iteration in range(num_iterations):
        args = [
            (copy.deepcopy(policy_model.state_dict()), policy_params_pool, rollouts_per_worker)
            for _ in range(num_workers)
        ]
        wins = 0
        loss = 0
        results = pool.starmap(generate_rollouts, args)
        policy_model.to(device)
        for rollout_list in results:
            for states, actions, masks, final_reward in rollout_list:
                states_tensor = torch.from_numpy(states).float().to(device)
                actions_tensor = torch.tensor(actions).to(device)
                masks_tensor = torch.from_numpy(masks).to(device)
                log_probs = policy_model.log_probs(states_tensor, actions_tensor, masks_tensor)
                loss += -torch.mean(log_probs * final_reward)
                wins += 1 if final_reward > 0 else 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy_model.cpu()

        win_percentage = wins / total_rollouts
        print(f"Iteration {iteration}: Loss = {loss.item():.3f}, Train win% = {win_percentage:.2f}")

        if wins > best_num_wins:
            best_num_wins = wins
            torch.save(policy_model.state_dict(), "best_policy_model.pth")
            print(f"New best model with wins% = {win_percentage:.2f}.")
            dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
            with torch.no_grad():
                torch.onnx.export(
                    policy_model,
                    (dummy_state,),
                    "policy_model.onnx",
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
                # Remove the oldest policy from the pool.
                policy_params_pool.pop(0)
            policy_params_pool.append(copy.deepcopy(policy_model.state_dict()))
            saturation_counter = 0
            print("Current policy added to historical pool.")

    pool.close()
    pool.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
