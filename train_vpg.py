import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
from policy import Policy
from random_policy import RandomPolicy
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_rollouts(policy_params, num_rollouts):
    """
    Worker function to generate multiple rollouts.
    Each worker creates its own environment and policy (using the passed-in parameters)
    and then runs several episodes (rollouts) under torch.no_grad().
    Returns a list of tuples: (states, actions, masks, final_reward) for each trajectory.
    """
    env = OthelloEnv(opponent=RandomPolicy())
    policy = Policy(othello.BOARD_SIZE**2)
    policy.load_state_dict(policy_params)

    rollouts = []
    for _ in range(num_rollouts):
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
                flat_action = policy.select_action(state_tensor.reshape(1,-1), action_mask)

            actions.append(flat_action.item())
            action = env.inflate_action(flat_action.item())
            state, reward, done, _, info = env.step(action)

        states = np.stack(states).reshape(len(states), -1)
        actions = np.array(actions)
        masks = np.stack(masks)

        rollouts.append((states, actions, masks, reward))
    return rollouts

def main():
    seed = 42
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
    best_model_params = None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for iteration in range(num_iterations):
            futures = [executor.submit(generate_rollouts, policy_model.state_dict(), rollouts_per_worker) for _ in range(num_workers)]
            policy_model.to(device)
            wins = 0
            loss = 0
            for future in as_completed(futures):
                rollouts = future.result()
                for states, actions, masks, final_reward in rollouts:
                    states_tensor = torch.from_numpy(states).float()
                    actions_tensor = torch.tensor(actions)
                    masks_tensor = torch.from_numpy(masks)
                    log_probs = policy_model.log_probs(states_tensor.to(device), actions_tensor.to(device), masks_tensor.to(device))
                    loss += -torch.mean(log_probs * final_reward)
                    wins += final_reward > 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_model.cpu()

            print(f"Iteration {iteration}: Loss = {loss.item():.3f}, Train win% = {wins/total_rollouts:.2f}")

            if wins > best_num_wins:
                best_num_wins = wins
                best_model_params = policy_model.state_dict()
                torch.save(best_model_params, "best_policy_model.pth")
                print(f"New best model with wins% = {best_num_wins/total_rollouts:.2f}.")

                # Export the best model as an ONNX file.
                if best_model_params is not None:
                    policy_model.load_state_dict(best_model_params)
                    dummy_state = torch.randn(othello.BOARD_SIZE**2).float()
                    torch.onnx.export(policy_model,
                                    (dummy_state,),
                                    "policy_model.onnx",
                                    input_names=["state"],
                                    output_names=["logits"],
                                    opset_version=11)
                    print("Model exported to policy_model.onnx")

if __name__ == '__main__':
    main()
