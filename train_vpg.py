import torch
import torch.optim as optim
import numpy as np
import random
import othello
from othello_env import OthelloEnv
from policy import Policy
from random_policy import RandomPolicy

def test_policy(model, env, num_episodes=200):
    model.eval()
    total_reward = 0.0
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float()
            action_mask = torch.from_numpy(info['action_mask'])
            with torch.no_grad():
                flat_action = model.select_action(state_tensor, action_mask)
                action = env.inflate_action(flat_action.item())
            state, reward, done, _, info = env.step(action)

        win = reward > 0
        total_reward += win
    model.train()
    return total_reward / num_episodes

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = OthelloEnv(opponent=RandomPolicy(othello.BOARD_SIZE))
    policy_model = Policy(othello.BOARD_SIZE)
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

    num_iterations = 1000
    num_workers = 16
    num_episodes_per_iter = num_workers * 8
    best_test_reward = float('-inf')
    best_model_params = None

    for iteration in range(num_iterations):
        batch_log_probs = []
        batch_rewards = []
        wins = 0

        for _ in range(num_episodes_per_iter):
            state, info = env.reset()
            done = False
            episode_log_probs = []

            while not done:
                state_tensor = torch.from_numpy(state).float()
                action_mask = torch.from_numpy(info['action_mask'])
                with torch.no_grad():
                    flat_action = policy_model.select_action(state_tensor, action_mask)
                episode_log_probs.append(policy_model.log_probs(state_tensor, flat_action, action_mask))
                state, reward, done, _, info = env.step(env.inflate_action(flat_action.item()))

            batch_log_probs.append(torch.stack(episode_log_probs).squeeze(-1))
            batch_rewards.append(reward)
            wins += reward > 0

        # Compute policy loss: negative log probability weighted by rewards to go
        loss = 0
        for log_probs, rewards in zip(batch_log_probs, batch_rewards):
            loss += -torch.mean(log_probs * rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Determine performance using the deterministic test run.
        avg_test_reward = test_policy(policy_model, env, num_episodes=100)
        print(f"Iteration {iteration}: Loss = {loss.item():.3f}, Test win% = {avg_test_reward:.3f}, Train win% = {wins/num_episodes_per_iter:.3f}")

        # Save best model parameters based on deterministic test reward.
        if avg_test_reward > best_test_reward:
            best_test_reward = avg_test_reward
            best_model_params = policy_model.state_dict()
            torch.save(best_model_params, "best_policy_model.pth")
            print(f"New best model saved with average test reward = {best_test_reward:.3f}.")

    # Export the best model as an ONNX file
    if best_model_params is not None:
        policy_model.load_state_dict(best_model_params)
        # Create dummy inputs for ONNX export (adjust shape/type as needed)
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
