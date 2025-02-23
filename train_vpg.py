import time
import torch
import torch.optim as optim
import numpy as np
import random
from policy import Policy
from othello_env import OthelloEnv
import othello

# A helper function to compute rewards to go
def rewards_to_go(rewards):
    togo = []
    R = 0
    for r in reversed(rewards):
        R = r + R
        togo.insert(0, R)
    return togo

def test_policy(model, env, num_episodes=200):
    """
    Runs the policy in a deterministic manner.
    Instead of sampling, it uses the model's forward pass to get action probabilities
    and selects the action with the highest probability.
    """
    model.eval()
    total_reward = 0.0
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.from_numpy(state).float()
            action_mask = torch.from_numpy(info['action_mask'])
            with torch.no_grad():
                action, _ = model.select_action(state_tensor, action_mask)
            state, reward, done, _, info = env.step(action.numpy())
            episode_reward = 1 if reward else episode_reward

            # Play against a random policy if the game is not done
            if not done:
                action_mask = torch.from_numpy(info['action_mask'])
                action = env.sample_random_action(action_mask)
                state, reward, done, _, info = env.step(action)

                # Adjust reward since opponent’s reward affects our outcome
                episode_reward = 0 if reward else episode_reward

        total_reward += episode_reward
    model.train()
    return total_reward / num_episodes

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = OthelloEnv()
    policy_model = Policy(othello.BOARD_SIZE)
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

    num_iterations = 1000
    num_episodes_per_iter = 100
    best_test_reward = float('-inf')
    best_model_params = None

    for iteration in range(num_iterations):
        batch_log_probs = []
        batch_rewards = []

        t_policy = 0
        t_env = 0
        wins = 0

        # Run a batch of episodes
        for episode in range(num_episodes_per_iter):
            state, info = env.reset()
            done = False
            episode_log_probs = []
            episode_rewards = []

            while not done:
                start = time.time()
                state_tensor = torch.from_numpy(state).float()
                action_mask = torch.from_numpy(info['action_mask'])
                action, log_prob = policy_model.select_action(state_tensor, action_mask)
                t_policy += time.time() - start

                start = time.time()
                state, reward, done, _, info = env.step(action.numpy())
                t_env += time.time() - start

                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)

                # Play against a random policy if the game is not done
                if not done:
                    start = time.time()
                    op_action_mask = torch.from_numpy(info['action_mask'])
                    op_action = env.sample_random_action(op_action_mask)
                    state, reward, done, _, info = env.step(op_action)
                    t_env += time.time() - start

                    # Adjust reward since opponent’s reward affects our outcome
                    episode_rewards[-1] -= reward

            batch_log_probs.append(torch.stack(episode_log_probs).squeeze(-1))
            batch_rewards.append(torch.tensor(rewards_to_go(episode_rewards), dtype=torch.float))
            wins += episode_rewards[-1] == 1

        # Compute policy loss: negative log probability weighted by rewards to go
        loss = 0
        for log_probs, rewards in zip(batch_log_probs, batch_rewards):
            loss += -torch.mean(log_probs * rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Determine performance using the deterministic test run.
        avg_test_reward = test_policy(policy_model, env, num_episodes=100)
        print(f"Iteration {iteration}: Loss = {loss.item():.3f}, Test win% = {avg_test_reward:.3f}, Train win% = {wins/num_episodes_per_iter:.3f}, Time: Policy = {t_policy:.3f}, Env = {t_env:.3f}")

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
        dummy_state = torch.randn(othello.BOARD_SIZE).float()
        dummy_mask = torch.ones(othello.BOARD_SIZE).float()
        torch.onnx.export(policy_model,
                          (dummy_state, dummy_mask),
                          "policy_model.onnx",
                          input_names=["state", "action_mask"],
                          output_names=["action", "log_prob"],
                          opset_version=11)
        print("Model exported to policy_model.onnx")

if __name__ == '__main__':
    main()
