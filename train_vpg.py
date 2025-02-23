import time
import torch
import torch.optim as optim
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

env = OthelloEnv()
obs, info = env.reset()
done = False

policy_model = Policy(othello.BOARD_SIZE)

optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

num_iterations = 1000
num_episodes_per_iter = 100

for iteration in range(num_iterations):
    batch_log_probs = []
    batch_rewards = []
    wins = []

    t_policy = 0
    t_env = 0

    # Run a batch of episodes
    for episode in range(num_episodes_per_iter):
        state, info = env.reset()
        done = False
        episode_log_probs = []
        episode_rewards = []

        while not done:
            start = time.time()
            state = torch.from_numpy(state).float()
            action_mask = torch.from_numpy(info['action_mask'])
            action, log_prob = policy_model.select_action(state, action_mask)
            t_policy += time.time() - start

            start = time.time()
            state, reward, done, _, info = env.step(action.numpy())
            t_env += time.time() - start

            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            # play against random policy
            if not done:
                start = time.time()
                op_action_mask = torch.from_numpy(info['action_mask'])
                op_action = env.sample_random_action(op_action_mask)
                state, reward, done, _, info = env.step(op_action)
                t_env += time.time() - start

                episode_rewards[-1] -= reward

            assert(episode_rewards[-1] == 0 or done)

        wins.append(episode_rewards[-1] == 1)
        batch_log_probs.append(torch.stack(episode_log_probs).squeeze(-1))
        batch_rewards.append(torch.tensor(rewards_to_go(episode_rewards), dtype=torch.float))


    # Compute policy loss.
    loss = 0
    for log_probs, rewards in zip(batch_log_probs, batch_rewards):
        # The policy gradient loss: -log_prob * reward.
        loss += -torch.mean(log_probs * rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iteration {iteration}: Loss = {loss.item():.3f}, Wins = {sum(wins)} / {num_episodes_per_iter}, Time: Policy = {t_policy:.3f}, Env = {t_env:.3f}")
