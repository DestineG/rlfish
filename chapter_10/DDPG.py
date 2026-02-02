import torch
import gymnasium as gym
from collections import deque
import random
import numpy as np
import copy


def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    return env


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        # store raw numpy arrays (or lists) for easier batching
        self.buffer.append((np.array(s, dtype=np.float32),
                            np.array(a, dtype=np.float32),
                            float(r),
                            np.array(s2, dtype=np.float32),
                            float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),  # (B,1)
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1)   # (B,1)
        )

    def __len__(self):
        return len(self.buffer)


class policyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(policyNet, self).__init__()
        self.max_action = max_action
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.fc(state)


class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, device='cpu'):
        self.gamma = 0.98
        self.device = torch.device(device)
        self.replay_buffer = ReplayBuffer(capacity=200000)
        self.max_action = max_action

        self.policy_net = policyNet(state_dim, action_dim, max_action).to(self.device)
        self.q_net = QNet(state_dim, action_dim).to(self.device)
        self.target_policy_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        # optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)

        # 目标网络初始化
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, noise_scale=0.0):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_net(state_t).cpu().numpy()[0]
        if noise_scale > 0.0:
            action = action + noise_scale * np.random.randn(*action.shape)
        # 确保动作在合法范围内
        return np.clip(action, -self.max_action, self.max_action)

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done, tau=0.005):
        state = batch_state.to(self.device)         # (B, state_dim)
        action = batch_action.to(self.device)       # (B, action_dim)
        reward = batch_reward.to(self.device)       # (B,1)
        next_state = batch_next_state.to(self.device)
        done = batch_done.to(self.device)           # (B,1)

        # 目标Q值计算
        with torch.no_grad():
            next_action = self.target_policy_net(next_state)          # (B, action_dim)
            target_q = self.target_q_net(next_state, next_action)     # (B,1)
            target_q_value = reward + (1.0 - done) * self.gamma * target_q  # (B,1)

        # Q网络更新
        current_q = self.q_net(state, action)  # (B,1)
        q_loss = torch.nn.functional.mse_loss(current_q, target_q_value)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # policy网络更新
        policy_action = self.policy_net(state)  # (B, action_dim)
        actor_loss = -self.q_net(state, policy_action).mean()
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # 4) 软更新目标网络
        self.soft_update(tau)


def train(episodes, device='cpu'):
    env_name = "Pendulum-v1"
    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dim, action_dim, max_action=max_action, device=device)

    batch_size = 64
    replay_warmup = 1000
    noise_scale = 0.3
    min_noise = 0.05
    noise_decay = (noise_scale - min_noise) / (episodes * 0.5)

    recent_rewards = deque(maxlen=10)
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, noise_scale=noise_scale)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer) >= replay_warmup:
                b_s, b_a, b_r, b_s2, b_d = agent.replay_buffer.sample(batch_size)
                agent.update(b_s, b_a, b_r, b_s2, b_d)

        noise_scale = max(min_noise, noise_scale - noise_decay)

        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, noise_scale: {noise_scale:.2f}")

        recent_rewards.append(episode_reward)
        if len(recent_rewards) == recent_rewards.maxlen:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            if avg_reward >= -100.0:
                print(f"Solved in episode {episode} with average reward {avg_reward:.2f}")
                break

    env.close()
    return agent


def test_pretrained_agent(agent, episodes=3, device='cpu'):
    env_name = "Pendulum-v1"
    env = make_env(env_name, render_mode="human")
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state, noise_scale=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state = next_state
            episode_reward += reward
        print(f"Test Episode {episode}, Reward: {episode_reward:.2f}")
    env.close()


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    agent = train(episodes=1000, device=device)
    test_pretrained_agent(agent, episodes=3, device=device)
