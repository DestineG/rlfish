import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# 直接使用采样数据会导致训练数据相关性过强，会导致梯度方向受局部影响，从而影响训练效果。
# 为了解决这个问题，引入了经验回放缓冲区（Replay Buffer），用于存储智能体与环境交互过程中产生的经验数据。
# 通过从缓冲区中随机采样小批量数据进行训练，可以打破数据之间的相关性，提高训练的稳定性和效果。
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
    
    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones


class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.05
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bufffer_size = 100000
        self.batch_size = 64
        self.replay_buffer = ReplayBuffer(self.bufffer_size, self.batch_size)
        self.q_net = QNet(state_dim=state_dim, action_dim=action_dim)
        self.target_q_net = QNet(state_dim=state_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.synch_target()

    def synch_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # 计算当前Q值 state = [小车位置, 小车速度, 杆子角度, 杆子角速度]
        states = torch.FloatTensor(states)
        # (batch_size, state_dim) -> (batch_size, action_dim)
        q_s = self.q_net(states)
        # (batch_size,) -> (batch_size, 1) -> (batch_size,)
        q_s_a = q_s.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)

        # 计算目标Q值(贪婪策略)
        next_states = torch.FloatTensor(next_states)
        # (batch_size, state_dim) -> (batch_size, action_dim)
        next_qs = self.target_q_net(next_states)
        # (batch_size, action_dim) -> (batch_size, 2) -> (batch_size,)
        next_q = next_qs.max(1)[0]
        rewards -= (0.01 * abs(state[0]) + 0.01 * abs(state[2]))
        target_q = torch.FloatTensor(rewards) + self.gamma * next_q * (1 - torch.FloatTensor(dones))

        # 优化Q网络
        loss = torch.nn.functional.mse_loss(q_s_a, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def trainQLearning(show_plot=True):
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    episodes = 2000
    sync_steps = 10
    rewards_history = []
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        if episode % sync_steps == 0:
            agent.synch_target()
        rewards_history.append(total_reward)

    if not show_plot:
        return agent, rewards_history

    # 平滑一下曲线（滑动平均）
    window = 10
    if len(rewards_history) >= window:
        smoothed_rewards = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    else:
        smoothed_rewards = rewards_history

    # 绘制奖励曲线
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_history, color='orange', alpha=0.4, label='Raw Reward')
    plt.plot(range(window-1, len(smoothed_rewards)+window-1), smoothed_rewards, color='red', label='Smoothed Reward (window=10)')
    plt.title('DQN on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def play_trained_agent(agent, episodes=3, render_mode="human"):
    """
    使用训练好的 agent 在环境中可视化游戏。
    参数：
        agent: 训练好的 DQNAgent
        episodes: 要玩的回合数
        render_mode: "human" 表示窗口渲染；"rgb_array" 可用于录制
    """
    env = gym.make('CartPole-v1', render_mode=render_mode)
    agent.epsilon = 0  # 关闭探索，纯利用策略

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"\n🎮 Episode {ep + 1} start")

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1

        print(f"✅ Episode {ep + 1} finished | Steps: {step}, Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    agent, _ = trainQLearning(show_plot=False)
    play_trained_agent(agent)