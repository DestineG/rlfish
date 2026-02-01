import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from collections import deque
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_env(render_mode=None):
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True
    )
    env = FrameStack(env, 4)
    return env


# 直接使用采样数据会导致训练数据相关性过强，会导致梯度方向受局部影响，从而影响训练效果。
# 为了解决这个问题，引入了经验回放缓冲区（Replay Buffer），用于存储智能体与环境交互过程中产生的经验数据。
# 通过从缓冲区中随机采样小批量数据进行训练，可以打破数据之间的相关性，提高训练的稳定性和效果。
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_shape, dtype=np.float32):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_shape = state_shape

        # 预分配数组
        self.states = np.zeros((buffer_size, *state_shape), dtype=dtype)
        self.next_states = np.zeros((buffer_size, *state_shape), dtype=dtype)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=dtype)
        self.dones = np.zeros(buffer_size, dtype=np.uint8)

        self.pos = 0      # 当前写入位置
        self.size = 0     # 当前有效样本数
    
    def add(self, state, action, reward, next_state, done):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # 移动指针
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)
        batch_states = self.states[idx]
        batch_actions = self.actions[idx]
        batch_rewards = self.rewards[idx]
        batch_next_states = self.next_states[idx]
        batch_dones = self.dones[idx]
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size


class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.frame_stack = state_dim[0]
        self.height = state_dim[1]
        self.width = state_dim[2]
        self.conv1 = torch.nn.Conv2d(self.frame_stack, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(128 * (self.height // 4) * (self.width // 4), 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQNAgent:
    def __init__(self, state_dim, action_dim, device=torch.device("cpu")):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.05
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.bufffer_size = 100000
        self.batch_size = 64
        self.replay_buffer = ReplayBuffer(self.bufffer_size, self.batch_size, state_shape=state_dim)
        self.q_net = QNet(state_dim=state_dim, action_dim=action_dim)
        self.q_net.to(device)
        self.target_q_net = QNet(state_dim=state_dim, action_dim=action_dim)
        self.target_q_net.to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.synch_target()

    def synch_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def get_action(self, state):
        # FrameStack 返回 LazyFrames，显式转成 numpy 数组
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        # (frame_stack, height, width) -> (1, frame_stack, height, width)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.q_net(state)
        return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # 计算当前Q值
        states = torch.from_numpy(states).float().to(self.device)
        # (batch_size, state_dim) -> (batch_size, action_dim)
        q_s = self.q_net(states)
        # (batch_size,) -> (batch_size, 1)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        # (batch_size, action_dim), (batch_size, 1) -> (batch_size,)
        q_s_a = q_s.gather(1, actions_t).squeeze(1)
        
        # 计算目标Q值(贪婪策略)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        # 用当前Q网络选择下一个动作 (batch_size, state_dim) -> (batch_size, action_dim)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        # (batch_size, state_dim) -> (batch_size, action_dim)
        next_qs = self.target_q_net(next_states)
        # 用目标Q网络评估下一个动作 (batch_size, action_dim) -> (batch_size, 2) -> (batch_size,)
        next_q = next_qs.gather(1, next_actions).squeeze(1)
        target_q = torch.FloatTensor(rewards).to(self.device) + self.gamma * next_q * (1 - torch.FloatTensor(dones).to(self.device))

        # 优化Q网络
        loss = torch.nn.functional.mse_loss(q_s_a, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def trainQLearning(show_plot=True, device=torch.device("cpu")):
    env = make_env()
    # env.observation_space.shape = (4, 84, 84)
    agent = DQNAgent(state_dim=env.observation_space.shape, action_dim=env.action_space.n, device=device)
    episodes = 100
    sync_steps = 1
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
    env = make_env(render_mode=render_mode)
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
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    agent, _ = trainQLearning(show_plot=False, device=device)
    play_trained_agent(agent, device=device)