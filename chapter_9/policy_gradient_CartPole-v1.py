import torch
import gymnasium as gym

class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Softmax(dim=-1)   # 输出动作概率分布
        )
    
    def forward(self, x):
        return self.fc(x)


class Agent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.lr = 0.0005
        self.gamma = 0.95
        self.memory = []
        self.device = device

        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state):
        # (state_dim,) -> (1, state_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # (1, state_dim) -> (1, action_dim)
        probs = self.policy(state)  # 输出动作概率
        # 创建离散分布
        dist = torch.distributions.Categorical(probs)
        # 从分布中采样动作
        action = dist.sample()
        # 取出action的概率 然后对此概率取对数
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def add(self, reward, log_prob):
        self.memory.append((reward, log_prob))
    
    def update(self):
        R = 0
        returns = []
        policy_loss = []

        # 计算每一步的回报（折扣累计奖励）
        for r, _ in reversed(self.memory):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 累积损失
        for (r, log_prob), R in zip(self.memory, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.memory = []


def train(episodes, device="cpu"):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = Agent(state_dim, action_dim, device=device)
    
    num_episodes = episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward -= (0.01 * abs(next_state[0]) + 0.01 * abs(next_state[2]))
            agent.add(reward, log_prob)
            state = next_state
            total_reward += reward
        
        agent.update()
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
    env.close()
    return agent


def play_trained_agent(agent, episodes=3, render_mode="human", device="cpu"):
    """
    使用训练好的策略梯度 agent 进行测试和可视化。
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    agent.policy.eval()  # 切换到评估模式（关闭 dropout 等）

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"\n🎮 Episode {ep + 1} start")

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs, dim=1).item()  # 选择概率最高的动作

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1

        print(f"✅ Episode {ep + 1} finished | Steps: {step}, Total reward: {total_reward}")

    env.close()
    agent.policy.train()  # 恢复训练模式


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    agent = train(episodes=2000, device=device)
    play_trained_agent(agent, episodes=3, render_mode="human", device=device)