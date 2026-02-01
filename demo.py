import torch
import gymnasium as gym

class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.policy(state)  # 输出动作概率
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()      # 采样动作
        log_prob = dist.log_prob(action)  # log π(a|s)
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
        
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 累积损失
        for (r, log_prob), R in zip(self.memory, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = Agent(state_dim, action_dim, device=device)
    
    num_episodes = 10000
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.add(reward, log_prob)
            state = next_state
            total_reward += reward
        
        agent.update()
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
