import torch
import gymnasium as gym


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        self.value_function = ValueNet(state_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=1e-3)
        self.gamma = 0.99

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, action_prob, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(0).to(self.device)

        # valueNet loss 计算
        target = (reward_tensor + (1 - done_tensor) * self.gamma * self.value_function(next_state_tensor)).detach()
        value = self.value_function(state_tensor)
        value_loss = torch.nn.functional.mse_loss(value, target)

        # policyNet loss 计算(value 作为 baseline)
        advantage = (target - value).detach()
        policy_loss = -action_prob * advantage
        
        # valueNet 参数更新
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # policyNet 参数更新
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
def train(episodes=500, device="cpu"):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCriticAgent(state_dim, action_dim, device)
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward

        if (ep + 1) % 5 == 0:
            print(f"Episode {ep + 1}, Total Reward: {total_reward}")
    return agent


def play_trained_agent(agent, episodes=3, render_mode="human", device="cpu"):
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
    trained_agent = train(episodes=1000, device=device)
    play_trained_agent(trained_agent, episodes=3, render_mode="human", device=device)