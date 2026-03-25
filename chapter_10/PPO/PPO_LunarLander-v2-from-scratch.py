import torch
import torch.nn as nn
import torch.functional as F
import gymnasium as gym
from collections import deque
import numpy as np


class VNet(nn.Module):
    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class PPOAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.gamma = 0.99  # 折扣因子
        self.kl_constraint = 0.01  # KL 散度约束（当前更新逻辑主要用 PPO Clip）
        self.ls_alpha = 0.5  # 线性搜索衰减系数（TRPO 代码被注释）
        self.clip_epsilon = 0.1  # PPO 的剪切范围

        self.vnet = VNet(state_dim).to(self.device)
        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        self.vnet_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.memory = []

    def select_action(self, state):
        # (state_dim,) -> (1, state_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(state)

        # 离散动作概率分布
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # 返回动作的索引和对应的 log 概率（log_prob 在当前实现里未使用到）
        return action.item(), dist.log_prob(action).item()

    def add(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def update(self):
        # tuple[numpy.ndarray], tuple[int], tuple[float], tuple[bool]
        states, actions, rewards, dones = zip(*self.memory)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)

        if len(states_tensor.shape) == 1:
            states_tensor = states_tensor.unsqueeze(1)
        if len(actions_tensor.shape) == 1:
            actions_tensor = actions_tensor.unsqueeze(1)

        # 计算每个时刻的回报 Q(s_t) = sum_{k>=t} gamma^{k-t} r_k
        Qs = []
        next_state_value = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # 此处不使用 done 重置 next_state_value：假设 memory 始终是一条完整轨迹
            next_state_value = reward + self.gamma * next_state_value
            Qs.insert(0, next_state_value)

        Qs_tensor = torch.tensor(Qs, dtype=torch.float32, device=self.device)
        if len(Qs_tensor.shape) == 1:
            Qs_tensor = Qs_tensor.unsqueeze(1)

        # 更新值函数网络
        state_values = self.vnet(states_tensor)
        v_loss = torch.nn.functional.mse_loss(state_values, Qs_tensor)
        self.vnet_optimizer.zero_grad()
        v_loss.backward()
        self.vnet_optimizer.step()

        # 计算优势函数 A = Q - V
        advantages = Qs_tensor - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = states_tensor.detach()
        actions_tensor = actions_tensor.detach()

        # 计算 old_log_probs
        with torch.no_grad():
            probs = self.policy(states_tensor)
            log_probs = torch.log(probs.gather(1, actions_tensor))
            old_log_probs = log_probs.detach()

        for _ in range(4):  # PPO 多次更新
            current_probs = self.policy(states_tensor)
            log_probs = torch.log(current_probs.gather(1, actions_tensor))

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            surrogate_loss = -torch.mean(torch.min(surr1, surr2))

            self.policy.zero_grad()
            surrogate_loss.backward()
            self.policy_optimizer.step()

        self.memory = []

    def hvp(self, vector, states):
        """计算 Hessian-Vector Product H * vector，其中 H 是 KL 散度的 Hessian"""
        # KL(θ_old || θ)
        self.policy.zero_grad()
        probs = self.policy(states)
        old_probs = probs.detach()
        kl = torch.sum(old_probs * torch.log(old_probs / (probs + 1e-8)), dim=1).mean()

        # ∇_θ KL(θ_old || θ)
        kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])

        # ∇_θ KL(θ_old || θ) · vector
        fk_grad_vector = (flat_kl_grad * vector).sum()

        # ∇_θ (∇_θ KL(θ_old || θ) · vector)
        hvp = torch.autograd.grad(fk_grad_vector, self.policy.parameters())
        flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp]).detach()

        # 这里沿用你原脚本的数值处理
        return flat_hvp + 0.1 * vector

    def cg(self, grad, states, nsteps=10):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            hvp = self.hvp(p, states)
            alpha = rdotr / (torch.dot(p, hvp) + 1e-8)
            x += alpha * p
            r -= alpha * hvp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def get_step_size(self, step_direction, states):
        shs = 0.5 * torch.dot(step_direction, self.hvp(step_direction, states))
        beta = torch.sqrt(self.kl_constraint / (shs + 1e-8))
        return beta

    def get_flat_params(self, model):
        params = [p.view(-1) for p in model.parameters()]
        return torch.cat(params)

    def set_flat_params(self, model, flat_params):
        prev_ind = 0
        for p in model.parameters():
            flat_size = p.numel()
            p.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(p.size()))
            prev_ind += flat_size


def train(episodes, device="cpu"):
    env = gym.make("LunarLander-v2")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device=device)

    recent_rewards = deque(maxlen=20)
    solved_score = 200.0  # 参考值：LunarLander-v2 的“好表现”阈值因训练而异

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        average_reward = 0
        total_reward = 0

        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.add(state, action, reward, done)
            state = next_state
            total_reward += reward

        agent.update()

        average_reward += total_reward
        if (episode + 1) % 5 == 0:
            average_reward /= 5
            print(f"Episode {episode+1}, Average Reward: {average_reward:.2f}")
            average_reward = 0

        recent_rewards.append(total_reward)
        if len(recent_rewards) == 20 and np.mean(recent_rewards) >= solved_score:
            print(f"Solved (avg reward >= {solved_score}) in {episode+1} episodes!")
            break

    env.close()
    return agent


def play_trained_agent(agent, episodes=3, render_mode="human", device="cpu"):
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    agent.policy.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n🎮 Episode {ep + 1} start")
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1

        print(f"✅ Episode {ep + 1} finished | Steps: {step}, Total reward: {total_reward}")

    env.close()
    agent.policy.train()


if __name__ == "__main__":
    device = torch.device("cpu")
    agent = train(500, device=device)
    play_trained_agent(agent, episodes=3, render_mode="human", device=device)

