import torch
import gymnasium as gym
import numpy as np
from collections import deque


class VNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

class TRPOAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.gamma = 0.99
        self.kl_constraint = 0.01  # 最大的 KL 散度约束
        self.ls_alpha = 0.5        # 线性搜索衰减系数
        
        self.vnet = VNet(state_dim).to(self.device)
        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.vnet_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=1e-3)
        
        self.memory = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def add(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def get_flat_params(self, model):
        params = [p.view(-1) for p in model.parameters()]
        return torch.cat(params)

    def set_flat_params(self, model, flat_params):
        prev_ind = 0
        for p in model.parameters():
            flat_size = p.numel()
            p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
            prev_ind += flat_size

    def conjugate_gradient(self, b, states, nsteps=10):
        """求解 Hx = b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            _Hp = self.hessian_vector_product(p, states)
            alpha = rdotr / (torch.dot(p, _Hp) + 1e-8)
            x += alpha * p
            r -= alpha * _Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10: break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def hessian_vector_product(self, vector, states):
        """计算 Fisher 信息矩阵与向量的乘积"""
        self.policy.zero_grad()
        probs = self.policy(states)
        old_probs = probs.detach()
        kl = torch.sum(old_probs * torch.log(old_probs / (probs + 1e-8)), dim=1).mean()
        
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])
        
        kl_v = (flat_grad_kl * vector).sum()
        grads_v = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads_v]).detach()
        
        return flat_grad_grad_kl + 0.1 * vector # 0.1 为阻尼系数

    def update(self):
        states = torch.tensor(np.array([m[0] for m in self.memory]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.int64).to(self.device).view(-1, 1)
        rewards = [m[2] for m in self.memory]
        dones = [m[3] for m in self.memory]

        # 计算 Return 和 Advantage
        R = 0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        values = self.vnet(states).squeeze()
        advantages = (returns - values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 更新 Value 网络
        vf_loss = torch.nn.functional.mse_loss(values, returns)
        self.vnet_optimizer.zero_grad()
        vf_loss.backward()
        self.vnet_optimizer.step()

        # TRPO 策略更新
        # 计算当前策略的梯度 g
        probs = self.policy(states)
        log_probs = torch.log(torch.gather(probs, 1, actions).squeeze())
        old_log_probs = log_probs.detach()
        
        # 此处构造的损失函数的梯度与梯度策略法的梯度一致
        loss = (torch.exp(log_probs - old_log_probs) * advantages).mean()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()

        # 计算下降方向 s = H^-1 * g
        step_dir = self.conjugate_gradient(loss_grad, states)

        # 计算步长 beta
        shs = 0.5 * torch.dot(step_dir, self.hessian_vector_product(step_dir, states))
        lm = torch.sqrt(shs / self.kl_constraint)
        full_step = step_dir / lm
        
        # 线性搜索确保改进和 KL 约束
        old_params = self.get_flat_params(self.policy)
        success = False
        for i in range(10):
            new_params = old_params + (self.ls_alpha**i) * full_step
            self.set_flat_params(self.policy, new_params)
            
            with torch.no_grad():
                new_probs = self.policy(states)
                new_log_probs = torch.log(torch.gather(new_probs, 1, actions).squeeze())
                new_loss = (torch.exp(new_log_probs - old_log_probs) * advantages).mean()
                kl = torch.sum(probs * torch.log(probs / (new_probs + 1e-8)), dim=1).mean()
                
                if kl < self.kl_constraint and new_loss > loss:
                    success = True
                    break
        
        if not success:
            self.set_flat_params(self.policy, old_params)

        self.memory = []

# --- 训练逻辑基本不变 ---

def train(episodes, device="cpu"):
    env = gym.make("CartPole-v1")
    agent = TRPOAgent(env.observation_space.shape[0], env.action_space.n, device=device)
    recent_rewards = deque(maxlen=20)
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.add(state, action, reward, done)
            state = next_state
            total_reward += reward
        
        agent.update()
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}")
        
        recent_rewards.append(total_reward)
        if len(recent_rewards) == 20 and np.mean(recent_rewards) >= 450.0:
            print(f"Solved in {episode+1} episodes!")
            break
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
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    agent = train(500, device=device)
    play_trained_agent(agent, episodes=3, render_mode="human", device=device)
