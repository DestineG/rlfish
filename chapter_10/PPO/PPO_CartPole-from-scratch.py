import torch
import torch.nn as nn
import torch.functional as F
import gymnasium as gym
from collections import deque
import numpy as np

import sys


class VNet(nn.Module):
    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
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
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class PPOAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.gamma = 0.99           # 折扣因子
        self.kl_constraint = 0.01   # KL 散度约束
        self.ls_alpha = 0.5         # 线性搜索衰减系数
        self.clip_epsilon = 0.2        # PPO 的剪切范围

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
        # 构造离散动作的概率分布
        dist = torch.distributions.Categorical(probs)
        # 采样动作
        action = dist.sample()
        # 返回动作的索引和对应的 log 概率
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

        # 计算每个时刻的动作价值
        Qs = []
        next_state_value = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # 此处不需要使用 done 标志来重置 next_state_value，因为在此项目中的memory始终只有一条完整的轨迹
            next_state_value = reward + self.gamma * next_state_value
            Qs.insert(0, next_state_value)
        Qs_tensor = torch.tensor(Qs, dtype=torch.float32, device=self.device)
        if len(Qs_tensor.shape) == 1:
            Qs_tensor = Qs_tensor.unsqueeze(1)

        # 更新值函数网络
        # (T, state_dim) -> (T, 1)
        state_values = self.vnet(states_tensor)
        v_loss = torch.nn.functional.mse_loss(state_values, Qs_tensor)
        self.vnet_optimizer.zero_grad()
        v_loss.backward()
        self.vnet_optimizer.step()

        # 计算优势函数
        advantages = Qs_tensor - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算 log_probs
        # (T, state_dim) -> (T, action_dim)
        probs = self.policy(states_tensor)
        # gather 根据 actions_tensor 中的动作索引从 probs 中选择对应的概率值，并计算 log 概率
        log_probs = torch.log(probs.gather(1, actions_tensor))
        old_log_probs = log_probs.detach()


        # ############# PPO Clip_v1 的更新步骤 #############
        # # Advantages * min(θ(a|s) / θ_old(a|s), clip(θ(a|s) / θ_old(a|s), 1 - ε, 1 + ε))

        # # 计算 ∇_θ(L(θ) - η(θ_old)) 以及 在旧策略处的优势函数加权和
        # ratio = torch.exp(log_probs - old_log_probs)
        # clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        # min_clipped_ratio = torch.min(ratio, clipped_ratio)
        # # NOTE: L(θ) - η(θ_old) = E[ (θ(a|s) / θ_old(a|s)) * A ] = E[ exp(log θ(a|s) - log θ_old(a|s)) * A ]
        # surrogate_loss = -torch.mean(min_clipped_ratio * advantages)
        # self.policy.zero_grad()
        # surrogate_loss.backward()
        # self.policy_optimizer.step()


        ############# PPO Clip_v2 的更新步骤 #############
        # min(θ(a|s) / θ_old(a|s) * A, clip(θ(a|s) / θ_old(a|s), 1 - ε, 1 + ε) * A)

        # 计算 ∇_θ(L(θ) - η(θ_old)) 以及 在旧策略处的优势函数加权和
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        # NOTE: L(θ) - η(θ_old) = E[ (θ(a|s) / θ_old(a|s)) * A ] = E[ exp(log θ(a|s) - log θ_old(a|s)) * A ]
        surrogate_loss = -torch.mean(torch.min(surr1, surr2))
        self.policy.zero_grad()
        surrogate_loss.backward()
        self.policy_optimizer.step()


        ############# TRPO 的更新步骤 #############

        # # 计算 ∇_θ(L(θ) - η(θ_old)) 以及 在旧策略处的优势函数加权和
        # ratio = torch.exp(log_probs - old_log_probs)
        # # NOTE: L(θ) - η(θ_old) = E[ (θ(a|s) / θ_old(a|s)) * A ] = E[ exp(log θ(a|s) - log θ_old(a|s)) * A ]
        # surrogate_loss = torch.mean(ratio * advantages)
        # grads = torch.autograd.grad(surrogate_loss, self.policy.parameters())
        # # 将梯度展平为一个向量
        # loss_grad = torch.cat([g.view(-1) for g in grads]).detach()

        # # 共轭梯度法求解 θ 更新方向 H^-1 * g
        # step_direction = self.cg(loss_grad, states_tensor)

        # # 根据 KL 约束计算初始步长
        # step_size = self.get_step_size(step_direction, states_tensor)

        # # 线性搜索找到最优的更新步长
        # full_step = step_direction * step_size
        # old_params = self.get_flat_params(self.policy)
        # success = False
        # for i in range(10):
        #     # 参数更新步长从最大步长开始逐渐减小
        #     new_params = old_params + (self.ls_alpha ** i) * full_step
        #     self.set_flat_params(self.policy, new_params)

        #     # 计算新的 surrogate loss 和 KL 散度，确保满足约束并且有改进(比较新旧优势函数的加权平均)
        #     with torch.no_grad():
        #         new_probs = self.policy(states_tensor)
        #         new_log_probs = torch.log(new_probs.gather(1, actions_tensor))
        #         # NOTE: L(θ) - η(θ_old) = E[ (θ(a|s) / θ_old(a|s)) * A ] = E[ exp(log θ(a|s) - log θ_old(a|s)) * A ]
        #         new_surrogate_loss = torch.mean(torch.exp(new_log_probs - old_log_probs) * advantages)
        #         kl_divergence = torch.sum(probs * torch.log(probs / (new_probs + 1e-8)), dim=1).mean()

        #         if kl_divergence < self.kl_constraint and new_surrogate_loss > surrogate_loss:
        #             success = True
        #             break

        # # 如果没有找到满足条件的更新步长，恢复原始参数
        # if not success:
        #     self.set_flat_params(self.policy, old_params)
        
        self.memory = []

    def hvp(self, vector, states):
        '''计算 Hessian-Vector Product H * vector，其中 H 是 KL 散度的 Hessian'''
        # KL(θ_old || θ)
        self.policy.zero_grad()
        probs = self.policy(states)
        old_probs = probs.detach()
        kl = torch.sum(old_probs * torch.log(old_probs / (probs + 1e-8)), dim=1).mean()

        # ∇_θ KL(θ_old || θ)
        kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        # 将梯度展平为一个向量
        flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])

        # ∇_θ KL(θ_old || θ) · vector
        fk_grad_vector = (flat_kl_grad * vector).sum()

        # ∇_θ (∇_θ KL(θ_old || θ) · vector)
        hvp = torch.autograd.grad(fk_grad_vector, self.policy.parameters())
        # 将 Hessian-Vector Product 展平为一个向量
        flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp]).detach()

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
            if new_rdotr < 1e-10: break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def get_step_size(self, step_direction, states):
        '''
        根据共轭梯度法的结果在约束条件下计算初始步长

        1/2 * (β * step_direction)^T * H * (β * step_direction) <= kl_constraint

        β = sqrt(kl_constraint / [1/2 * (step_direction^T * H * step_direction)])
        '''
        # 1/2 * (step_direction^T * H * step_direction)
        shs = 0.5 * torch.dot(step_direction, self.hvp(step_direction, states))

        # β = sqrt(kl_constraint / shs)
        beta = torch.sqrt(self.kl_constraint / (shs + 1e-8))
        return beta

    def get_flat_params(self, model):
        # 将模型参数展平为一个向量
        params = [p.view(-1) for p in model.parameters()]
        return torch.cat(params)

    def set_flat_params(self, model, flat_params):
        # 将展平的参数向量重新赋值给模型参数
        prev_ind = 0
        for p in model.parameters():
            flat_size = p.numel()
            p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
            prev_ind += flat_size

def train(episodes, device="cpu"):
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device=device)
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
