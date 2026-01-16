import numpy as np
import matplotlib.pyplot as plt


class NonStatBandit:
    def __init__(self, arms=10, change_interval=100):
        self.arms = arms
        self.change_interval = change_interval
        self.rates = np.random.rand(arms)
        self.step_count = 0
    
    def play(self, arm=None):
        self.step_count += 1
        # 环境内部状态定期变化
        if self.step_count % self.change_interval == 0:
            self.rates += np.random.normal(0, 0.1, size=self.arms)
            reset_mask = (self.rates < 0) | (self.rates > 1)
            self.rates[reset_mask] = np.random.rand(np.sum(reset_mask))
        if arm == None: arm = np.random.randint(self.arms)
        if arm < 0 or arm >= len(self.rates): raise ValueError("Invalid arm index")
        reward = 1 if np.random.rand() < self.rates[arm] else 0
        return arm, reward

class Agent:
    def __init__(self, epsilon=0.1, action_size=10, alpha=0.1):
        self.epsilon = epsilon
        self.action_size = action_size
        self.alpha = alpha
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.Qs)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

if __name__ == "__main__":
    arms = 5
    # 非静态环境初始化，此处的环境会随着时间变化，但在这个场景下不会告知智能体，
    # 所以对于智能体来说仍然是无状态的，所以可以共享环境
    # 注：虽然复用 bandit 实例，但其随机演化在统计上
    # 等价于为每个 epsilon 使用一个独立的非稳态 bandit
    bandit = NonStatBandit(arms=arms, change_interval=100)
    runs = 200
    steps = 10000
    epsilons = [0.0, 0.05, 0.2, 0.5]
    total_rates = np.zeros((len(epsilons), runs, steps))

    for idx, epsilon in enumerate(epsilons):

        for run in range(runs):
            # 智能体初始化
            agent = Agent(epsilon=epsilon, action_size=arms)
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                _, reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))
            total_rates[idx, run] = rates

    # 平均奖励可视化
    avg_rates = np.average(total_rates, axis=1)

    plt.figure(figsize=(8, 5))
    for i, epsilon in enumerate(epsilons):
        plt.plot(avg_rates[i], label=f"epsilon = {epsilon}")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
