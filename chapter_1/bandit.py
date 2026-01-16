import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)
        print(f"Initialized bandit with rates: {self.rates}")
    
    def play(self, arm=None):
        if arm == None: arm = np.random.randint(self.arms)
        if arm < 0 or arm >= len(self.rates): raise ValueError("Invalid arm index")
        reward = 1 if np.random.rand() < self.rates[arm] else 0
        return arm, reward

class Agent:
    def __init__(self, epsilon=0.1, action_size=10):
        self.epsilon = epsilon
        self.action_size = action_size
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.Qs)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

if __name__ == "__main__":
    arms = 5
    # 无状态环境初始化
    bandit = Bandit(arms=arms)
    runs = 200
    steps = 1000
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
