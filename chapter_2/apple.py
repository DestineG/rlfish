import numpy as np
import matplotlib.pyplot as plt


class AppleMap:
    def __init__(self, init_ind=0):
        self.ind = init_ind
        self.appleMap = [0, 1]
    
    def getInd(self):
        return self.ind

    def move(self, action='left'):
        if action == 'left':
            if self.ind == 0:
                reward = -1
            else:
                self.ind -= 1
                reward = self.appleMap[self.ind]
                self.appleMap[self.ind-1] = 1
        elif action == 'right':
            if self.ind == len(self.appleMap) - 1:
                reward = -1
            else:
                self.ind += 1
                reward = self.appleMap[self.ind]
                self.appleMap[self.ind] = 0
        else:
            raise ValueError("Invalid action")
        return self.ind, reward

if __name__ == "__main__":
    u = [
        {
            0: "left",
            1: "left"
        },
        {
            0: "left",
            1: "right"
        },
        {
            0: "right",
            1: "left"
        },
        {
            0: "right",
            1: "right"
        }
    ]
    gama = 0.9
    steps = 1000
    total_values = np.zeros((2, len(u), steps))

    # 不同初始位置和策略组合的实验
    for ind in range(2):
        for ui, policy in enumerate(u):
            env = AppleMap(init_ind=ind)
            total_reward = 0
            values = []
            for step in range(steps):
                action = policy[env.getInd()]
                _, reward = env.move(action)
                total_reward += (gama ** step) * reward
                values.append(total_reward)
            total_values[ind, ui] = values

    # values 可视化（折线图）
    values = total_values[:, :, -1]   # shape: (2, num_policy)
    x = np.arange(2)  # 初始位置 0, 1
    num_policies = values.shape[1]

    plt.figure(figsize=(6, 4))

    # 对每个策略画一条线
    for ui in range(num_policies):
        plt.plot(x, values[:, ui], marker='o', label=f'Policy {ui}')

    plt.xlabel("Initial Position")
    plt.ylabel("Total Discounted Reward")
    plt.title("Total Discounted Reward by Initial Position and Policy")
    plt.xticks(x, ['Start 0', 'Start 1'])
    plt.legend()
    plt.grid(True)
    plt.show()
