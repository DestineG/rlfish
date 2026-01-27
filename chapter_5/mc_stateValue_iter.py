import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.action_vector = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_meaning = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        self.reward_map = np.array(
            [[0, 0, 0, 1],
             [0, None, 0, -1],
             [0, 0, 0, 0]]
        )
        self.goal_states = (0, 3)
        self.wall_states = (1, 1)
        self.start_state = (2, 0)
        self.state = self.start_state
    
    @property
    def height(self):
        return self.reward_map.shape[0]
    
    @property
    def width(self):
        return self.reward_map.shape[1]
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space
    
    def states(self):
        for i in range(self.height):
            for j in range(self.width):
                yield (i, j)
    
    def next_state(self, state, action):
        move = self.action_vector[action]
        next_s = (state[0] + move[0], state[1] + move[1])
        if (0 <= next_s[0] < self.height and
            0 <= next_s[1] < self.width and
            next_s != self.wall_states):
            return next_s
        return state
    
    def reward(self, state):
        return self.reward_map[state]
    
    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        next_s = self.next_state(self.state, action)
        r = self.reward(next_s)
        self.state = next_s
        done = (next_s == self.goal_states)
        return next_s, r, done

    def render_v(self, V, title="Value Function"):
        """
        使用 matplotlib 以 grid 形式渲染状态价值函数 V
        """
        fig, ax = plt.subplots()

        # 基础背景
        grid = np.zeros(self.shape)
        ax.imshow(grid, cmap="Greys", alpha=0.2)

        for i in range(self.height):
            for j in range(self.width):
                if self.reward_map[i, j] is None:
                    # Wall
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="black")
                    )
                    ax.text(j, i, "W", ha="center", va="center", color="white", fontsize=12)
                else:
                    # 普通状态 / 终止状态
                    v = V[i, j]
                    r = self.reward_map[i, j]

                    if (i, j) == self.goal_states:
                        color = "lightgreen"
                    elif r == -1:
                        color = "lightcoral"
                    else:
                        color = "white"

                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      edgecolor="black", facecolor=color)
                    )
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)

        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))
        ax.set_title(title)

        # 坐标系调整（符合 grid world 直觉）
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")

        plt.show()
    
    def render_q(self, pi, title="Policy (Action Directions)"):
        """
        使用 matplotlib 以 grid 形式渲染策略 pi，
        显示每个状态的最优动作（箭头方向）
        """
        fig, ax = plt.subplots()

        # 基础背景
        grid = np.zeros(self.shape)
        ax.imshow(grid, cmap="Greys", alpha=0.2)

        for i in range(self.height):
            for j in range(self.width):
                # 如果是墙
                if self.reward_map[i, j] is None:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="black")
                    )
                    ax.text(j, i, "W", ha="center", va="center", color="white", fontsize=12)
                else:
                    # 普通状态 / 终止状态
                    r = self.reward_map[i, j]

                    # 终止状态显示绿色
                    if (i, j) == self.goal_states:
                        color = "lightgreen"
                    elif r == -1:
                        color = "lightcoral"
                    else:
                        color = "white"

                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    edgecolor="black", facecolor=color)
                    )
                    # 没有显示值，只显示箭头方向
                    ax.text(j, i, "", ha="center", va="center", fontsize=10)

                    # 绘制箭头表示最优动作
                    best_action = max(pi[(i, j)], key=pi[(i, j)].get)  # 获取最佳动作
                    dx, dy = self.action_vector[best_action]  # 获取动作的偏移量
                    ax.arrow(j, i, dy * 0.2, dx * 0.2, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))
        ax.set_title(title)

        # 坐标系调整（符合 grid world 直觉）
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")

        plt.show()


class Agent:
    def __init__(self):
        self.gama = 0.9
        self.action_size = 4

        default_policy = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: default_policy.copy())
        self.V = defaultdict(lambda: 0.0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add_memory(self, state, action, reward):
        self.memory.append((state, action, reward))
    
    def reset_memory(self):
        self.memory.clear()
    
    # 状态期望评估更新
    def eval(self):
        G = 0
        # 反向遍历回合数据
        for state, action, reward in reversed(self.memory):
            G = reward + self.gama * G
            self.cnts[state] += 1
            alpha = 1 / self.cnts[state]
            self.V[state] += alpha * (G - self.V[state])


def mc_eval():
    env = GridWorld()
    agent = Agent()
    num_episodes = 5000

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_memory()
        done = False

        # 回合数据收集
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add_memory(state, action, reward)
            state = next_state

        # 回合结束后进行评估更新
        agent.eval()

    # 转换 V 为二维数组以便渲染
    V_array = np.zeros(env.shape)
    for i in range(env.height):
        for j in range(env.width):
            V_array[i, j] = agent.V[(i, j)]

    env.render_v(V_array, title="Estimated Value Function after MC Evaluation")

if __name__ == "__main__":
    mc_eval()