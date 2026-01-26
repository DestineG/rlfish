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

def eval_onestep(env, V, pi, gama=0.9):
    new_V = {}

    for state in env.states():
        # 终止状态：价值定义为 0
        if state == env.goal_states:
            new_V[state] = 0.0
            continue

        v = 0.0
        for action, action_prob in pi[state].items():
            next_s = env.next_state(state, action)
            r = env.reward(next_s)
            v += action_prob * (r + gama * V[next_s])

        new_V[state] = v

    return new_V

def policy_eval(env, V, pi, gama=0.9, theta=1e-6):
    while True:
        old_V = V.copy()
        V = eval_onestep(env, V, pi, gama)
        delta = max(abs(old_V[s] - V[s]) for s in env.states())
        if delta < theta:
            break
    return V

if __name__ == "__main__":
    env = GridWorld()
    # 每个状态的初始价值为 0
    V = defaultdict(lambda: 0.0)
    # 每个状态的均匀随机策略
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # 进行策略评估
    V = policy_eval(env, V, pi, gama=0.9, theta=1e-6)
    env.render_v(V, title="Policy Evaluation Result")