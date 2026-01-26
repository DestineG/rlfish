import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.action_vector = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_meaning = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        self.reward_map = np.array(
            [[0, 0, 0, 2],
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


def value_iteration(env, gama=0.9, theta=1e-6):
    V = defaultdict(lambda: 0.0)

    while True:
        old_V = V.copy()
        delta = 0.0

        for state in env.states():
            # 终止状态：价值定义为 0
            if state == env.goal_states:
                V[state] = 0.0
                continue

            action_values = []
            for action in env.actions():
                next_s = env.next_state(state, action)
                r = env.reward(next_s)
                action_value = r + gama * old_V[next_s]
                action_values.append(action_value)

            # 选择最大动作价值作为状态价值(隐含策略为贪婪策略)
            V[state] = max(action_values)
            delta = max(delta, abs(old_V[state] - V[state]))

        if delta < theta:
            break

    return V

def argmax_dict(d):
    """返回字典中值最大的键"""
    max_key = None
    max_value = float("-inf")
    for k, v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key

def greedy_policy(env, V, gama=0.9):
    pi = {}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_s = env.next_state(state, action)
            r = env.reward(next_s)
            action_values[action] = r + gama * V[next_s]

        best_action = argmax_dict(action_values)
        # 构造确定性贪婪策略
        pi[state] = {a: 1.0 if a == best_action else 0.0 for a in env.actions()}

    return pi

if __name__ == "__main__":
    env = GridWorld()
    V = value_iteration(env)
    pi = greedy_policy(env, V)
    env.render_q(pi)