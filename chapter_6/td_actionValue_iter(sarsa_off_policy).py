import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

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
    
    def render_q(self, Q, title="Q-values Triangles"):
        """
        在每个方格画两条对角线形成四个等腰直角三角形，
        用绿色填充表示最优动作方向
        Q: dict, key=(state, action), value=Q(s,a)
        """
        fig, ax = plt.subplots(figsize=(self.width*1.5, self.height*1.5))

        for i in range(self.height):
            for j in range(self.width):
                if self.reward_map[i, j] is None:
                    # 墙
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="black"))
                    ax.text(j, i, "W", ha="center", va="center", color="white", fontsize=12)
                    continue

                # 背景颜色
                r = self.reward_map[i, j]
                if (i, j) == self.goal_states:
                    color = "lightgreen"
                elif r == -1:
                    color = "lightcoral"
                else:
                    color = "white"

                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=color, edgecolor="black"))

                # 四个动作 Q 值
                q_values = [Q[((i, j), a)] for a in range(4)]
                best_action = np.argmax(q_values)

                # 定义四个三角形的坐标（上、下、左、右）
                center = (j, i)
                half = 0.5
                triangles = {
                    0: [( (j-half,i-half), (j+half,i-half), center )],  # Up
                    1: [( (j-half,i+half), (j+half,i+half), center )],  # Down
                    2: [( (j-half,i-half), (j-half,i+half), center )],  # Left
                    3: [( (j+half,i-half), (j+half,i+half), center )],  # Right
                }

                # 画绿色填充表示最优动作
                for action, tri_list in triangles.items():
                    for tri in tri_list:
                        if action == best_action:
                            ax.add_patch(plt.Polygon(tri, color='green', alpha=0.6))
                        else:
                            ax.add_patch(plt.Polygon(tri, color='white', alpha=0.0))  # 空白透明

                # 可选：在每个三角形里标注 Q 值
                offset = 0.25
                ax.text(j, i-half+0.1, f"{q_values[0]:.2f}", ha="center", va="center", fontsize=7, color="black")  # Up
                ax.text(j, i+half-0.1, f"{q_values[1]:.2f}", ha="center", va="center", fontsize=7, color="black")  # Down
                ax.text(j-half+0.1, i, f"{q_values[2]:.2f}", ha="center", va="center", fontsize=7, color="black")  # Left
                ax.text(j+half-0.1, i, f"{q_values[3]:.2f}", ha="center", va="center", fontsize=7, color="black")  # Right

        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))
        ax.set_xlim(-0.5, self.width-0.5)
        ax.set_ylim(self.height-0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title)
        plt.show()


def greedy_probs(Q, state, action_size, epsilon=0.0):
    qs = [Q[(state, a)] for a in range(action_size)]
    max_q = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {a: base_prob for a in range(action_size)}
    action_probs[max_q] += 1.0 - epsilon

    return action_probs

class Agent:
    def __init__(self):
        self.gama = 0.9
        self.action_size = 4

        default_policy = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: default_policy.copy())
        self.b = defaultdict(lambda: default_policy.copy())
        self.Q = defaultdict(lambda: 0.0)
        self.memory = deque(maxlen=2)
    
    def get_action(self, state):
        # 行为策略
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset_memory(self):
        self.memory.clear()
    
    # 策略更新
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[(next_state, next_action)]
        rho = 1 if done else self.pi[next_state][next_action] / self.b[next_state][next_action]
        target = rho * (reward + self.gama * next_q)
        self.Q[(state, action)] += 0.1 * (target - self.Q[(state, action)])

        # 目标策略更新
        self.pi[state] = greedy_probs(self.Q, state, self.action_size, epsilon=0)
        # 行为策略更新
        self.b[state] = greedy_probs(self.Q, state, self.action_size, epsilon=0.1)

def td_eval():
    env = GridWorld()
    agent = Agent()
    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # 进行一个回合
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # 实时更新
            agent.update(state, action, reward, done)
            state = next_state

    # 提取状态价值函数 V 和 策略 pi
    V = np.zeros(env.shape)
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            qs = [agent.Q[(state, a)] for a in env.actions()]
            V[i, j] = max(qs)
    env.render_v(V, title="TD Control: State Value Function V")
    env.render_q(agent.Q, title="TD Control: Q-values & Policy")

if __name__ == "__main__":
    td_eval()