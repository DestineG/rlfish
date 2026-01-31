import matplotlib.pyplot as plt
import torch
import numpy as np

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


def one_hot(state, height=4, width=4):
    one_hot_vector = np.zeros(height * width)
    index = state[0] * width + state[1]
    one_hot_vector[index] = 1
    return one_hot_vector

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, epsilon=0.1):
        # 将 qNet 当作Q函数使用
        self.qnet = Qnet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        # 奖励衰减越大越偏向于最短路径
        self.gamma = gamma
        # 探索力度越大越偏向于安全路径
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.qnet(state_tensor)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        # 当前状态的 Q 值 shape: [1, action_dim]
        q_values = self.qnet(state_tensor)
        # 下一状态的 Q 值 shape: [1, action_dim]
        next_q_values = self.qnet(next_state_tensor)

        done = int(done)

        # 构造目标 Q 值 这里假设下一状态Q值能正确引导当前状态的Q值更新
        target = reward + (1 - done) * self.gamma * torch.max(next_q_values).item()
        # 复制当前 Q 值用于构造目标 target_f.shape: [1, action_dim]
        target_f = q_values.clone().detach()
        # 只更新所采取动作的 Q 值 target_f[0].shape: [action_dim]
        target_f[0][action] = target

        # 计算当前状态下action Q值预测与真实Q值的误差
        loss = torch.nn.functional.mse_loss(q_values, target_f)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    env = GridWorld()
    state_dim = env.height * env.width
    action_dim = len(env.actions())
    agent = QLearningAgent(state_dim, action_dim, lr=0.01, gamma=0.9, epsilon=0.1)

    num_episodes = 500
    loss_list = []
    for episode in range(num_episodes):
        state = env.reset()
        state_oh = one_hot(state, height=env.height, width=env.width)
        done = False
        loss = 0
        step = 1
        total_reward = 0
        while not done:
            action = agent.select_action(state_oh)
            next_state, reward, done = env.step(action)
            next_state_oh = one_hot(next_state, height=env.height, width=env.width)
            loss += agent.update(state_oh, action, reward, next_state_oh, done)
            step += 1
            state_oh = next_state_oh
            total_reward += reward
        loss_list.append(loss/step)
    
    # loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Episode')
    plt.grid(True)
    plt.show()

    # 提取 Q 值用于渲染
    Q_values = {}
    for i in range(env.height):
        for j in range(env.width):
            state_oh = one_hot((i, j), height=env.height, width=env.width)
            state_tensor = torch.FloatTensor(state_oh).unsqueeze(0)
            q_values = agent.qnet(state_tensor).detach().numpy().flatten()
            for a in range(action_dim):
                Q_values[((i, j), a)] = q_values[a]

    env.render_q(Q_values, title="Learned Q-values after Q-Learning")