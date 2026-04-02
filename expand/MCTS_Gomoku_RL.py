# ------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------
import json
import numpy as np

def load_gomoku_data(file_path):
    data_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            board_size = game['board_size']
            winner = game['winner']
            moves = game['moves']
            
            for move in moves:
                board = np.array(move['board'], dtype=np.int32)
                player = move['player'] # 当前落子方
                opponent = 3 - player # 对手编号

                # 棋盘状态构造
                # 通道0: 当前玩家棋子; 通道1: 对手棋子; 通道2: 当前玩家颜色标识
                # TODO: 添加最近落子点通道以帮助模型感知局部的攻防焦点
                state = np.zeros((3, board_size, board_size), dtype=np.float32)
                state[0] = (board == player).astype(np.float32)
                state[1] = (board == opponent).astype(np.float32)
                state[2] = 1.0 if player == 1 else 0.0 # 假设1是黑棋

                # 访问频率构造 (Policy Target)
                mcts_raw_visits = move['mcts_raw_visits']
                visits = np.zeros((board_size, board_size), dtype=np.float32)
                for key, v in mcts_raw_visits.items():
                    x, y = map(int, key.split(','))
                    visits[x][y] = v
                
                # 归一化为概率分布 (作为策略网络的训练标签)
                s = visits.sum()
                if s > 0:
                    visits /= s

                # 局面胜率构造 (Value Target)
                mcts_raw_values = move['mcts_raw_values']
                values_map = np.zeros((board_size, board_size), dtype=np.float32)
                for key, v in mcts_raw_values.items():
                    x, y = map(int, key.split(','))
                    values_map[x][y] = v
                
                # 使用 MCTS 的加权平均价值作为基础胜率
                victory_value = (visits * values_map).sum()

                # 结合最终胜负进行修正 (Value 训练目标控制在 [0, 1] 之间)
                if winner == player:
                    victory_value = min(1.0, victory_value * 1.1)
                elif winner != 0: # 输了
                    victory_value = max(0.0, victory_value * 0.9)
                
                # 存入列表
                data_pairs.append({
                    'state': state,          # Shape: (3, BS, BS)
                    'visits': visits,        # Shape: (BS, BS)
                    'victory': victory_value # Scalar
                })

    return data_pairs


samples = load_gomoku_data('train_data/gomoku_game.jsonl')
print(f"Loaded {len(samples)} training samples from data file.")

# ------------------------------------------------------------
# Data loading for Training
# ------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader

class GomokuDataset(Dataset):
    def __init__(self, pairs, split="train"):
        self.pairs = pairs
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        state = item['state']  # (C, H, W)
        probs = item['visits'] # (H, W)
        value = item['victory']


        if self.split == "train":
            # 随机旋转 (0, 90, 180, 270) 和 翻转
            n = np.random.randint(0, 8)
            # 旋转
            state = np.rot90(state, n % 4, axes=(1, 2))
            probs = np.rot90(probs, n % 4)
            # 翻转
            if n >= 4:
                state = np.flip(state, axis=2)
                probs = np.flip(probs, axis=1)

        return torch.from_numpy(state.copy()).float(), \
            torch.from_numpy(probs.copy().flatten()).float(), \
            torch.tensor([value], dtype=torch.float32)


# 转换并加载
split_ratio = 0.9
split_idx = int(len(samples) * split_ratio)
train_dataset = GomokuDataset(samples[:split_idx], split="train")
val_dataset = GomokuDataset(samples[split_idx:], split="val")
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

input_channels = 3
board_size = 8

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 跳跃连接
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, num_res_blocks=3):
        super(PolicyValueNet, self).__init__()
        
        # Backbone
        self.start_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 堆叠残差块
        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 确保输出在 [0, 1]
        )

    def forward(self, x):
        # 共享特征计算
        x = self.start_block(x)
        x = self.res_blocks(x)

        # 策略支路
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 价值支路
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        value = self.value_fc(v)

        return policy_logits, value


# 实例化模型
model = PolicyValueNet(num_res_blocks=3)

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------
import os
import torch.optim as optim
import matplotlib.pyplot as plt

checkpoint_dir = 'checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
model_path = os.path.join(checkpoint_dir, 'gomoku_policy_value_net.pth')
plot_path = os.path.join(checkpoint_dir, 'train_loss_curve.png')

num_epochs = 100
lr = 0.001

criterion_policy = nn.KLDivLoss(reduction='batchmean')
criterion_value = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# 更新 history 结构
history = {
    'train_policy': [], 'train_value': [], 'train_total': [],
    'val_policy': [], 'val_value': [], 'val_total': []
}

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_policy = 0.0
    total_train_value = 0.0
    
    for states, policies, values in train_dataloader:
        optimizer.zero_grad()
        policy_logits, value_preds = model(states)
        
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = criterion_policy(log_probs, policies)
        value_loss = criterion_value(value_preds.squeeze(), values.squeeze())
        loss = policy_loss + value_loss
        
        loss.backward()
        optimizer.step()
        
        total_train_policy += policy_loss.item()
        total_train_value += value_loss.item()

    scheduler.step()

    # 验证阶段
    model.eval()
    total_val_policy = 0.0
    total_val_value = 0.0
    
    with torch.no_grad():
        for states, policies, values in val_dataloader:
            policy_logits, value_preds = model(states)
            
            log_probs = F.log_softmax(policy_logits, dim=1)
            v_policy_loss = criterion_policy(log_probs, policies)
            v_value_loss = criterion_value(value_preds.squeeze(), values.squeeze())
            
            total_val_policy += v_policy_loss.item()
            total_val_value += v_value_loss.item()

    # 计算平均值
    avg_train_p = total_train_policy / len(train_dataloader)
    avg_train_v = total_train_value / len(train_dataloader)
    avg_val_p = total_val_policy / len(val_dataloader)
    avg_val_v = total_val_value / len(val_dataloader)

    history['train_policy'].append(avg_train_p)
    history['train_value'].append(avg_train_v)
    history['train_total'].append(avg_train_p + avg_train_v)
    history['val_policy'].append(avg_val_p)
    history['val_value'].append(avg_val_v)
    history['val_total'].append(avg_val_p + avg_val_v)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_p+avg_train_v:.4f} | "
              f"Val Loss: {avg_val_p+avg_val_v:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), model_path)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Policy Loss (Train vs Val)
ax1.plot(history['train_policy'], 'b-', label='Train Policy Loss')
ax1.plot(history['val_policy'], 'b--', alpha=0.6, label='Val Policy Loss')
ax1.set_title('Policy Network Loss (KL Divergence)')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Value Loss (Train vs Val)
ax2.plot(history['train_value'], 'r-', label='Train Value Loss')
ax2.plot(history['val_value'], 'r--', alpha=0.6, label='Val Value Loss')
ax2.set_title('Value Network Loss (MSE)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(plot_path)
plt.show()