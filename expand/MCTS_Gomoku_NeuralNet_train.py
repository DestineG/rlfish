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
                state[2] = player # 当前玩家颜色标识 (1 或 2)，让模型区分黑白棋

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
batch_size = 128
split_ratio = 0.9
split_idx = int(len(samples) * split_ratio)
train_dataset = GomokuDataset(samples[:split_idx], split="train")
val_dataset = GomokuDataset(samples[split_idx:], split="val")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

input_channels = 3
board_size = 8

class SEResBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 1. 全局平均池化 -> (b, c, 1, 1)
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c) # 2. 展平为 (b, c)
        
        # 3. 经过全连接层 -> (b, c)
        y = self.se_fc(y).view(b, c, 1, 1) # 4. 恢复维度为 (b, c, 1, 1)
        
        # 5. 通道注意力加权
        out = out * y.expand_as(out)
        
        out += residual
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, num_res_blocks=6, board_size=8): # 稍微加深网络
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        
        # 1. 特征提取层
        self.start_block = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # 提升宽度到 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 2. 堆叠带注意力的残差块
        self.res_blocks = nn.Sequential(
            *[SEResBlock(128) for _ in range(num_res_blocks)]
        )

        # 3. Policy Head (策略头)
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)

        # 4. Value Head (价值头)
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(4 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.start_block(x)
        x = self.res_blocks(x)

        # 策略预测
        p = self.policy_head(x).view(x.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 价值预测
        v = self.value_head(x).view(x.size(0), -1)
        value = self.value_fc(v)

        return policy_logits, value


if __name__ == "__main__":
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    num_epochs = 200
    lr = 0.001
    weights = [1.0, 10.0] # 平衡策略和价值的训练

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 更新 history 结构
    history = {
        'train_steps': [], # 记录 step
        'train_policy': [], 'train_value': [], 
        'val_epochs': [],  # 记录 epoch
        'val_policy': [], 'val_value': []
    }

    global_step = 0
    log_interval = 10

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        
        for states, policies, values in train_dataloader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            policy_logits, value_preds = model(states)
            
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = criterion_policy(log_probs, policies)
            value_loss = criterion_value(value_preds.squeeze(), values.squeeze())
            loss = weights[0] * policy_loss + weights[1] * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # 每隔 10 step 记录一次训练 Loss
            if global_step % log_interval == 0:
                history['train_steps'].append(global_step)
                history['train_policy'].append(policy_loss.item())
                history['train_value'].append(value_loss.item())

        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        total_val_policy = 0.0
        total_val_value = 0.0
        
        with torch.no_grad():
            for states, policies, values in val_dataloader:
                states, policies, values = states.to(device), policies.to(device), values.to(device)
                policy_logits, value_preds = model(states)
                log_probs = F.log_softmax(policy_logits, dim=1)
                
                total_val_policy += criterion_policy(log_probs, policies).item()
                total_val_value += criterion_value(value_preds.squeeze(), values.squeeze()).item()

        avg_val_p = total_val_policy / len(val_dataloader)
        avg_val_v = total_val_value / len(val_dataloader)
        
        history['val_epochs'].append(global_step)
        history['val_policy'].append(avg_val_p)
        history['val_value'].append(avg_val_v)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                f"Val Policy Loss: {avg_val_p:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Policy Loss
    ax1.plot(history['train_steps'], history['train_policy'], 'b-', alpha=0.3, label='Train Step Loss')
    train_p_smooth = np.convolve(history['train_policy'], np.ones(5)/5, mode='same')
    ax1.plot(history['train_steps'], train_p_smooth, 'b-', label='Train Smooth Loss')
    ax1.plot(history['val_epochs'], history['val_policy'], 'ro-', label='Val Epoch Loss')
    ax1.set_title('Policy Loss (KL Divergence) over Steps')
    ax1.legend()

    # Value Loss
    ax2.plot(history['train_steps'], history['train_value'], 'r-', alpha=0.3, label='Train Step Loss')
    train_v_smooth = np.convolve(history['train_value'], np.ones(5)/5, mode='same')
    ax2.plot(history['train_steps'], train_v_smooth, 'r-', label='Train Smooth Loss')
    ax2.plot(history['val_epochs'], history['val_value'], 'bo-', label='Val Epoch Loss')
    ax2.set_title('Value Loss (MSE) over Steps')
    ax2.set_xlabel('Global Training Steps')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()