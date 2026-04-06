import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from .MCTS_Gomoku_NeuralNet import PolicyValueNet

# ------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------
def load_gomoku_data(file_path):
    data_pairs = []
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return data_pairs

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            game_data = json.loads(line)

            board_size = game_data['board_size']
            winner = game_data['winner']  # 1, 2 或 0 (平局)
            moves_history = game_data['moves']
            
            for i, move_step in enumerate(moves_history):
                # 提取输入状态; Shape: (4, BS, BS)
                state = np.array(move_step['inputs'], dtype=np.float32)

                # 构造策略目标
                probs = np.zeros((board_size, board_size), dtype=np.float32)
                visit_dict = move_step['visit_counts']
                # print(sum(list(visit_dict.values())))
                # print(f"Processing move {i+1}/{len(moves_history)}: visit counts = {visit_dict}")
                # import sys
                # sys.exit(0)  # 临时退出以检查数据格式
                for pos_str, count in visit_dict.items():
                    r, c = map(int, pos_str.split(','))
                    probs[r][c] = count
                
                # 归一化
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs.fill(1.0 / (board_size * board_size))

                # 构造价值目标
                current_player = move_step['player']
                
                if winner == 0:
                    target_value = 0.5
                elif winner == current_player:
                    target_value = 1.0
                else:
                    target_value = 0.0

                mcts_v = move_step['value_estimate']
                # 经验公式平衡最终胜负与 MCTS 评估
                target_value = 0.8 * target_value + 0.2 * mcts_v

                data_pairs.append({
                    'state': state,          # (4, BS, BS)
                    'probs': probs.flatten(),# (BS*BS,)
                    'value': target_value    # 标量
                })

    return data_pairs

# ------------------------------------------------------------
# Dataset Definition
# ------------------------------------------------------------
class GomokuDataset(Dataset):
    def __init__(self, pairs, split="train"):
        self.pairs = pairs
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        state = item['state']  # (C, H, W)
        probs = item['probs']  # 一维 (H*W,)
        value = item['value']

        # 核心修复：旋转前必须 reshape 为二维
        bs = int(np.sqrt(len(probs))) 
        probs = probs.reshape(bs, bs)

        if self.split == "train":
            # 随机旋转 (0, 90, 180, 270) 和 翻转
            n = np.random.randint(0, 8)
            # 旋转轴 (1, 2) 对应 (H, W)
            state = np.rot90(state, n % 4, axes=(1, 2))
            probs = np.rot90(probs, n % 4)
            # 翻转
            if n >= 4:
                state = np.flip(state, axis=2)
                probs = np.flip(probs, axis=1)

        return (torch.from_numpy(state.copy()).float(), 
                torch.from_numpy(probs.copy().flatten()).float(), 
                torch.tensor([value], dtype=torch.float32))

# ------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------
def train(model_index=0, data_index=0, num_epochs=100):
    # 配置路径
    data_file = f'train_data/self_play_data_{data_index}.jsonl'
    checkpoint_dir = 'checkpoint'
    model_path = os.path.join(checkpoint_dir, f'gomoku_policy_value_net_{model_index}.pth')
    plot_path = os.path.join(checkpoint_dir, f'train_loss_curve_{model_index}.png')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 加载数据
    samples = load_gomoku_data(data_file)
    if not samples:
        print("No data to train on. Exiting.")
        return

    print(f"Loaded {len(samples)} training samples.")

    # 数据划分
    batch_size = 128
    split_ratio = 0.9
    split_idx = int(len(samples) * split_ratio)
    train_dataset = GomokuDataset(samples[:split_idx], split="train")
    val_dataset = GomokuDataset(samples[split_idx:], split="val")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PolicyValueNet(num_res_blocks=3).to(device)
    
    # 如果已有模型则加载
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Resuming from existing checkpoint.")

    num_epochs = num_epochs
    lr = 0.001
    weights = [1.0, 10.0] # Policy vs Value loss weight

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        'train_steps': [], 'train_policy': [], 'train_value': [], 
        'val_epochs': [], 'val_policy': [], 'val_value': []
    }

    global_step = 0
    log_interval = 10

    for epoch in range(num_epochs):
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
            if global_step % log_interval == 0:
                history['train_steps'].append(global_step)
                history['train_policy'].append(policy_loss.item())
                history['train_value'].append(value_loss.item())

        scheduler.step()

        # 验证
        model.eval()
        total_val_policy = 0.0
        total_val_value = 0.0
        with torch.no_grad():
            for states, policies, values in val_dataloader:
                states, policies, values = states.to(device), policies.to(device), values.to(device)
                p_logits, v_preds = model(states)
                
                total_val_policy += criterion_policy(F.log_softmax(p_logits, dim=1), policies).item()
                total_val_value += criterion_value(v_preds.squeeze(), values.squeeze()).item()

        avg_val_p = total_val_policy / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_val_v = total_val_value / len(val_dataloader) if len(val_dataloader) > 0 else 0
        
        history['val_epochs'].append(global_step)
        history['val_policy'].append(avg_val_p)
        history['val_value'].append(avg_val_v)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | Val P-Loss: {avg_val_p:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存与绘图
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 绘图逻辑
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(history['train_steps'], history['train_policy'], alpha=0.3, color='blue')
    ax1.plot(history['val_epochs'], history['val_policy'], 'ro-', label='Val Policy Loss')
    ax1.set_title('Policy Loss (KL)')
    ax1.legend()

    ax2.plot(history['train_steps'], history['train_value'], alpha=0.3, color='red')
    ax2.plot(history['val_epochs'], history['val_value'], 'bo-', label='Val Value Loss')
    ax2.set_title('Value Loss (MSE)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    # plt.show()

# if __name__ == "__main__":
#     train()