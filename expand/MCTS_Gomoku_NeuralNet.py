# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_channels=4, num_res_blocks=6, board_size=8): # 稍微加深网络
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        
        # 1. 特征提取层
        self.start_block = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1), # 提升宽度到 128
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