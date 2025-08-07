# === 文件: motion_encoder.py ===
# 功能：将 motion chunk (B, 4, 2, H, W) 编码成 motion token (B, N, D)

import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=128, token_dim=256, num_tokens=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, token_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((num_tokens, 1))  # 改成 (N, 1)

    def forward(self, motion_chunk: torch.Tensor):
        B, T, C, H, W = motion_chunk.shape
        motion_chunk = motion_chunk.view(B * T, C, H, W)
        features = self.conv(motion_chunk)       # (B*T, D, h, w)
        pooled = self.pool(features)             # (B*T, D, N, 1)
        pooled = pooled.squeeze(-1).transpose(1, 2)  # (B*T, N, D)
        motion_tokens = pooled.reshape(B, T * pooled.shape[1], -1)  # (B, N=T*N_pooled, D)
        return motion_tokens
