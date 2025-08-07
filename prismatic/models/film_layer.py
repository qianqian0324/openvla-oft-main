# === 文件: film_layer.py ===
# 功能：FiLM调制模块，使用 motion token 对 appearance token 进行调制

import torch
import torch.nn as nn

class FiLMModulator(nn.Module):
    def __init__(self, token_dim: int, motion_dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = motion_dim // reduction

        # 使用 motion_token.mean(dim=1) 作为 pooled 向量进行调制
        self.gamma_fc = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.ln = nn.LayerNorm(token_dim)

    def forward(self, appearance_token: torch.Tensor, motion_token: torch.Tensor):
        """
        Args:
            appearance_token: (B, N, D)
            motion_token: (B, M, D_m)
        Returns:
            modulated_token: (B, N, D)
        """
        pooled = motion_token.mean(dim=1)  # (B, D_m)
        gamma = self.gamma_fc(pooled).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_fc(pooled).unsqueeze(1)    # (B, 1, D)
        
        #print("[DEBUG] motion_token dim:", pooled.shape[-1]) ##
        #print("[DEBUG] appearance_token dim:", appearance_token.shape[-1]) ##
        
        modulated = gamma * appearance_token + beta
        return self.ln(modulated)  # 可以在外部 print(gamma, beta) if needed
