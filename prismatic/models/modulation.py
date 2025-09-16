import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging
from tqdm import tqdm

# 设置日志
logger = logging.getLogger(__name__)

class MotionEncoder(nn.Module):
    """
    MotionEncoder（空间分patch -> frame-level pooling -> temporal transformer -> final query pooling）
    相比 naive T*P Transformer 更节省显存，优化梯度追踪安全性与代码可读性。
    """

    def __init__(
        self,
        in_channels=2,
        hidden_dim=128,
        token_dim=256,
        num_tokens=8,
        frame_tokens=2,          # 每帧先pool成几个 frame-level token
        num_heads=4,
        num_temporal_layers=4,
        dropout=0.1,
        num_groups=8             # ✅ GroupNorm 分组数
    ):
        super().__init__()
        self.token_dim = token_dim  
        self.frame_tokens = frame_tokens  

        # 1. 空间卷积主干：BN → GN，更稳健
        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.ReLU(),
        )

        self.patch_proj = nn.Linear(hidden_dim, token_dim)

        # 2. 帧级注意力池化
        self.frame_query = nn.Parameter(torch.randn(1, frame_tokens, token_dim))
        self.frame_attn = nn.MultiheadAttention(
            embed_dim=token_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )

        # 3. 时间维度 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=num_heads, 
            dim_feedforward=token_dim * 4,
            dropout=dropout, 
            activation="gelu", 
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)

        # 4. 最终注意力池化
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        self.final_attn = nn.MultiheadAttention(
            embed_dim=token_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入校验
        if x.ndim != 5:
            raise ValueError(f"输入x需为5D张量 (B, T, C, H, W)，当前为 {x.ndim}D")
        B, T, C, H, W = x.shape

        # 1. 空间特征提取
        x_flat = x.reshape(B * T, C, H, W)
        spatial_feat = self.spatial_backbone(x_flat)
        B_T, hidden_dim, Hf, Wf = spatial_feat.shape

        # 2. 转为patch序列并投影
        P = Hf * Wf
        patch_feat = spatial_feat.flatten(2).transpose(1, 2)  # (B*T, P, hidden_dim)
        patch_feat = self.patch_proj(patch_feat)              # (B*T, P, D)

        # 3. 帧级注意力池化
        frame_queries = self.frame_query.expand(B * T, -1, -1).clone()
        frame_tokens, _ = self.frame_attn(frame_queries, patch_feat, patch_feat)  # (B*T, F, D)

        # 4. 重塑帧级 token 为时间序列
        frame_tokens_seq = frame_tokens.reshape(B, T * self.frame_tokens, self.token_dim)

        # 5. 时间维度 Transformer
        temporal_feat = self.temporal_encoder(frame_tokens_seq)

        # 6. 最终注意力池化
        final_queries = self.query_tokens.expand(B, -1, -1).clone()
        motion_tokens, _ = self.final_attn(final_queries, temporal_feat, temporal_feat)  # (B, K, D)

        # 7. 归一化与 Dropout
        motion_tokens = self.norm(motion_tokens)
        motion_tokens = self.dropout(motion_tokens)

        return motion_tokens
    
class GatedFiLMModulator(nn.Module):
    """
    Gated FiLM: self-adaptive modulation of appearance features by motion features.
    output = (1 - alpha) * appearance + alpha * FiLM(appearance, motion)
    alpha is learned from motion features.
    """
    def __init__(self, token_dim: int, motion_dim: int, reduction: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.motion_dim = motion_dim
        hidden_dim = max(motion_dim // reduction, 1)

        # FiLM projection: maps motion -> scale and shift for appearance
        self.film_scale = nn.Linear(motion_dim, token_dim)
        self.film_shift = nn.Linear(motion_dim, token_dim)

        # Gate network: maps motion -> gating alpha ∈ [0,1]
        self.gate = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # LayerNorm
        self.input_norm = nn.LayerNorm(token_dim)
        self.output_norm = nn.LayerNorm(token_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, appearance_token: torch.Tensor, motion_token: torch.Tensor):
        """
        Args:
            appearance_token: (B, N, token_dim)
            motion_token: (B, M, motion_dim)
        Returns:
            modulated_token: (B, N, token_dim)
        """
        B, N, D = appearance_token.shape
        # 输入归一化
        appearance_token = self.input_norm(appearance_token)

        # 平均池化 motion token -> (B, motion_dim)
        motion_feat = motion_token.mean(dim=1)

        # FiLM: scale & shift
        scale = self.film_scale(motion_feat).unsqueeze(1)  # (B,1,D)
        shift = self.film_shift(motion_feat).unsqueeze(1)  # (B,1,D)
        film_out = appearance_token * (1 + scale) + shift

        # Gate alpha
        alpha = self.gate(motion_feat).view(B, 1, 1)  # (B,1,1)

        # Gated combination
        modulated = (1 - alpha) * appearance_token + alpha * film_out

        # 输出归一化
        modulated = self.output_norm(modulated)
        return modulated

class ModulatedVisionEncoder(nn.Module):
    def __init__(
        self,
        vision_backbone: nn.Module,
        image_size: int = 224,
        motion_token_dim: int = 256,
        llm_dim: int = 4096,
        num_images_in_input: int = 1,   # ✅ 新增参数
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.motion_encoder = MotionEncoder(token_dim=motion_token_dim)
        self._dtype_printed = False

        # 优先使用外部传入的num_images_in_input，而非动态判断
        self.num_images_in_input = num_images_in_input
        # 同步视觉主干的输入图像数配置（关键！）
        if hasattr(vision_backbone, "set_num_images_in_input"):
            vision_backbone.set_num_images_in_input(self.num_images_in_input)
        
        self.modulate_dim = self.vision_backbone.embed_dim
        self.image_size = image_size

        # 初始化 FiLM 调制器
        self.modulator = GatedFiLMModulator(motion_dim=motion_token_dim, token_dim=self.modulate_dim)

    def forward(
        self,
        image_tensor: Optional[torch.Tensor] = None,
        flow_tensor: Optional[torch.Tensor] = None,
        wrist_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播处理视觉和运动信息的调制
        
        Args:
            image_tensor: 第三视角图像张量 (B, C, H, W)
            flow_tensor: 光流张量 (B, T, C, H, W)
            wrist_tensor: 腕部相机图像张量 (B, C, H, W)
            
        Returns:
            modulated_token: 调制后的视觉token (B, N, D)
            
        Raises:
            ValueError: 当必需的输入缺失或形状不匹配时
        """

         # 输入验证
        self._validate_inputs(image_tensor, flow_tensor, wrist_tensor)
        
        # 设备和数据类型处理
        device, dtype = self._get_device_dtype()
        image_tensor, flow_tensor, wrist_tensor = self._prepare_tensors(
            image_tensor, flow_tensor, wrist_tensor, device, dtype
        )
        # 构建像素值张量
        pixel_values = self._build_pixel_values(image_tensor, wrist_tensor)
        #print("Built pixel_values shape:", pixel_values.shape)  # 应输出 (B, 12, H, W)
        # 视觉编码
        vision_output = self.vision_backbone(pixel_values)
        appearance_token = self._extract_appearance_tokens(vision_output)
        # 处理维度重塑 - 将4D特征图转换为token序列
        appearance_token = self._reshape_to_tokens(appearance_token)

        # 根据输入图像数量选择处理策略
        if self.num_images_in_input == 2:
            modulated_token = self._process_dual_image_tokens(appearance_token, flow_tensor)
        else:
            modulated_token = self._process_single_image_tokens(appearance_token, flow_tensor)

        # 记录调试信息
        if not self._dtype_printed:
            logger.debug(f"Vision backbone dtype: {dtype}")
            self._dtype_printed = True

        return modulated_token

    def _reshape_to_tokens(self, appearance_token: torch.Tensor) -> torch.Tensor:
        """将4D特征图重塑为token序列"""
        if appearance_token.ndim == 4:
            B, C, H, W = appearance_token.shape
            appearance_token = appearance_token.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif appearance_token.ndim != 3:
            raise ValueError(f"期望3D或4D的appearance_token，但得到{appearance_token.ndim}D")
        
        return appearance_token

    def _validate_inputs(
        self,
        image_tensor: Optional[torch.Tensor],
        flow_tensor: Optional[torch.Tensor],
        wrist_tensor: Optional[torch.Tensor]
    ) -> None:
        """验证输入参数的有效性"""
        if flow_tensor is None:
            raise ValueError("flow_tensor不能为None，运动调制需要光流信息")
        
        if image_tensor is None:
            raise ValueError("image_tensor不能为None，需要第三视角图像")
            
        if self.num_images_in_input == 2 and wrist_tensor is None:
            raise ValueError("当num_images_in_input=2时，wrist_tensor不能为None")
            
        # 检查张量维度
        if image_tensor.ndim != 4:
            raise ValueError(f"image_tensor应为4维 (B,C,H,W)，但得到{image_tensor.ndim}维")
        if flow_tensor.ndim != 5:
            raise ValueError(f"flow_tensor应为5维 (B,T,C,H,W)，但得到{flow_tensor.ndim}维")
        if wrist_tensor is not None and wrist_tensor.ndim != 4:
            raise ValueError(f"wrist_tensor应为4维 (B,C,H,W)，但得到{wrist_tensor.ndim}维")

    def _get_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """获取模型的设备和数据类型"""
        device = next(self.parameters()).device
        dtype = next(self.vision_backbone.parameters()).dtype
        return device, dtype

    def _prepare_tensors(
        self,
        image_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        wrist_tensor: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """将张量移动到正确的设备和数据类型"""
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        flow_tensor = flow_tensor.to(device=device, dtype=dtype)
        if wrist_tensor is not None:
            wrist_tensor = wrist_tensor.to(device=device, dtype=dtype)
        return image_tensor, flow_tensor, wrist_tensor

    def _build_pixel_values(
        self,
        image_tensor: torch.Tensor,
        wrist_tensor: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """构建用于视觉主干的像素值张量，兼容 fused backbone 双图像输入。

    Args:
        image_tensor: 第三视角图像 (B,3,H,W)
        wrist_tensor: 腕部相机图像 (B,3,H,W) 或 None
    Returns:
        pixel_values: (B, num_images*6, H, W)"""
        if hasattr(self.vision_backbone, 'use_fused_vision_backbone') and self.vision_backbone.use_fused_vision_backbone:
            # 单视角6通道已在finetune.py中处理，这里直接拼接
            if self.num_images_in_input == 2 and wrist_tensor is not None:
                # 双视角：6（第三视角）+6（腕部）=12通道
                pixel_values = torch.cat([image_tensor, wrist_tensor], dim=1)  # (B, 12, H, W)
            else:
                # 单视角：6通道
                pixel_values = image_tensor
        else:
            # 非融合主干模式（不推荐双视角，此处仅作兼容）
            pixel_values = image_tensor
            if self.num_images_in_input == 2 and wrist_tensor is not None:
                pixel_values = torch.cat([pixel_values, wrist_tensor], dim=1)  # (B, 6, H, W)

        return pixel_values


    def _extract_appearance_tokens(self, vision_output: Any) -> torch.Tensor:
        """从视觉输出中提取外观token"""
        if isinstance(vision_output, dict):
            appearance_token = vision_output.get("patch_tokens")
            if appearance_token is None:
                raise KeyError("视觉输出字典中缺少'patch_tokens'键")
        else:
            appearance_token = vision_output
            
        return appearance_token

    def _process_dual_image_tokens(
        self,
        appearance_token: torch.Tensor,
        flow_tensor: torch.Tensor
    ) -> torch.Tensor:
        """处理双图像输入的token分离和调制"""
        #print("===== Dual image tokens =====")
        total_patches = appearance_token.shape[1]
        single_image_patches = total_patches // 2
        
        if total_patches % 2 != 0:
            raise ValueError(f"期望双图像输入的patch数量为偶数，但得到 {total_patches}")
        
        # 分离第三视角和腕部相机的token
        third_person_tokens = appearance_token[:, :single_image_patches, :]
        wrist_tokens = appearance_token[:, single_image_patches:, :]
        
        # 只对第三视角图像的token进行运动调制
        motion_token = self.motion_encoder(flow_tensor)
        #before = third_person_tokens.clone()   # 保存调制前的token
        modulated_third_person_tokens = self.modulator(third_person_tokens, motion_token)

        #with torch.no_grad():
        #    diff = (modulated_third_person_tokens - before).abs().mean().item()
        #  max_diff = (modulated_third_person_tokens - before).abs().max().item()
        #    print(f"[VisionEncoder Debug] FiLM changed tokens: mean={diff:.6f}, max={max_diff:.6f}")
        
        # 将调制后的第三视角token与原始腕部相机token拼接
        modulated_token = torch.cat([modulated_third_person_tokens, wrist_tokens], dim=1)

        return modulated_token

    def _process_single_image_tokens(
        self,
        appearance_token: torch.Tensor,
        flow_tensor: torch.Tensor
    ) -> torch.Tensor:
        """处理单图像输入的token调制"""
        #print("===== Single image tokens =====")
        motion_token = self.motion_encoder(flow_tensor)
        modulated_token = self.modulator(appearance_token, motion_token)
        return modulated_token

    def get_num_patches(self) -> int:
        """获取patch数量"""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """获取输入图像数量"""
        return self.num_images_in_input

    def enable_gradient_checkpointing(self) -> None:
        """启用梯度检查点以节省内存"""
        if hasattr(self.vision_backbone, 'enable_gradient_checkpointing'):
            self.vision_backbone.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """禁用梯度检查点"""
        if hasattr(self.vision_backbone, 'disable_gradient_checkpointing'):
            self.vision_backbone.disable_gradient_checkpointing()

    def get_memory_usage(self) -> Dict[str, float]:
        """获取模型的内存使用情况（MB）"""
        def get_model_memory(model):
            return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

        return {
            'vision_backbone_mb': get_model_memory(self.vision_backbone),
            'motion_encoder_mb': get_model_memory(self.motion_encoder),
            'modulator_mb': get_model_memory(self.modulator),
            'total_mb': get_model_memory(self)
        }

    def set_training_mode(self, mode: bool = True) -> 'ModulatedVisionEncoder':
        """设置训练模式并返回自身以支持链式调用"""
        super().train(mode)
        return self

    def freeze_vision_backbone(self) -> None:
        """冻结视觉主干的参数"""
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        logger.info("Vision backbone parameters frozen")

    def unfreeze_vision_backbone(self) -> None:
        """解冻视觉主干的参数"""
        for param in self.vision_backbone.parameters():
            param.requires_grad = True
        logger.info("Vision backbone parameters unfrozen")

    def get_trainable_parameters(self) -> Dict[str, int]:
        """获取可训练参数的统计信息"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'vision_backbone': count_parameters(self.vision_backbone),
            'motion_encoder': count_parameters(self.motion_encoder),
            'modulator': count_parameters(self.modulator),
            'total': count_parameters(self)
        }

