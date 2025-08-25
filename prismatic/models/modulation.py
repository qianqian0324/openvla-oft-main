import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging

# 设置日志
logger = logging.getLogger(__name__)

class MotionEncoder(nn.Module):
    """运动编码器：将光流序列编码为运动token"""
    
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 128,
        token_dim: int = 256,
        num_tokens: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        
        # 卷积特征提取器
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(hidden_dim, token_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, token_dim), num_channels=token_dim),
            nn.ReLU(),
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((num_tokens, 1))
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, motion_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_chunk: 运动序列 (B, T, C, H, W)
        Returns:
            motion_tokens: 运动token (B, T*num_tokens, token_dim)
        """
        if motion_chunk.ndim != 5:
            raise ValueError(f"期望5D输入 (B,T,C,H,W)，但得到{motion_chunk.ndim}D")
            
        B, T, C, H, W = motion_chunk.shape
        
        if C != self.in_channels:
            raise ValueError(f"期望{self.in_channels}个输入通道，但得到{C}个")
        
        # 重塑为批次处理
        motion_reshaped = motion_chunk.view(B * T, C, H, W)
        
        # 卷积特征提取
        features = self.conv_layers(motion_reshaped)  # (B*T, token_dim, h, w)
        
        # 自适应池化
        pooled = self.adaptive_pool(features)  # (B*T, token_dim, num_tokens, 1)
        pooled = pooled.squeeze(-1).transpose(1, 2)  # (B*T, num_tokens, token_dim)
        
        # 重塑回原始批次结构
        motion_tokens = pooled.reshape(B, T * self.num_tokens, self.token_dim)
        
        return motion_tokens


class FiLMModulator(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 调制器：使用运动信息调制视觉特征"""
    
    def __init__(
        self,
        token_dim: int,
        motion_dim: int,
        reduction: int = 4,
        dropout_rate: float = 0.1,
        use_attention_pooling: bool = False
    ):
        super().__init__()
        if motion_dim % reduction != 0:
            raise ValueError(f"motion_dim ({motion_dim}) 必须能被reduction ({reduction}) 整除")
            
        self.token_dim = token_dim
        self.motion_dim = motion_dim
        self.reduction = reduction
        self.use_attention_pooling = use_attention_pooling
        
        hidden_dim = motion_dim // reduction

        # Gamma (缩放) 分支
        self.gamma_network = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, token_dim),
            nn.Sigmoid()  # 确保gamma为正值
        )
        
        # Beta (偏移) 分支
        self.beta_network = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, token_dim),
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(token_dim)
        
        # 可选的注意力池化
        if use_attention_pooling:
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=motion_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.pooling_query = nn.Parameter(torch.randn(1, 1, motion_dim))
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _pool_motion_features(self, motion_token: torch.Tensor) -> torch.Tensor:
        """池化运动特征"""
        if self.use_attention_pooling:
            B = motion_token.shape[0]
            query = self.pooling_query.expand(B, -1, -1)  # (B, 1, motion_dim)
            pooled, _ = self.attention_pooling(
                query, motion_token, motion_token
            )  # (B, 1, motion_dim)
            pooled = pooled.squeeze(1)  # (B, motion_dim)
        else:
            # 简单平均池化
            pooled = motion_token.mean(dim=1)  # (B, motion_dim)
            
        return pooled

    def forward(
        self,
        appearance_token: torch.Tensor,
        motion_token: torch.Tensor
    ) -> torch.Tensor:
        """
        使用运动信息调制外观token
        
        Args:
            appearance_token: 外观token (B, N, token_dim)
            motion_token: 运动token (B, M, motion_dim)
            
        Returns:
            modulated_token: 调制后的token (B, N, token_dim)
        """
        # 输入验证
        if appearance_token.ndim != 3:
            raise ValueError(f"appearance_token应为3D，但得到{appearance_token.ndim}D")
        if motion_token.ndim != 3:
            raise ValueError(f"motion_token应为3D，但得到{motion_token.ndim}D")
        if appearance_token.shape[-1] != self.token_dim:
            raise ValueError(f"appearance_token最后一维应为{self.token_dim}，但得到{appearance_token.shape[-1]}")
        if motion_token.shape[-1] != self.motion_dim:
            raise ValueError(f"motion_token最后一维应为{self.motion_dim}，但得到{motion_token.shape[-1]}")
        
        # 池化运动特征
        pooled_motion = self._pool_motion_features(motion_token)  # (B, motion_dim)
        
        # 生成调制参数
        gamma = self.gamma_network(pooled_motion).unsqueeze(1)  # (B, 1, token_dim)
        beta = self.beta_network(pooled_motion).unsqueeze(1)    # (B, 1, token_dim)
        
        # FiLM调制: gamma * x + beta
        modulated = gamma * appearance_token + beta
        
        # 输出层归一化
        modulated = self.output_norm(modulated)
        
        return modulated


class ModulatedVisionEncoder(nn.Module):
    def __init__(
        self,
        vision_backbone: nn.Module,
        image_size: int = 224,
        motion_token_dim: int = 256,
        llm_dim: int = 4096,
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.motion_encoder = MotionEncoder(token_dim=motion_token_dim)
        self._dtype_printed = False

        self.num_images_in_input = (
            vision_backbone.get_num_images_in_input()
            if hasattr(vision_backbone, "get_num_images_in_input")
            else 1
        )

        
        self.modulate_dim = self.vision_backbone.embed_dim
        self.image_size = image_size

        # 初始化 FiLM 调制器
        self.modulator = FiLMModulator(motion_dim=motion_token_dim, token_dim=self.modulate_dim)

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

        # 动态判断单图/双图
        self.num_images_in_input = 2 if wrist_tensor is not None else 1
        #print(f"[DEBUG] num_images_in_input = {self.num_images_in_input}")
        # 输入验证
        self._validate_inputs(image_tensor, flow_tensor, wrist_tensor)
        
        # 设备和数据类型处理
        device, dtype = self._get_device_dtype()
        image_tensor, flow_tensor, wrist_tensor = self._prepare_tensors(
            image_tensor, flow_tensor, wrist_tensor, device, dtype
        )
        
        # 构建像素值张量
        pixel_values = self._build_pixel_values(image_tensor, wrist_tensor)
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
        """构建用于视觉主干的像素值张量"""
        #print(f"[DEBUG] image_tensor.shape={image_tensor.shape}, "
        #    f"wrist_tensor.shape={None if wrist_tensor is None else wrist_tensor.shape}")
        # 检查是否使用融合视觉主干
        if hasattr(self.vision_backbone, "use_fused_vision_backbone") and self.vision_backbone.use_fused_vision_backbone:
            pixel_values = image_tensor
            if wrist_tensor is not None and self.num_images_in_input == 2:
                pixel_values = torch.cat([pixel_values, wrist_tensor], dim=1)
        else:
            pixel_values = image_tensor if wrist_tensor is None or self.num_images_in_input == 1 else torch.cat([image_tensor, wrist_tensor], dim=1)

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
        total_patches = appearance_token.shape[1]
        single_image_patches = total_patches // 2
        
        if total_patches % 2 != 0:
            raise ValueError(f"期望双图像输入的patch数量为偶数，但得到 {total_patches}")
        
        # 分离第三视角和腕部相机的token
        third_person_tokens = appearance_token[:, :single_image_patches, :]
        wrist_tokens = appearance_token[:, single_image_patches:, :]
        
        # 只对第三视角图像的token进行运动调制
        motion_token = self.motion_encoder(flow_tensor)
        modulated_third_person_tokens = self.modulator(third_person_tokens, motion_token)
        
        # 将调制后的第三视角token与原始腕部相机token拼接
        modulated_token = torch.cat([modulated_third_person_tokens, wrist_tokens], dim=1)
        #print("appearance_token.shape=", appearance_token.shape)
        #print("third_person_tokens.shape=", third_person_tokens.shape)
        #print("wrist_tokens.shape=", wrist_tokens.shape)
        #print("modulated_third_person_tokens.shape=", modulated_third_person_tokens.shape)
        #print("modulated_token.shape=", modulated_token.shape)
        return modulated_token

    def _process_single_image_tokens(
        self,
        appearance_token: torch.Tensor,
        flow_tensor: torch.Tensor
    ) -> torch.Tensor:
        """处理单图像输入的token调制"""
        #print("[DEBUG] appearance_token BEFORE modulation:", appearance_token.shape)
        #print("[DEBUG] appearance_token SAMPLE BEFORE modulation:", appearance_token[:, :5, :])
        motion_token = self.motion_encoder(flow_tensor)
        modulated_token = self.modulator(appearance_token, motion_token)
        #print("[DEBUG] modulated_token AFTER modulation:", modulated_token.shape)
        #print("[DEBUG] modulated_token SAMPLE AFTER modulation:", modulated_token[:, :5, :])
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

