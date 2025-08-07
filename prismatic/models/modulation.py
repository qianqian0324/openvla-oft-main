import torch
import torch.nn as nn
from prismatic.models.motion_encoder import MotionEncoder
from prismatic.models.film_layer import FiLMModulator

class ModulatedVisionEncoder(nn.Module):
    def __init__(
        self,
        vision_backbone: nn.Module,
        image_size: int = 224,
        motion_token_dim: int = 256,
        modulate_dim: int = 1024,  # 默认 dinosiglip-vit-so-224px 的 token dim
        raw_vision_backbone: nn.Module = None,
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

        # 获取设备和精度
        backbone_for_inference = raw_vision_backbone or vision_backbone
        sample_device = next(backbone_for_inference.parameters()).device
        orig_dtype = next(backbone_for_inference.parameters()).dtype

        # 临时设置 float32（防止 meta device 报错）
        self.vision_backbone = self.vision_backbone.float()
        self.motion_encoder = self.motion_encoder.float()

        # 保存其他参数
        self.modulate_dim = modulate_dim
        self.image_size = image_size

        # 初始化 FiLM 调制器
        self.modulator = FiLMModulator(motion_dim=motion_token_dim, token_dim=self.modulate_dim)

        # 恢复原始精度
        self.vision_backbone = self.vision_backbone.to(orig_dtype)
        self.motion_encoder = self.motion_encoder.to(orig_dtype)

    def forward(self, image_tensor: torch.Tensor = None, flow_tensor: torch.Tensor = None, wrist_tensor: torch.Tensor = None):
        print("[ModulatedVisionEncoder] forward() 被调用")
        #print(f"image_tensor shape: {image_tensor.shape if image_tensor is not None else 'None'}")
        #print(f"flow_tensor shape: {flow_tensor.shape if flow_tensor is not None else 'None'}")
        #print(f"wrist_tensor shape: {wrist_tensor.shape if wrist_tensor is not None else 'None'}")

    # === 强制检查 flow_tensor 是否传入 ===
        if flow_tensor is None:
            raise ValueError("[ModulatedVisionEncoder] flow_tensor is None —— 请确保调用 forward() 时传入光流张量。")

        device = next(self.parameters()).device
        dtype = next(self.vision_backbone.parameters()).dtype

        image_tensor = image_tensor.to(device=device, dtype=dtype)
        flow_tensor = flow_tensor.to(device=device, dtype=dtype)
        if wrist_tensor is not None:
            wrist_tensor = wrist_tensor.to(device=device, dtype=dtype)

    # 构造 pixel_values 输入视觉主干
        if self.num_images_in_input == 2:
            if wrist_tensor is not None:
                pixel_values = torch.cat([image_tensor, wrist_tensor], dim=1)
            else:
                pixel_values = torch.cat([image_tensor, image_tensor], dim=1)
        else:
            pixel_values = image_tensor

        if not self._dtype_printed:
            #print("[DEBUG] vision_backbone dtype:", dtype)
            self._dtype_printed = True

    # === 编码 appearance 图像 ===
        vision_output = self.vision_backbone(pixel_values)
        appearance_token = vision_output.get("patch_tokens") if isinstance(vision_output, dict) else vision_output
        if appearance_token.ndim == 4:
            B, C, H, W = appearance_token.shape
            appearance_token = appearance_token.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # === 编码 motion 光流并调制 ===
        motion_token = self.motion_encoder(flow_tensor)
        modulated_token = self.modulator(appearance_token, motion_token)

        return modulated_token

    def get_num_patches(self):
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self):
        return self.num_images_in_input

