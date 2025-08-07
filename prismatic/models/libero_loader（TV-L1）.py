import numpy as np
import cv2
from typing import List, Callable, Tuple, Dict
from datasets import load_dataset
from torch.utils.data import Dataset

# ====== 光流计算 ======
def compute_optical_flow_tvl1(prev_img, next_img):
    """
    使用 TV-L1 算法计算光流，输入为灰度图
    """
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(prev_img, next_img, None)  # (H, W, 2)
    return flow

# ====== 合法中心帧索引（滑动窗口） ======
def get_valid_center_indices(num_frames: int, stride: int = 3) -> List[int]:
    """
    合法中心帧索引 t ∈ [3, num_frames - 2]，以提取5帧 ⇒ 4光流
    """
    return list(range(2, num_frames - 2, stride))

# ====== 核心函数：提取 motion chunk + 当前帧（保留原函数名） ======
def extract_motion_and_frame(
    episode: List[Dict],
    center_idx: int,
    resize_resolution: Tuple[int, int] = (224, 224) #(224,224)
):
    """
    提取 motion_chunk (T=4, 2, H, W) + appearance_frame (H, W, 3)
    """
    if center_idx < 3 or center_idx + 1 >= len(episode):
        raise ValueError(f"非法中心帧索引 center_idx={center_idx}，需满足 [3, len-2]")

    # === 读取原始图像序列 ===
    rgb_seq = []
    for step in episode:
        img = step["image"]

        # 如果图像维度为 (1, H, W, 3)，去除掉第一个维度
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)  # 去掉维度为1的轴，变为 (H, W, 3)

        # 如果是灰度图（单通道），强制转换为 RGB
        if img.ndim == 2:  # shape: (H, W)
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:  # shape: (H, W, 1)
            img = np.concatenate([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass  # RGB，正常
        else:
            raise ValueError(f"非法图像 shape={img.shape}，不能转为 RGB")

        rgb_seq.append(img)

    # === 灰度图用于光流 ===
    frames_gray = [cv2.cvtColor(rgb_seq[i], cv2.COLOR_RGB2GRAY) for i in range(center_idx - 2, center_idx + 3)]

    # === 光流序列 ===
    motion_list = []
    for i in range(4):
        flow = compute_optical_flow_tvl1(frames_gray[i], frames_gray[i + 1])  # (H, W, 2)
        motion_list.append(flow)
    motion_chunk = np.stack(motion_list, axis=0)  # (4, H, W, 2)
    motion_chunk = np.transpose(motion_chunk, (0, 3, 1, 2))  # (4, 2, H, W)

    # === 当前帧 ===
    appearance_frame = rgb_seq[center_idx]
    appearance_frame = cv2.resize(appearance_frame, resize_resolution)

    return motion_chunk, appearance_frame

