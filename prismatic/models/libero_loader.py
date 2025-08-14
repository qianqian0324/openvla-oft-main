import numpy as np
import cv2
import argparse
from typing import List, Tuple, Dict
from datasets import load_dataset
from torch.utils.data import Dataset

# ====== RAFT optical flow =====================
import torch
from prismatic.models.raftt.core.raft import RAFT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_raft_model(model_path, small=True):
    from prismatic.models.raftt.core.raft import RAFT
    import argparse

    args = argparse.Namespace(small=small, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)

    raw_state = torch.load(model_path, map_location=DEVICE)

    # 去掉 "module." 前缀
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in raw_state.items() }

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model
# weight路径
raft_model = load_raft_model("prismatic/models/raftt/model/raft-small.pth", small=True)

@torch.no_grad()
def compute_optical_flow_raft(prev_img, next_img):
    """
    prev_img / next_img: numpy uint8 RGB (H, W, 3)
    return: (H, W, 2)
    """
    t1 = torch.from_numpy(prev_img).permute(2, 0, 1).float()[None] / 255.0
    t2 = torch.from_numpy(next_img).permute(2, 0, 1).float()[None] / 255.0
    t1, t2 = t1.to(DEVICE), t2.to(DEVICE)
    flow_low, flow_up = raft_model(t1, t2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()


# ====== 合法中心帧索引（滑动窗口） ======
def get_valid_center_indices(num_frames: int, stride: int = 3) -> List[int]:
    """
    合法中心帧索引 t ∈ [4, num_frames - 1]，以提取前4帧+当前帧 ⇒ 4光流
    """
    return list(range(4, num_frames, stride))

# ====== 核心函数：提取 motion chunk + 当前帧（已改为RAFT） ======
def extract_motion_and_frame(
    episode: List[Dict],
    center_idx: int,
    resize_resolution: Tuple[int, int] = (224, 224)
):
    """
    只用前4帧 + 当前帧，提取 motion_chunk (T=4, 2, H, W) + appearance_frame (H, W, 3)
    """
    if center_idx < 4 or center_idx >= len(episode):
        raise ValueError(f"非法中心帧索引 center_idx={center_idx}，需满足 [4, len-1]")

    rgb_seq = []
    for img in episode:
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        rgb_seq.append(img)

    # 只取前4帧 + 当前帧
    frames = [rgb_seq[i] for i in range(center_idx - 4, center_idx + 1)]
    motion_list = []
    for i in range(4):
        flow = compute_optical_flow_raft(frames[i], frames[i+1])   # (H,W,2)
        motion_list.append(flow)
    motion_chunk = np.stack(motion_list, axis=0)  # (4,H,W,2)
    motion_chunk = np.transpose(motion_chunk, (0,3,1,2))  # (4,2,H,W)

    appearance_frame = rgb_seq[center_idx]
    appearance_frame = cv2.resize(appearance_frame, resize_resolution)

    return motion_chunk, appearance_frame

