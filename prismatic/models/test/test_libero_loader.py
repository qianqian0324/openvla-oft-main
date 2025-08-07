import os
import h5py
import numpy as np
import cv2
from libero_loader import (
    load_rgb_sequence,
    compute_optical_flow_tvl1,
    build_motion_chunk,
    extract_motion_and_frame
)

# --------------------------
# 1. 准备测试数据（生成一个小型HDF5文件）
# --------------------------
def create_test_hdf5(file_path: str, num_frames: int = 10, img_shape: tuple = (128, 128, 3)):
    """生成包含模拟RGB序列的测试HDF5文件"""
    with h5py.File(file_path, 'w') as f:
        # 创建观测组和图像数据集（模拟LIBERO格式）
        obs_group = f.create_group('observations')
        # 生成随机RGB图像（第5帧后添加一个明显运动，方便光流验证）
        images = []
        for i in range(num_frames):
            if i < 5:
                img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)
            else:
                # 第5帧后向右移动10像素（制造明显运动）
                img = np.roll(np.random.randint(0, 256, size=img_shape, dtype=np.uint8), 10, axis=1)
            images.append(img)
        obs_group.create_dataset('image', data=np.stack(images))
    print(f"生成测试HDF5文件: {file_path}")

# --------------------------
# 2. 单元测试
# --------------------------
def test_load_rgb_sequence(h5_path: str):
    print("\n=== 测试 load_rgb_sequence ===")
    rgb_frames = load_rgb_sequence(h5_path)
    # 检查输出类型和长度
    assert isinstance(rgb_frames, list), "输出应为列表"
    assert len(rgb_frames) > 0, "未读取到图像"
    # 检查单帧形状和数据类型
    first_frame = rgb_frames[0]
    assert first_frame.shape == (128, 128, 3), f"图像形状错误，实际: {first_frame.shape}"
    assert first_frame.dtype == np.uint8, f"数据类型错误，实际: {first_frame.dtype}"
    print("✅ load_rgb_sequence 测试通过")

def test_compute_optical_flow_tvl1(h5_path: str):
    print("\n=== 测试 compute_optical_flow_tvl1 ===")
    rgb_frames = load_rgb_sequence(h5_path)
    # 取第4和第5帧（有明显运动）
    prev_img = cv2.cvtColor(rgb_frames[4], cv2.COLOR_RGB2GRAY)
    next_img = cv2.cvtColor(rgb_frames[5], cv2.COLOR_RGB2GRAY)
    flow = compute_optical_flow_tvl1(prev_img, next_img)
    # 检查光流形状
    assert flow.shape == (128, 128, 2), f"光流形状错误，实际: {flow.shape}"
    # 检查运动区域是否有非零值（因第5帧向右移动，x方向应有正光流）
    x_flow = flow[..., 0]
    assert np.mean(x_flow) > 0, f"光流值不合理（运动不明显），平均x光流: {np.mean(x_flow)}"
    print("✅ compute_optical_flow_tvl1 测试通过")

def test_build_motion_chunk(h5_path: str):
    print("\n=== 测试 build_motion_chunk ===")
    rgb_frames = load_rgb_sequence(h5_path)
    # 测试正常情况（center_idx=5，远离边界）
    motion_chunk = build_motion_chunk(rgb_frames, center_idx=5)
    assert motion_chunk.shape == (4, 2, 128, 128), f"motion_chunk形状错误，实际: {motion_chunk.shape}"
    # 测试边界情况（center_idx=0，前3个flow应全为0）
    motion_chunk_boundary = build_motion_chunk(rgb_frames, center_idx=0)
    assert np.all(motion_chunk_boundary[0] == 0), "边界情况第1个flow应为0"
    assert np.all(motion_chunk_boundary[1] == 0), "边界情况第2个flow应为0"
    print("✅ build_motion_chunk 测试通过")

def test_extract_motion_and_frame(h5_path: str):
    print("\n=== 测试 extract_motion_and_frame ===")
    motion_chunk, center_frame = extract_motion_and_frame(h5_path, center_idx=5)
    # 检查输出形状
    assert motion_chunk.shape == (4, 2, 128, 128), f"motion_chunk形状错误"
    assert center_frame.shape == (128, 128, 3), f"center_frame形状错误"
    # 检查数据类型
    assert motion_chunk.dtype == np.float32, f"motion_chunk类型错误"
    assert center_frame.dtype == np.uint8, f"center_frame类型错误"
    print("✅ extract_motion_and_frame 测试通过")

# --------------------------
# 3. 运行测试
# --------------------------
if __name__ == "__main__":
    # 生成测试用HDF5文件（自动清理）
    test_h5_path = "/home/QWJ/openvla-oft/openvla-oft-main/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    create_test_hdf5(test_h5_path)
    
    # 执行所有测试
    test_load_rgb_sequence(test_h5_path)
    test_compute_optical_flow_tvl1(test_h5_path)
    test_build_motion_chunk(test_h5_path)
    test_extract_motion_and_frame(test_h5_path)
    
    # 清理测试文件
    os.remove(test_h5_path)
    print("\n所有测试通过！libero_loader.py 功能正常")

