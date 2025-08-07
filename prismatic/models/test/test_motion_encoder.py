import torch
from motion_encoder import MotionEncoder

def test_motion_encoder():
    print("\n=== 测试 MotionEncoder ===")
    # 设置模型参数
    in_channels = 2  # 光流的x,y两个通道
    hidden_dim = 128
    token_dim = 256
    num_tokens = 8  # 每个flow生成的token数量
    
    # 初始化模型
    model = MotionEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        token_dim=token_dim,
        num_tokens=num_tokens
    )
    model.eval()  # 设置为评估模式
    
    # 测试不同的输入尺寸
    test_cases = [
        {"batch_size": 2, "height": 128, "width": 128},
        {"batch_size": 1, "height": 64, "width": 64},
        {"batch_size": 4, "height": 256, "width": 256},
    ]
    
    for i, case in enumerate(test_cases):
        B = case["batch_size"]
        H = case["height"]
        W = case["width"]
        
        # 生成随机输入（模拟motion_chunk）
        motion_chunk = torch.randn(B, 4, in_channels, H, W)  # (B, 4, 2, H, W)
        
        # 前向传播
        with torch.no_grad():  # 禁用梯度计算，加速测试
            motion_tokens = model(motion_chunk)
        
        # 验证输出形状
        expected_tokens = 4 * num_tokens  # T=4, 每个flow生成num_tokens个token
        expected_shape = (B, expected_tokens, token_dim)
        
        assert motion_tokens.shape == expected_shape, \
            f"测试用例 {i+1} 失败：" \
            f"输入形状={motion_chunk.shape}, " \
            f"期望输出形状={expected_shape}, " \
            f"实际输出形状={motion_tokens.shape}"
        
        print(f"✅ 测试用例 {i+1} 通过：输入{(B, 4, 2, H, W)} → 输出{expected_shape}")
    
    print("\n所有测试通过！MotionEncoder 功能正常")

if __name__ == "__main__":
    test_motion_encoder()

