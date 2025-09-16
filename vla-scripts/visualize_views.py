import torch 
import matplotlib.pyplot as plt

def visualize_views(pixel_values):
    """
    pixel_values: (B, 12, H, W) 的张量
    """
    B, C, H, W = pixel_values.shape
    assert C == 12, f"Expecting 12 channels (dual view + dual backbone), got {C}"

    # 原始 RGB
    third_view = pixel_values[0, :3]     # 第三视角原始
    wrist_view = pixel_values[0, 6:9]    # 腕部视角原始

    # 拷贝 RGB
    third_copy = pixel_values[0, 3:6]    # 第三视角拷贝
    wrist_copy = pixel_values[0, 9:12]   # 腕部视角拷贝

    def normalize(img):
        img = img.permute(1, 2, 0).cpu().float().numpy()
        return (img - img.min()) / (img.max() - img.min() + 1e-6)

    third_img = normalize(third_view)
    wrist_img = normalize(wrist_view)
    third_copy_img = normalize(third_copy)
    wrist_copy_img = normalize(wrist_copy)

    # 可视化 4 张图
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(third_img);       axs[0].set_title("Third View (RGB)"); axs[0].axis("off")
    axs[1].imshow(third_copy_img);  axs[1].set_title("Third Copy (RGB)"); axs[1].axis("off")
    axs[2].imshow(wrist_img);       axs[2].set_title("Wrist View (RGB)"); axs[2].axis("off")
    axs[3].imshow(wrist_copy_img);  axs[3].set_title("Wrist Copy (RGB)"); axs[3].axis("off")
    plt.show()
