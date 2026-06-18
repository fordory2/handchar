"""TTA: 8 个增强版本 softmax 平均."""
import torch
import torch.nn.functional as functional


def _affine(images, angle_deg=0.0, dx_px=0.0, dy_px=0.0):
    """仿射变换: 角度 + 像素平移. images: [B, 1, H, W]."""
    batch_size, _, height, width = images.shape
    device = images.device
    angle = angle_deg * 3.14159 / 180
    cos_a, sin_a = torch.cos(torch.tensor(angle, device=device)), torch.sin(torch.tensor(angle, device=device))
    dx = dx_px / width * 2
    dy = dy_px / height * 2
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = dx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = dy
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(images, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)


TTA_VIEWS = [
    # 纯平移 TTA: 不含旋转, 避免 bdpq69MWNZun 等敏感类被旋出错类.
    (0.0, 0.0, 0.0),     # 原图
    (0.0, 2.0, 0.0),     # 右 2px
    (0.0, -2.0, 0.0),    # 左 2px
    (0.0, 0.0, 2.0),     # 下 2px
    (0.0, 0.0, -2.0),    # 上 2px
    (0.0, 2.0, 2.0),     # 右下
    (0.0, -2.0, 2.0),    # 左下
    (0.0, 2.0, -2.0),    # 右上
    (0.0, -2.0, -2.0),   # 左上
]


@torch.no_grad()
def tta_predict(model, x, views=None):
    """返回 softmax 平均后的概率 [B, num_classes]."""
    model.eval()
    if views is None:
        views = TTA_VIEWS
    probs = None
    for ang, dx, dy in views:
        x_view = _affine(x, ang, dx, dy) if (ang or dx or dy) else x
        out = model(x_view)
        logits = out[0] if isinstance(out, tuple) else out
        p = functional.softmax(logits, dim=-1)
        probs = p if probs is None else probs + p
    return probs / len(views)
