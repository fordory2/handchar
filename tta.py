"""TTA: 9 视图纯平移 softmax 平均.

灰度保证: 输入已是 [B, 1, H, W] 单通道, grid_sample 不改通道维度.
不含旋转: 避免 bdpq69MWNZun 等敏感字符旋转后变成另一类.
"""
import torch
import torch.nn.functional as functional


TTA_VIEWS = [
    # (dx_px, dy_px) — 仅平移, 单位为像素
    (0.0, 0.0),     # 原图
    (2.0, 0.0),     # 右 2
    (-2.0, 0.0),    # 左 2
    (0.0, 2.0),     # 下 2
    (0.0, -2.0),    # 上 2
    (2.0, 2.0),     # 右下
    (-2.0, 2.0),    # 左下
    (2.0, -2.0),    # 右上
    (-2.0, -2.0),   # 左上
]


def _translate(images, dx_px, dy_px):
    """纯平移仿射, 不改通道."""
    batch_size, _, height, width = images.shape
    device = images.device
    dx = dx_px / width * 2
    dy = dy_px / height * 2
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = dx
    theta[:, 1, 2] = dy
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(images, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)


def _main_logits(out):
    """HybridHandCharNet forward 返回 5 元组, 取主分类 logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def tta_predict(model, x, views=None):
    """对一个 batch 做 TTA, 返回 softmax 平均后的概率 [B, num_classes]."""
    model.eval()
    if views is None:
        views = TTA_VIEWS
    probs = None
    for dx, dy in views:
        x_view = _translate(x, dx, dy) if (dx or dy) else x
        logits = _main_logits(model(x_view))
        p = functional.softmax(logits, dim=-1)
        probs = p if probs is None else probs + p
    return probs / len(views)
