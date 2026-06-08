"""自监督预训练: 遮罩重建 → 学通用笔画表示"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, random, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, ECA as ECA_ATTN
from data_utils import load_split_data, CharDataset
from project_constants import DEVICE, BATCH_SIZE

MASK_RATIO = 0.3   # 遮 30% 的 patch
PATCH_SIZE = 4     # patch 大小
PRETRAIN_EPOCHS = 100


class MaskedEncoder(make_hand_char(ECA_ATTN)):
    """编码器: 去掉分类头, 输出特征图"""
    def forward(self, x):
        x = self.direction(self.stem(x))
        x = self.shape2(self.shape1(x))
        return self.detail(x)  # [B, 160, 4, 3]


class LightDecoder(nn.Module):
    """轻量解码器: 从特征图重建原图"""
    def __init__(self):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(160, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.up(x)  # [B, 1, 64, 48]


def random_mask(images, ratio=0.3, patch_size=4):
    """按 patch 随机遮罩"""
    B, C, H, W = images.shape
    H_p, W_p = H // patch_size, W // patch_size
    n_patches = H_p * W_p
    n_mask = int(n_patches * ratio)
    # 构建 mask (1=保留, 0=遮住)
    mask_patches = torch.ones(B, H_p, W_p, device=images.device)
    for b in range(B):
        indices = torch.randperm(n_patches)[:n_mask]
        mask_patches[b].view(-1)[indices] = 0
    # 上采样到原尺寸
    mask = mask_patches.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    masked = images * mask.unsqueeze(1)
    return masked, mask


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    # 用全部 55 张/类做预训练 (不涉及标签)
    train_data, _, test_data, _, _ = load_split_data()
    all_images = [(f, l) for f, l in train_data + test_data]
    ds = CharDataset(all_images, train=False)  # 不增强, 只要原图
    loader = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=0)

    encoder = MaskedEncoder().to(DEVICE)
    decoder = LightDecoder().to(DEVICE)
    crit = nn.MSELoss()
    opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

    for ep in range(PRETRAIN_EPOCHS):
        encoder.train(); decoder.train()
        total_loss = 0
        for images, _ in loader:
            images = images.to(DEVICE)
            masked_imgs, mask = random_mask(images, MASK_RATIO, PATCH_SIZE)
            opt.zero_grad()
            features = encoder(masked_imgs)
            reconstructed = decoder(features)
            # 只计算遮罩区域的损失
            loss = crit(reconstructed * (1 - mask.unsqueeze(1)),
                       images * (1 - mask.unsqueeze(1)))
            loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        if (ep + 1) % 20 == 0:
            print("Ep%3d: loss=%.4f" % (ep + 1, total_loss / len(loader)))

    # 保存预训练编码器
    torch.save(encoder.state_dict(), "output/pretrain_encoder.pth")
    print("\nSaved: output/pretrain_encoder.pth")
    print("Next: python train_with_pretrain.py")
