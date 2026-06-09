"""自监督预训练: 遮罩重建 → 学通用笔画表示 (使用 HybridHandCharNet 编码器)."""
import datetime
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import CharDataset, load_split_data
from models import HybridHandCharNet
from project_constants import (
    BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_CLASSES, PRETRAIN_EPOCHS,
)

MASK_RATIO = 0.3   # 遮 30% 的 patch
PATCH_SIZE = 4     # patch 大小


class LightDecoder(nn.Module):
    """轻量解码器: 从 HybridHandCharNet.encode() 的 decoder_out [B,64,16,12] 重建原图 [B,1,64,48]."""
    def __init__(self):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.up(x)


def random_mask(images, ratio=0.3, patch_size=4):
    """按 patch 随机遮罩, mask: 1=保留, 0=遮住."""
    batch_size, _, height, width = images.shape
    h_patches, w_patches = height // patch_size, width // patch_size
    n_patches = h_patches * w_patches
    n_mask = int(n_patches * ratio)
    mask_patches = torch.ones(batch_size, h_patches, w_patches, device=images.device)
    for sample_idx in range(batch_size):
        indices = torch.randperm(n_patches)[:n_mask]
        mask_patches[sample_idx].view(-1)[indices] = 0
    mask = mask_patches.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    masked = images * mask.unsqueeze(1)
    return masked, mask


def main():
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, _, _ = load_split_data()
    all_images = train_data + test_data  # 不涉及标签, 全量用
    dataset = CharDataset(all_images, train=False)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=0)

    # 用 HybridHandCharNet 当编码器 (只用 encode); RNN 关闭以加快 MAE
    encoder = HybridHandCharNet(num_classes=NUM_CLASSES, use_rnn=False).to(DEVICE)
    decoder = LightDecoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                      lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    print("MAE Pretrain | encoder=Hybrid(no_rnn) | epochs=%d" % PRETRAIN_EPOCHS)

    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        for images, _ in loader:
            images = images.to(DEVICE)
            masked_imgs, mask = random_mask(images, MASK_RATIO, PATCH_SIZE)
            optimizer.zero_grad()
            _, decoder_features = encoder.encode(masked_imgs)
            reconstructed = decoder(decoder_features)
            # 只计算遮罩区域 (1-mask) 的损失
            mask_loss_region = (1 - mask).unsqueeze(1)
            loss = criterion(reconstructed * mask_loss_region,
                             images * mask_loss_region)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Ep%3d/%d: loss=%.4f" %
                  (epoch + 1, PRETRAIN_EPOCHS, total_loss / len(loader)))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "output/pretrain_encoder_%s.pth" % timestamp
    torch.save(encoder.state_dict(), save_path)
    # 同时保存到固定路径供 train_hybrid 默认加载
    torch.save(encoder.state_dict(), "output/pretrain_encoder.pth")
    print("\nSaved:", save_path)
    print("Next: python train_hybrid.py --pretrained output/pretrain_encoder.pth")


if __name__ == "__main__":
    main()
