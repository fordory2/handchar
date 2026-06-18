"""FCMAE 自监督预训练: 全 3410 张图 mask 重建.

mask_ratio=0.6, patch=8×8, 重建损失仅在 mask 区域算 L2.
保存编码器权重到 output/pretrain.pth.
"""
import argparse
import math
import os

import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data import (CharDataset, GeoAugment, IMG_SIZE, load_all_data,
                  make_mae_mask)
from model import ConvNeXtV2UNet, count_params


def collate_pretrain_local(batch):
    images, _ = zip(*batch)
    images = torch.stack(images)
    images = GeoAugment.apply(images.cuda() if torch.cuda.is_available() else images)
    return images


def warmup_cosine(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def mae_loss(recon, target, mask_full):
    """仅在 mask 区域算 L2. mask_full: [B, 1, H, W], 1=可见, 0=mask."""
    masked = 1.0 - mask_full
    diff = (recon - target).pow(2) * masked
    return diff.sum() / (masked.sum() + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    data = load_all_data()
    ds = CharDataset(data)
    loader = DataLoader(ds, args.batch, shuffle=True, num_workers=0, drop_last=True,
                        collate_fn=lambda b: torch.stack([x for x, _ in b]))

    net = ConvNeXtV2UNet().to(device)
    n_params = count_params(net)
    print("FCMAE pretrain | Params: %.2fM | Epochs: %d | Mask ratio: %.2f" %
          (n_params / 1e6, args.epochs, args.mask_ratio))
    print("Data: %d imgs | Batches/epoch: %d" % (len(ds), len(loader)))

    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = LambdaLR(opt, lr_lambda=lambda e: warmup_cosine(e, args.warmup, args.epochs))

    log_path = "output/pretrain_log.txt"
    width, height = IMG_SIZE
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,lr\n")

    for ep in range(args.epochs):
        net.train()
        total_loss = 0.0
        n_batch = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            batch = GeoAugment.apply(batch)
            mask = make_mae_mask(batch.size(0), height, width,
                                  patch=args.patch, ratio=args.mask_ratio,
                                  device=device)
            recon = net(batch, mask=mask, return_recon=True)
            loss = mae_loss(recon, batch, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        sched.step()
        avg = total_loss / n_batch
        cur_lr = opt.param_groups[0]['lr']
        done = ep + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        print("\r  Ep%3d/%d [%s] loss=%.4f lr=%.2e" %
              (done, args.epochs, bar, avg, cur_lr), end="", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("%d,%.6f,%.6e\n" % (done, avg, cur_lr))
    print()
    torch.save(net.state_dict(), "output/pretrain.pth")
    print("Saved: output/pretrain.pth")


if __name__ == "__main__":
    main()
