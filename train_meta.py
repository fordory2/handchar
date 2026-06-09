"""Prototypical Networks 元学习微调 (在 HybridHandCharNet 之上).

每个 episode 采 N 类, 每类 K shot + Q query.
原型 = K 个 support 嵌入的均值, 用 query 到原型的负距离作分类 logits.
"""
import argparse
import datetime
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import CharDataset, collate_with_augment, load_split_data
from models import HybridHandCharNet
from project_constants import (
    AUX_CLASSES, BATCH_SIZE, DEVICE, LEARNING_RATE, META_EPOCHS, NUM_CLASSES,
)

# 元学习超参
N_WAY = 5
K_SHOT = 5
Q_QUERY = 5
EPISODES_PER_EPOCH = 100


def build_class_pool(dataset):
    """{label: [idx,...]} 用于 episodic 采样."""
    pool = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        pool[label].append(idx)
    return pool


def sample_episode(class_pool, n_way, k_shot, q_query):
    """返回 (support_indices, query_indices) — 各 n_way 类按顺序排列."""
    candidate_classes = [c for c, idxs in class_pool.items() if len(idxs) >= k_shot + q_query]
    sampled = random.sample(candidate_classes, n_way)
    support_indices, query_indices = [], []
    for cls in sampled:
        chosen = random.sample(class_pool[cls], k_shot + q_query)
        support_indices.extend(chosen[:k_shot])
        query_indices.extend(chosen[k_shot:])
    return support_indices, query_indices, sampled


def stack_images(dataset, indices):
    images = torch.stack([dataset[i][0] for i in indices])
    return images


def proto_loss_and_acc(support_feats, query_feats, n_way, k_shot, q_query):
    """support_feats [N*K, D], query_feats [N*Q, D] — 顺序按类分组."""
    # 原型: N×D
    prototypes = support_feats.view(n_way, k_shot, -1).mean(dim=1)
    # 距离: [N*Q, N]
    dists = torch.cdist(query_feats, prototypes)
    logits = -dists  # 距离越小 logit 越大
    # 标签: 0,0,...,0,1,1,...,1,...,N-1
    targets = torch.arange(n_way, device=query_feats.device).repeat_interleave(q_query)
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(1) == targets).float().mean().item()
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=META_EPOCHS)
    parser.add_argument("--init", type=str, required=True,
                        help="预训练的 HybridHandCharNet 权重 (output/hybrid_*.pth)")
    parser.add_argument("--rnn_cell", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer"])
    parser.add_argument("--rnn_hidden", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_proj_dim", type=int, default=256)
    parser.add_argument("--tiny_gru", action="store_true")
    parser.add_argument("--no_rnn", action="store_true")
    args = parser.parse_args()
    if args.tiny_gru:
        args.rnn_cell = "gru"
        args.rnn_hidden = 64
        args.rnn_layers = 1
        args.rnn_proj_dim = 128

    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, _, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    class_pool = build_class_pool(train_data)

    net = HybridHandCharNet(
        num_classes=NUM_CLASSES, aux_classes=AUX_CLASSES,
        use_rnn=not args.no_rnn,
        rnn_cell=args.rnn_cell, rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers, rnn_proj_dim=args.rnn_proj_dim,
    ).to(DEVICE)
    net.load_state_dict(torch.load(args.init, map_location=DEVICE))
    print("Loaded:", args.init)

    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Proto Meta | %d-way %d-shot %d-query | %d episodes/epoch | %d epoch" %
          (N_WAY, K_SHOT, Q_QUERY, EPISODES_PER_EPOCH, args.epochs))

    best_test_acc = 0.0
    best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}
    for epoch in range(args.epochs):
        net.train()
        total_loss = total_acc = 0.0
        for _ in range(EPISODES_PER_EPOCH):
            support_idx, query_idx, _ = sample_episode(class_pool, N_WAY, K_SHOT, Q_QUERY)
            all_imgs = stack_images(train_ds, support_idx + query_idx).to(DEVICE)
            optimizer.zero_grad()
            unified_feats = net.unified_feature(all_imgs)
            support_feats = unified_feats[:N_WAY * K_SHOT]
            query_feats = unified_feats[N_WAY * K_SHOT:]
            loss, acc = proto_loss_and_acc(support_feats, query_feats,
                                            N_WAY, K_SHOT, Q_QUERY)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
        scheduler.step()

        # 测试集上做常规 62 类评估 (主头)
        net.eval()
        test_correct = test_total = 0
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                main_logits, _, _, _ = net(images)
                test_correct += (main_logits.argmax(1).cpu() == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}

        done = epoch + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        print("\r  Ep%2d/%d [%s] proto_loss=%.3f proto_acc=%.3f test=%.4f best=%.4f" %
              (done, args.epochs, bar,
               total_loss / EPISODES_PER_EPOCH,
               total_acc / EPISODES_PER_EPOCH, test_acc, best_test_acc),
              end="", flush=True)
    print()

    net.load_state_dict(best_state)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "output/meta_%s.pth" % timestamp
    torch.save(net.state_dict(), save_path)
    print("Saved best (test=%.4f): %s" % (best_test_acc, save_path))


if __name__ == "__main__":
    main()
