"""5-fold CV 评估: 每折 ckpt 在自身 val 集上单视图+9视图TTA, 汇聚全量预测出完整指标."""
import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import CharDataset, load_kfold_splits
from metrics import classification_metrics
from models import (ConvNeXtV2Char, HybridHandCharNet, HybridHandCharNetCnxBypassA,
                    HybridHandCharNetCnxBypassB, HybridHandCharNetCnxStem,
                    HybridHandCharNetR18Stem, ResNet18Pretrained, ResNet50UNetChar,
                    TransferBackbone, TransferBackboneMS, TransferBackboneAdapter)
from project_constants import BATCH_SIZE, CONFUSABLE_PAIRS, DEVICE, NUM_CLASSES
from tta import tta_predict


def _main_logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _build_net(state, arch):
    has_rnn = any(k.startswith("lstm") for k in state)
    if arch.startswith("transfer_ms"):
        parts = arch.split(":")
        model_name = parts[1] if len(parts) > 1 else "resnet50"
        input_size = int(parts[2]) if len(parts) > 2 else 160
        ms_stages = int(parts[3]) if len(parts) > 3 else 2
        net = TransferBackboneMS(model_name=model_name, num_classes=NUM_CLASSES,
                                 pretrained=False, input_size=input_size,
                                 ms_stages=ms_stages).to(DEVICE)
        net.load_state_dict(state, strict=False); net.eval(); return net
    if arch.startswith("transfer_adapter"):
        # 谱: transfer_adapter:model:size:adapt_dim:branchspec:head (后面可省, 用默认)
        # branchspec 用 + 连接, 如 spatial+channel+multiscale+gabor+morph
        parts = arch.split(":")
        model_name = parts[1] if len(parts) > 1 else "resnet50"
        input_size = int(parts[2]) if len(parts) > 2 else 160
        adapt_dim = int(parts[3]) if len(parts) > 3 else 64
        branches = tuple(parts[4].split("+")) if len(parts) > 4 and parts[4] else \
            ("spatial", "channel", "multiscale")
        head = parts[5] if len(parts) > 5 else "linear"
        intensity_mode = parts[6] if len(parts) > 6 else "stats"
        weber = len(parts) > 7 and parts[7] in ("1", "weber", "true")
        net = TransferBackboneAdapter(model_name=model_name, num_classes=NUM_CLASSES,
                                      pretrained=False, input_size=input_size,
                                      adapt_dim=adapt_dim, branches=branches,
                                      intensity_mode=intensity_mode, weber_norm=weber,
                                      head=head).to(DEVICE)
        net.load_state_dict(state, strict=False); net.eval(); return net
    if arch.startswith("transfer"):
        # arch 格式: "transfer:<timm模型名>:<input_size>" (后两段可省, 默认 resnet50/160)
        parts = arch.split(":")
        model_name = parts[1] if len(parts) > 1 else "resnet50"
        input_size = int(parts[2]) if len(parts) > 2 else 160
        net = TransferBackbone(model_name=model_name, num_classes=NUM_CLASSES,
                               pretrained=False, input_size=input_size,
                               gray_to_3ch=True, imagenet_norm=True).to(DEVICE)
        net.load_state_dict(state, strict=False)
        net.eval()
        return net
    if arch == "convnextv2p":
        net = ConvNeXtV2Char(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    elif arch == "resnet18p":
        net = ResNet18Pretrained(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    elif arch == "hybrid_r18stem":
        net = HybridHandCharNetR18Stem(num_classes=NUM_CLASSES, pretrained=False, use_rnn=has_rnn).to(DEVICE)
    elif arch == "hybrid_cnxstem":
        net = HybridHandCharNetCnxStem(num_classes=NUM_CLASSES, pretrained=False, use_rnn=has_rnn).to(DEVICE)
    elif arch == "hybrid_cnxbypassA":
        net = HybridHandCharNetCnxBypassA(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    elif arch == "hybrid_cnxbypassB":
        net = HybridHandCharNetCnxBypassB(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    elif arch == "resnet50_unet":
        net = ResNet50UNetChar(num_classes=NUM_CLASSES, use_rnn=has_rnn).to(DEVICE)
    else:
        net = HybridHandCharNet(num_classes=NUM_CLASSES, use_rnn=has_rnn).to(DEVICE)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


@torch.no_grad()
def _eval(net, loader, tta=False):
    correct = total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        if tta:
            pred = tta_predict(net, images).argmax(1).cpu()
        else:
            pred = _main_logits(net(images)).argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def _collect_probs(net, loader, tta=False):
    """返回 (probs[N,C], labels[N]) 供汇聚算完整指标。"""
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        if tta:
            logits = tta_predict(net, images)
        else:
            logits = _main_logits(net(images))
        all_probs.append(F.softmax(logits, dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="output/hybrid_cv5_f*.pth",
                        help="ckpt glob 模式, 自动按 fN 编号匹配 fold")
    parser.add_argument("--arch", type=str, default="hybrid",
                        help="支持 hybrid/resnet18p/convnextv2p/... 及 transfer:<model>:<size>")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.pattern))
    if not ckpts:
        raise FileNotFoundError("no ckpt matched %s" % args.pattern)

    fold_to_ckpt = {}
    for p in ckpts:
        m = re.search(r"_f(\d+)_", os.path.basename(p))
        if m:
            fold_to_ckpt[int(m.group(1))] = p

    folds, _, _ = load_kfold_splits(n_splits=args.n_folds, random_state=args.seed)

    single_accs, tta_accs = [], []
    pooled_s, pooled_t = [], []  # (probs, labels) 对
    print("=" * 70)
    print("5-fold CV evaluation (single + 9-view TTA)")
    print("=" * 70)
    for fold_idx in range(args.n_folds):
        if fold_idx not in fold_to_ckpt:
            print("fold %d: no ckpt, skip" % fold_idx)
            continue
        ckpt = fold_to_ckpt[fold_idx]
        _, val_data = folds[fold_idx]
        ds = CharDataset(val_data, train=False)
        loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0)
        state = torch.load(ckpt, map_location=DEVICE)
        net = _build_net(state, args.arch)
        s = _eval(net, loader, tta=False)
        t = _eval(net, loader, tta=True)
        single_accs.append(s); tta_accs.append(t)
        # 收集概率用于汇聚指标
        ps, ls = _collect_probs(net, loader, tta=False)
        pt, lt = _collect_probs(net, loader, tta=True)
        pooled_s.append((ps, ls)); pooled_t.append((pt, lt))
        print("  fold %d (n=%d): single=%.4f  TTA=%.4f  Δ=%+.4f  | %s" %
              (fold_idx, len(val_data), s, t, t - s, os.path.basename(ckpt)))
        del net
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    def stats(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, v ** 0.5

    if single_accs:
        m_s, std_s = stats(single_accs)
        m_t, std_t = stats(tta_accs)
        print("-" * 70)
        print("Single mean=%.4f ± %.4f" % (m_s, std_s))
        print("TTA    mean=%.4f ± %.4f  (Δ=%+.4f)" % (m_t, std_t, m_t - m_s))
        print("=" * 70)

        # 汇聚全量预测 → 完整指标 (取 TTA 版本, 若 TTA 更差则取 single)
        use_tta = m_t >= m_s
        pooled = pooled_t if use_tta else pooled_s
        all_probs = torch.cat([p for p, _ in pooled])
        all_lbls = torch.cat([l for _, l in pooled])
        _, l2i = load_kfold_splits(n_splits=args.n_folds, random_state=args.seed)[1:]
        i2l = {v: k for k, v in l2i.items()}

        met, report = classification_metrics(all_probs, all_lbls, i2l, worst_k=10)
        print("\n汇聚 %d 折全量预测 (%s 视图) 完整指标:"
              % (len(single_accs), "TTA" if use_tta else "single"))
        print(report)
        # 混淆对
        print("混淆对:")
        from ensemble import _pair_report
        print(_pair_report(all_probs, all_lbls, i2l))
        print("=" * 70)


if __name__ == "__main__":
    main()
