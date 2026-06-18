"""评估: 单次或 TTA 推理 test acc."""
import argparse
import os
from collections import defaultdict

import torch

from data import (NUM_CLASSES, PAIR_NUM_CLASSES, load_random_split_data,
                   load_split_data, make_loader)
from model import ConvNeXtV2UNet
from tta import tta_predict


def _main_logits(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def eval_single(net, loader, device):
    net.eval()
    correct = total = 0
    per_class_c = defaultdict(int)
    per_class_t = defaultdict(int)
    for images, labels in loader:
        images = images.to(device)
        pred = _main_logits(net(images)).argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
        for i in range(len(labels)):
            lbl = int(labels[i].item())
            per_class_t[lbl] += 1
            if int(pred[i].item()) == lbl:
                per_class_c[lbl] += 1
    return correct / total, {c: per_class_c[c] / per_class_t[c] for c in per_class_t}


@torch.no_grad()
def eval_tta(net, loader, device):
    net.eval()
    correct = total = 0
    per_class_c = defaultdict(int)
    per_class_t = defaultdict(int)
    for images, labels in loader:
        images = images.to(device)
        probs = tta_predict(net, images)
        pred = probs.argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
        for i in range(len(labels)):
            lbl = int(labels[i].item())
            per_class_t[lbl] += 1
            if int(pred[i].item()) == lbl:
                per_class_c[lbl] += 1
    return correct / total, {c: per_class_c[c] / per_class_t[c] for c in per_class_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="output/finetune.pth")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--per_class", action="store_true")
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.random_split:
        _, _, test_d, all_labels, _ = load_random_split_data(
            ratio=(0.7, 0.2, 0.1), seed=args.split_seed)
    else:
        _, _, test_d, all_labels, _ = load_split_data()
    loader = make_loader(test_d, batch=64, train=False)

    if not os.path.exists(args.weights):
        raise FileNotFoundError("缺权重: %s" % args.weights)
    state = torch.load(args.weights, map_location=device)
    has_pair = any(k.startswith("pair_fc") for k in state)
    net = ConvNeXtV2UNet(num_classes=NUM_CLASSES,
                          num_pair_classes=PAIR_NUM_CLASSES if has_pair else 0).to(device)
    net.load_state_dict(state)

    if args.tta:
        acc, per_class = eval_tta(net, loader, device)
        tag = "TTA"
    else:
        acc, per_class = eval_single(net, loader, device)
        tag = "single"
    print("Test acc (%s): %.4f  (n=%d)" % (tag, acc, len(test_d)))

    if args.per_class:
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
        best = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Worst 10:")
        for c, a in worst:
            print("  %s: %.3f" % (all_labels[c], a))
        print("Best 10:")
        for c, a in best:
            print("  %s: %.3f" % (all_labels[c], a))


if __name__ == "__main__":
    main()
