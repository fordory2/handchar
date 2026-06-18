"""多权重集成: softmax 平均后 argmax. 支持不同结构 (含/不含 pair head)."""
import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as functional

from data import NUM_CLASSES, PAIR_NUM_CLASSES, load_split_data, make_loader
from model import ConvNeXtV2UNet
from tta import TTA_VIEWS, _affine


def load_model(path, device):
    state = torch.load(path, map_location=device)
    has_pair = any(k.startswith("pair_fc") for k in state)
    net = ConvNeXtV2UNet(num_classes=NUM_CLASSES,
                          num_pair_classes=PAIR_NUM_CLASSES if has_pair else 0).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _logits(net, x):
    out = net(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict_ensemble(nets, x, use_tta=False):
    views = TTA_VIEWS if use_tta else [(0.0, 0.0, 0.0)]
    probs = None
    for ang, dx, dy in views:
        x_view = _affine(x, ang, dx, dy) if (ang or dx or dy) else x
        for net in nets:
            p = functional.softmax(_logits(net, x_view), dim=-1)
            probs = p if probs is None else probs + p
    return probs / (len(views) * len(nets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True,
                        help="多个权重 .pth 路径")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--per_class", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_d, all_labels, _ = load_split_data()
    loader = make_loader(test_d, batch=64, train=False)

    nets = []
    for w in args.weights:
        if not os.path.exists(w):
            raise FileNotFoundError(w)
        nets.append(load_model(w, device))
    print("Ensemble of %d models: %s" % (len(nets), args.weights))

    correct = total = 0
    per_class_c = defaultdict(int)
    per_class_t = defaultdict(int)
    for images, labels in loader:
        images = images.to(device)
        probs = predict_ensemble(nets, images, use_tta=args.tta)
        pred = probs.argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
        for i in range(len(labels)):
            lbl = int(labels[i].item())
            per_class_t[lbl] += 1
            if int(pred[i].item()) == lbl:
                per_class_c[lbl] += 1
    tag = "TTA" if args.tta else "single"
    print("Ensemble test acc (%s): %.4f  (n=%d)" % (tag, correct / total, total))

    if args.per_class:
        per_class = {c: per_class_c[c] / per_class_t[c] for c in per_class_t}
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
        print("Worst 10:")
        for c, a in worst:
            print("  %s: %.3f" % (all_labels[c], a))


if __name__ == "__main__":
    main()
