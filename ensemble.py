"""双模型集成: Full+CA 和 -Dir+SE 互补推理"""
import torch
from torch.utils.data import DataLoader
import numpy as np, os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HandCharNetCA, HandCharNetNoDirection
from data_utils import load_split_data, CharDataset
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS
from training_utils import evaluate

MODEL_PATHS = {
    "ca": "output/model_Full+CA_*.pth",
    "no_dir": "output/model_-Dir+SE_*.pth",
}


def load_model(model_class, path):
    net = model_class(NUM_CLASSES).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def ensemble_predict(net_a, net_b, loader, weight_a=0.5):
    """加权平均两个模型的 logits"""
    net_a.eval(); net_b.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = weight_a * net_a(images) + (1 - weight_a) * net_b(images)
            preds = logits.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels


def main():
    import glob, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca_model", type=str, required=True)
    parser.add_argument("--no_dir_model", type=str, required=True)
    parser.add_argument("--weight", type=float, default=0.5,
                        help="CA 权重 (0~1), 默认0.5")
    args = parser.parse_args()

    _, _, test_d, all_labels, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}
    test_ds = CharDataset(test_d, train=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net_ca = load_model(HandCharNetCA, args.ca_model)
    net_no_dir = load_model(HandCharNetNoDirection, args.no_dir_model)

    # 单独评估
    ca_acc = evaluate(net_ca, test_loader)
    nd_acc = evaluate(net_no_dir, test_loader)
    ens_acc, preds, labels = ensemble_predict(net_ca, net_no_dir, test_loader, args.weight)

    print("=" * 55)
    print("Ensemble Results (test=051-055)")
    print("=" * 55)
    print("Full+CA:      %.4f" % ca_acc)
    print("-Dir+SE:      %.4f" % nd_acc)
    print("Ensemble(w=%.2f): %.4f" % (args.weight, ens_acc))
    print("Delta vs CA:  %+.4f" % (ens_acc - ca_acc))
    print("Delta vs ND:  %+.4f" % (ens_acc - nd_acc))

    # 逐混淆对 (基于上面 ensemble_predict 已得到的 preds/labels)
    print("\nConfusable pairs:")
    pair_stats = {("%s/%s" % (a, b)): {"c": 0, "t": 0} for a, b in CONFUSABLE_PAIRS}
    for i in range(len(labels)):
        tl = i2l[labels[i]]; pl = i2l[preds[i]]
        for a, b in CONFUSABLE_PAIRS:
            if tl in (a, b) and pl in (a, b):
                k = "%s/%s" % (a, b); pair_stats[k]["t"] += 1
                if pl == tl: pair_stats[k]["c"] += 1
    for a, b in CONFUSABLE_PAIRS:
        k = "%s/%s" % (a, b); s = pair_stats[k]
        if s["t"] > 0:
            print("  %s vs %s: %.3f (%d/%d)" % (a, b, s["c"]/s["t"], s["c"], s["t"]))


if __name__ == "__main__":
    main()
