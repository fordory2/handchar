"""四层消融: 逐混淆对分析，单次 train/test"""

import datetime
import os

import torch
from torch.utils.data import DataLoader

from cli_utils import parse_training_args
from data_utils import CharDataset, load_split_data
from models import (
    HandCharNet,
    HandCharNetNoDirection,
    HandCharNetNoMultiScale,
    HandCharNetNoSe,
)
from project_constants import BATCH_SIZE, CONFUSABLE_PAIRS, DEVICE, NUM_CLASSES
from training_utils import compute_pair_accuracy, fit_best_model


def train_model(model_cls, tag, train_loader, test_loader, epochs):
    print("\n" + "=" * 55)
    print("Training: %s" % tag)
    print("=" * 55)
    model = model_cls(NUM_CLASSES).to(DEVICE)
    best_accuracy, _ = fit_best_model(model, train_loader, test_loader, epochs)
    return model, best_accuracy

def main():
    args = parse_training_args(lambda parser: parser.add_argument("--models", type=str, default="all"))
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Device: %s | Ablation, %d epochs | seed=%d" % (DEVICE, args.epochs, args.seed))

    train_data, _, test_data, all_labels, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}
    print("Train: %d  Test(051-055): %d  Classes: %d" %
          (len(train_data), len(test_data), len(all_labels)))

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    all_models = [
        (HandCharNet,              "Full (4-Layer)"),
        (HandCharNetNoDirection,  "-Layer2 Direction"),
        (HandCharNetNoSe,         "-Layer3 SE"),
        (HandCharNetNoMultiScale, "-Layer4 MultiScale"),
    ]
    if args.models != "all":
        idx_list = [int(x.strip()) for x in args.models.split(",")]
        models = [all_models[i] for i in idx_list]
    else:
        models = all_models[:]

    results, pair_results = {}, {}
    for model_cls, tag in models:
        model, test_acc = train_model(model_cls, tag, train_loader, test_loader, args.epochs)
        results[tag] = test_acc
        safe = tag.replace(" ", "_").replace(":", "").replace("/", "_")
        torch.save(model.state_dict(), "output/ablation_%s_%s.pth" % (safe, timestamp))

        pair_acc = compute_pair_accuracy(model, test_loader, i2l)
        pair_results[tag] = pair_acc
        print("  Confusable pairs:")
        for a, b in CONFUSABLE_PAIRS:
            k = "%s/%s" % (a, b)
            if k in pair_acc:
                print("    %s vs %s: %.3f" % (a, b, pair_acc[k]))

    # Summary
    sep = "=" * 60
    lines = [sep, "Ablation Results (test=051-055, seed=%d) - %s" % (args.seed, timestamp), sep,
             "Layer2 DirectionConv -> 5/S, 0/o | Layer3 SE -> 1/I,l | Layer4 MultiScale -> O/o,C/c",
             "-" * 60]
    baseline = results[models[0][1]]
    lines.append("%-30s: %.4f  (baseline)" % (models[0][1], baseline))
    for _, tag in models[1:]:
        delta = baseline - results[tag]
        lines.append("%-30s: %.4f  (delta=%+.4f, %+.1f%%)" % (tag, results[tag], delta, delta * 100))

    pk = ["%s/%s" % (a, b) for a, b in CONFUSABLE_PAIRS]
    short = [k.replace("/", " vs ") for k in pk]
    lines.append("")
    lines.append("%-22s %s" % ("Model", " ".join("%-7s" % s for s in short)))
    lines.append("-" * 90)
    for _, tag in models:
        pr = pair_results.get(tag, {})
        parts = [tag[:21]]
        for k in pk:
            parts.append("%.3f" % pr[k] if k in pr else "-")
        lines.append("%-22s %s" % (parts[0], " ".join("%-7s" % v for v in parts[1:])))

    for line in lines:
        print(line)

    with open("output/ablation_%s.txt" % timestamp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open("output/ablation_%s.csv" % timestamp, "w", encoding="utf-8-sig") as f:
        f.write("Model,Accuracy,Delta," + ",".join(short) + "\n")
        f.write("%s,%.4f,0" % (models[0][1], baseline) + ",-" * len(pk) + "\n")
        for _, tag in models[1:]:
            f.write("%s,%.4f,%+.4f" % (tag, results[tag], baseline - results[tag]))
            pr = pair_results.get(tag, {})
            for k in pk:
                f.write(",%.3f" % pr[k] if k in pr else ",-")
            f.write("\n")
    print("\nFiles: output/ablation_%s.{txt,csv}" % timestamp)


if __name__ == "__main__":
    main()
