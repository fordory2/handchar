"""单次训练: 50张全训, 5张测试, 无验证集"""
import argparse, datetime, os
import torch

from torch.utils.data import DataLoader
from data_utils import load_split_data, CharDataset, collate_with_augment
from models import (
    HandCharNet, HandCharNetECA, HandCharNetCBAM, HandCharNetCA,
    HandCharNetNoSe, HandCharNetNoDirECA, HandCharNetGroupSE,
)
from project_constants import DEVICE, NUM_CLASSES
from training_utils import evaluate, train_one_epoch, LabelSmoothing
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

EPOCHS = 60
DROPOUT = 0.3
BATCH = 64

MODEL_MAP = {
    "se": HandCharNet, "eca": HandCharNetECA, "cbam": HandCharNetCBAM,
    "ca": HandCharNetCA, "nose": HandCharNetNoSe,
    "nodir_eca": HandCharNetNoDirECA, "gse": HandCharNetGroupSE,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="se", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, _, _ = load_split_data()

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0)

    model = MODEL_MAP[args.model](NUM_CLASSES, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model: %s | Params: %d | Epochs: %d | 50imgs/class train" %
          (args.model, n_params, args.epochs))

    criterion = LabelSmoothing(NUM_CLASSES)
    opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    for ep in range(args.epochs):
        tl, _ = train_one_epoch(model, train_loader, opt, criterion)
        sched.step()
        test_acc = evaluate(model, test_loader)
        done = ep + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        print("\r  Ep%2d/%d [%s] loss=%.3f test=%.4f" %
              (done, args.epochs, bar, tl, test_acc), end="", flush=True)
    print()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), "output/model_%s_%s.pth" % (args.model, ts))
    print("Saved: output/model_%s_%s.pth" % (args.model, ts))


if __name__ == "__main__":
    main()
