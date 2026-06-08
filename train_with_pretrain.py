"""加载预训练编码器 + 分类微调"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, ECA as ECA_ATTN
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE

EPOCHS = 60
CONFUSABLE = [("0","O"),("0","o"),("O","o"),("1","I"),("1","l"),
              ("I","l"),("5","S"),("C","c")]


class Classifier(make_hand_char(ECA_ATTN)):
    def forward(self, x):
        x = self.direction(self.stem(x))
        x = self.shape2(self.shape1(x))
        x = self.detail(x)
        return self.head(x)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_lbls, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net = Classifier(NUM_CLASSES).to(DEVICE)
    # 加载预训练编码器 (head 是新的)
    pretrained = torch.load("output/pretrain_encoder.pth", map_location=DEVICE)
    encoder_state = {k: v for k, v in pretrained.items()
                     if not k.startswith("head.")}
    net.load_state_dict(encoder_state, strict=False)
    print("Loaded pretrained encoder (%d/%d params)" %
          (len(encoder_state), len(pretrained)))

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        net.train(); total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = crit(net(images), labels)
            loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        net.eval(); corr, tot = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                corr += (net(images).argmax(1).cpu() == labels).sum().item()
                tot += len(labels)
        acc = corr / tot
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] loss=%.3f test=%.4f" %
              (done, EPOCHS, bar, total_loss / len(train_loader), acc), end="", flush=True)
    print()

    # Per-pair eval
    net.eval()
    pair_stats = {("%s/%s" % (a, b)): {"c": 0, "t": 0} for a, b in CONFUSABLE}
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE); preds = net(images).argmax(1).cpu()
            for i in range(len(labels)):
                tl = i2l[labels[i].item()]; pl = i2l[preds[i].item()]
                for a, b in CONFUSABLE:
                    if tl in (a, b) and pl in (a, b):
                        k = "%s/%s" % (a, b); pair_stats[k]["t"] += 1
                        if pl == tl: pair_stats[k]["c"] += 1
    print("Confusable pairs:")
    for a, b in CONFUSABLE:
        k = "%s/%s" % (a, b); s = pair_stats[k]
        if s["t"] > 0: print("  %s vs %s: %.3f (%d/%d)" % (a, b, s["c"]/s["t"], s["c"], s["t"]))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/pretrained_classifier_%s.pth" % ts)
    print("Saved: output/pretrained_classifier_%s.pth" % ts)
