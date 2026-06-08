"""Focal Loss: 自动聚焦困难样本, 不牺牲容易类"""
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HandCharNet
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE
from training_utils import evaluate

EPOCHS = 60
FOCAL_GAMMA = 2.0  # 聚焦强度: 越大越关注困难样本


class FocalLoss:
    def __init__(self, gamma=2.0, smoothing=0.1):
        self.gamma = gamma
        self.smoothing = smoothing

    def __call__(self, logits, targets):
        n = logits.shape[1]
        log_probs = F.log_softmax(logits, -1)
        probs = log_probs.exp()
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
            # p_t = sum(y * p), 然后 focal_weight = (1 - p_t)^gamma
            p_t = (probs * smooth_targets).sum(dim=-1)
            focal_weight = (1 - p_t).pow(self.gamma).unsqueeze(-1)
        return -(smooth_targets * log_probs * focal_weight).sum(-1).mean()


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_labels, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net = HandCharNet(NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        net.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        test_acc = evaluate(net, test_loader)
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] loss=%.3f test=%.4f" %
              (done, EPOCHS, bar, total_loss / len(train_loader), test_acc),
              end="", flush=True)
    print()

    # Per-class eval
    net.eval()
    correct = {c: 0 for c in range(NUM_CLASSES)}
    total = {c: 0 for c in range(NUM_CLASSES)}
    confusable = [("0","O"),("0","o"),("O","o"),("1","I"),("1","l"),
                  ("I","l"),("5","S"),("C","c")]
    pair_stats = {("%s/%s" % (a, b)): {"c": 0, "t": 0} for a, b in confusable}
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = net(images).argmax(1).cpu()
            for i in range(len(labels)):
                t_lbl = labels[i].item(); p_lbl = preds[i].item()
                total[t_lbl] += 1
                if p_lbl == t_lbl: correct[t_lbl] += 1
                for a, b in confusable:
                    if i2l[t_lbl] in (a, b) and i2l[p_lbl] in (a, b):
                        k = "%s/%s" % (a, b); pair_stats[k]["t"] += 1
                        if p_lbl == t_lbl: pair_stats[k]["c"] += 1

    print("Confusable pairs:")
    for a, b in confusable:
        k = "%s/%s" % (a, b); s = pair_stats[k]
        if s["t"] > 0: print("  %s vs %s: %.3f (%d/%d)" % (a, b, s["c"]/s["t"], s["c"], s["t"]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/focal_%s.pth" % timestamp)
    print("Saved: output/focal_%s.pth" % timestamp)
