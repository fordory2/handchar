"""三合一: 62类 + 3类辅助头 + 对比损失"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, ECA as ECA_ATTN
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS
from training_utils import compute_pair_accuracy

AUX_CLASSES = 3
CONTRAS_WEIGHT = 0.1
AUX_WEIGHT = 0.1
EPOCHS = 60


def get_aux_label(file_label):
    if file_label.isdigit(): return 0
    if file_label.isupper(): return 1
    return 2


def build_pair_lookup(pair_set, num_classes, device):
    """预构建 [C, C] 的混淆对查找表, 训练时用 gather 实现 O(B^2) 向量化."""
    table = torch.zeros(num_classes, num_classes, dtype=torch.bool, device=device)
    for a, b in pair_set:
        table[a, b] = True
    return table


def contrastive_loss(feats, lbls, pair_table, margin=0.2):
    normed = F.normalize(feats, dim=1)
    dist = 1.0 - normed @ normed.T
    n = len(lbls)
    tri = torch.triu(torch.ones(n, n, device=feats.device), diagonal=1).bool()
    same = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & tri
    conf = pair_table[lbls.unsqueeze(0), lbls.unsqueeze(1)] & tri
    pos = dist[same].mean() if same.any() else torch.tensor(0.0, device=feats.device)
    neg = torch.clamp(margin - dist[conf], min=0).mean() if conf.any() else torch.tensor(0.0, device=feats.device)
    return pos + neg


def train_epoch(model, loader, opt, cls_crit, aux_crit, pair_table, idx2label):
    model.train()
    t, c, a, ct = 0.0, 0.0, 0.0, 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        aux_lbls = torch.tensor([get_aux_label(idx2label[l.item()]) for l in labels], device=DEVICE)
        opt.zero_grad()
        logits, flat, aux_logits = model(images)
        cls_loss = cls_crit(logits, labels)
        aux_loss = aux_crit(aux_logits, aux_lbls)
        cnt_loss = contrastive_loss(flat, labels, pair_table)
        loss = cls_loss + AUX_WEIGHT * aux_loss + CONTRAS_WEIGHT * cnt_loss
        loss.backward(); opt.step()
        t += loss.item(); c += cls_loss.item()
        a += aux_loss.item(); ct += cnt_loss.item()
    n = len(loader)
    return t/n, c/n, a/n, ct/n


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_lbls, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    pair_set = set()
    for a, b in CONFUSABLE_PAIRS:
        if a in l2i and b in l2i:
            pair_set.add((l2i[a], l2i[b])); pair_set.add((l2i[b], l2i[a]))
    pair_table = build_pair_lookup(pair_set, NUM_CLASSES, DEVICE)

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    class DualHeadNet(make_hand_char(ECA_ATTN)):
        def __init__(self, num_classes):
            super().__init__(num_classes)
            self.aux_head = nn.Linear(160, AUX_CLASSES)

        def forward(self, x):
            x = self.direction(self.stem(x))
            x = self.shape2(self.shape1(x))
            fm = self.detail(x)
            flat = F.adaptive_avg_pool2d(fm, 1).flatten(1)
            main_logits = self.head[2:](self.head[1](flat))
            aux_logits = self.aux_head(flat)
            return main_logits, flat, aux_logits

    net = DualHeadNet(NUM_CLASSES).to(DEVICE)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    aux_crit = nn.CrossEntropyLoss()
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        tl, cl, al, cnt = train_epoch(net, train_loader, opt, cls_crit, aux_crit, pair_table, i2l)
        sched.step()
        net.eval(); corr, tot = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                corr += (net(images)[0].argmax(1).cpu() == labels).sum().item()
                tot += len(labels)
        acc = corr / tot
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] cls=%.3f aux=%.3f cnt=%.3f test=%.4f" %
              (done, EPOCHS, bar, cl, al, cnt, acc), end="", flush=True)
    print()

    # Per-pair eval (复用 training_utils, 用 wrapper 取主 logits)
    class _MainHeadWrapper(nn.Module):
        def __init__(self, multi_head_net):
            super().__init__()
            self.multi_head_net = multi_head_net

        def forward(self, x):
            return self.multi_head_net(x)[0]

    pair_acc = compute_pair_accuracy(_MainHeadWrapper(net), test_loader, i2l)
    print("Confusable pairs:")
    for k, v in pair_acc.items():
        print("  %s: %.3f" % (k, v))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/combined_%s.pth" % ts)
    print("Saved: output/combined_%s.pth" % ts)
