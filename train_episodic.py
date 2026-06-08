"""情节训练: 每个 batch 只含 5 个类, 必须包含至少一对混淆类"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime, numpy as np, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, ECA as ECA_ATTN
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS

EPOCHS = 60
CLASSES_PER_BATCH = 5


class EpisodicSampler(Sampler):
    """每个 batch 只抽 5 个类, 必须含至少一组混淆类"""
    def __init__(self, dataset, confusable_pairs, batch_size, classes_per_batch=5):
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)
        self.all_classes = list(self.class_indices.keys())
        self.pairs = confusable_pairs
        self.batch_size = batch_size
        self.n_per_class = batch_size // classes_per_batch
        self.length = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.length):
            pair = random.choice(self.pairs)
            selected = list(pair)
            remaining = [c for c in self.all_classes if c not in selected]
            for _ in range(5 - len(selected)):
                c = random.choice(remaining)
                if c not in selected: selected.append(c)
            indices = []
            for cls in selected:
                n = min(self.n_per_class, len(self.class_indices[cls]))
                indices += random.sample(self.class_indices[cls], n)
            random.shuffle(indices)
            yield from indices[:self.batch_size]

    def __len__(self):
        return self.length * self.batch_size


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_lbls, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    # 混淆类索引
    pair_idx = set()
    for a, b in CONFUSABLE_PAIRS:
        if a in l2i and b in l2i:
            pair_idx.add((l2i[a], l2i[b]))
    pair_idx_list = list(pair_idx)

    class PairHeadNet(make_hand_char(ECA_ATTN)):
        def forward(self, x):
            x = self.direction(self.stem(x))
            x = self.shape2(self.shape1(x))
            return self.head(self.detail(x))

    net = PairHeadNet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    sampler = EpisodicSampler(train_ds, pair_idx_list, BATCH_SIZE, CLASSES_PER_BATCH)
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    for ep in range(EPOCHS):
        net.train(); total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward(); opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        sched.step()
        net.eval(); t_corr, t_tot = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                t_corr += (net(images).argmax(1).cpu() == labels).sum().item()
                t_tot += len(labels)
        acc = t_corr / t_tot
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] loss=%.3f train=%.3f test=%.4f" %
              (done, EPOCHS, bar, total_loss / len(train_loader), correct/total, acc),
              end="", flush=True)
    print()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/episodic_%s.pth" % ts)
    print("Saved: output/episodic_%s.pth" % ts)
