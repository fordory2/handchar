"""对比损失: 在特征空间推远混淆对"""
import torch, torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, SE as SE_ATTENTION
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS
from training_utils import LabelSmoothing

CONTRAS_WEIGHT = 0.1
EPOCHS = 60


def build_pair_lookup(pair_set, num_classes, device):
    table = torch.zeros(num_classes, num_classes, dtype=torch.bool, device=device)
    for a, b in pair_set:
        table[a, b] = True
    return table


def contrastive_loss(features, label_tensor, pair_table, margin=0.2):
    """Triplet-style 向量化: 混淆对推远, 同类拉近"""
    normed = functional.normalize(features, dim=1)
    batch_size = int(normed.shape[0])
    dist_matrix = 1.0 - normed @ normed.T
    tri_mask = torch.triu(torch.ones(batch_size, batch_size, device=features.device), diagonal=1).bool()
    same_class = (label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)) & tri_mask
    conf_mask = pair_table[label_tensor.unsqueeze(0), label_tensor.unsqueeze(1)] & tri_mask
    if same_class.any():
        pos_loss = dist_matrix[same_class].mean()
    else:
        pos_loss = torch.tensor(0.0, device=features.device)
    if conf_mask.any():
        neg_loss = torch.clamp(margin - dist_matrix[conf_mask], min=0).mean()
    else:
        neg_loss = torch.tensor(0.0, device=features.device)
    return pos_loss + neg_loss


# 返回 (logits, features) 的变体
class FeatureNet(make_hand_char(SE_ATTENTION)):
    def forward(self, x):
        x = self.direction(self.stem(x))
        x = self.shape2(self.shape1(x))
        feature_map = self.detail(x)
        flat = functional.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        output_logits = self.head[2:](self.head[1](flat))
        return output_logits, flat


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_labels, l2i = load_split_data()
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

    net = FeatureNet(NUM_CLASSES).to(DEVICE)
    cls_criterion = LabelSmoothing(NUM_CLASSES)
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        net.train()
        total_loss, cls_sum, cnt_sum = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, feats = net(images)
            cls_loss = cls_criterion(logits, labels)
            cnt_loss = contrastive_loss(feats, labels, pair_table)
            loss = cls_loss + CONTRAS_WEIGHT * cnt_loss
            loss.backward()
            opt.step()
            total_loss += loss.item(); cls_sum += cls_loss.item(); cnt_sum += cnt_loss.item()
        sched.step()
        # eval
        net.eval(); correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                logits_out, _ = net(images)
                correct += (logits_out.argmax(1).cpu() == labels).sum().item()
                total += len(labels)
        test_acc = correct / total
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] loss=%.3f cls=%.3f cnt=%.3f test=%.4f" %
              (done, EPOCHS, bar, total_loss/len(train_loader),
               cls_sum/len(train_loader), cnt_sum/len(train_loader), test_acc),
              end="", flush=True)
    print()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/contrastive_%s.pth" % timestamp)
    print("Saved: output/contrastive_%s.pth" % timestamp)
