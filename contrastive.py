"""对比损失: 在特征空间推远混淆对"""
import torch, torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import make_hand_char, SE as SE_ATTENTION
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE
from training_utils import LabelSmoothing

CONFUSABLE = [("0","O"),("0","o"),("O","o"),("1","I"),("1","l"),
              ("I","l"),("5","S"),("C","c")]
CONTRAS_WEIGHT = 0.1
EPOCHS = 60


def contrastive_loss(features, label_tensor, confusable_pairs, margin=0.2):
    """Triplet-style 向量化: 混淆对推远, 同类拉近"""
    normed = functional.normalize(features, dim=1)
    batch_size = int(normed.shape[0])
    # 全对余弦距离矩阵 (向量化)
    dist_matrix = 1.0 - normed @ normed.T  # [B, B]
    # 上三角 mask
    tri_mask = torch.triu(torch.ones(batch_size, batch_size, device=features.device), diagonal=1)
    # 同类 mask
    same_class = label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)
    same_class = same_class & tri_mask.bool()
    # 混淆对 mask
    conf_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=features.device)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if (label_tensor[i].item(), label_tensor[j].item()) in confusable_pairs:
                conf_mask[i, j] = True
    # 同类损失: 拉近距离
    if same_class.any():
        pos_loss = dist_matrix[same_class].mean()
    else:
        pos_loss = torch.tensor(0.0, device=features.device)
    # 混淆对损失: 推远超过 margin
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
    for a, b in CONFUSABLE:
        if a in l2i and b in l2i:
            pair_set.add((l2i[a], l2i[b])); pair_set.add((l2i[b], l2i[a]))

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
            cnt_loss = contrastive_loss(feats, labels, pair_set)
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
