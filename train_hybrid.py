"""HybridHandCharNet 联合训练: focal 主损失 + aux 三分类 + contrastive 难例对比.

支持从 pretrain.py 产出的 MAE 权重加载编码器.
"""
import argparse
import datetime
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrastive import build_pair_lookup, contrastive_loss
from data_utils import CharDataset, collate_with_augment, load_split_data
from focal import FocalLoss
from models import HybridHandCharNet
from project_constants import (
    AUX_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS, DEVICE, LEARNING_RATE,
    NUM_CLASSES, TRAIN_EPOCHS,
)
from training_utils import compute_pair_accuracy

CONTRASTIVE_WEIGHT = 0.1
AUX_WEIGHT = 0.1


def aux_label_from_char(char):
    if char.isdigit():
        return 0
    if char.isupper():
        return 1
    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--pretrained", type=str, default="",
                        help="MAE 预训练编码器路径 (output/pretrain_encoder.pth), 空=从头训")
    parser.add_argument("--no_rnn", action="store_true", help="禁用 RNN 分支")
    parser.add_argument("--rnn_cell", type=str, default="lstm", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--rnn_hidden", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_proj_dim", type=int, default=256)
    parser.add_argument("--tiny_gru", action="store_true",
                        help="预设: TinyGRU (cell=gru, hidden=64, layers=1, proj=128)")
    args = parser.parse_args()
    if args.tiny_gru:
        args.rnn_cell = "gru"
        args.rnn_hidden = 64
        args.rnn_layers = 1
        args.rnn_proj_dim = 128

    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, _, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    # 混淆对查找表
    pair_set = set()
    for first_char, second_char in CONFUSABLE_PAIRS:
        if first_char in l2i and second_char in l2i:
            pair_set.add((l2i[first_char], l2i[second_char]))
            pair_set.add((l2i[second_char], l2i[first_char]))
    pair_table = build_pair_lookup(pair_set, NUM_CLASSES, DEVICE)

    # aux 标签查找表 (idx -> aux_class)
    aux_lookup = torch.tensor(
        [aux_label_from_char(i2l[i]) for i in range(NUM_CLASSES)],
        dtype=torch.long, device=DEVICE,
    )

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net = HybridHandCharNet(
        num_classes=NUM_CLASSES, aux_classes=AUX_CLASSES,
        use_rnn=not args.no_rnn,
        rnn_cell=args.rnn_cell,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_proj_dim=args.rnn_proj_dim,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    rnn_desc = "off" if args.no_rnn else "%s/h%d/L%d/d%d" % (
        args.rnn_cell, args.rnn_hidden, args.rnn_layers, args.rnn_proj_dim)
    print("HybridHandCharNet | Params: %d | Epochs: %d | RNN: %s" %
          (n_params, args.epochs, rnn_desc))

    if args.pretrained:
        encoder_state = torch.load(args.pretrained, map_location=DEVICE)
        # 只加载与编码器对应的 key (stem/branches/stages/up/fuse)
        load_state = {k: v for k, v in encoder_state.items()
                      if not k.startswith(("head_", "lstm", "lstm_proj"))}
        missing, unexpected = net.load_state_dict(load_state, strict=False)
        print("Loaded MAE: %d keys (missing=%d, unexpected=%d)" %
              (len(load_state), len(missing), len(unexpected)))

    focal_crit = FocalLoss(gamma=2.0, smoothing=0.1)
    aux_crit = nn.CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = 0.0
    best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}
    for epoch in range(args.epochs):
        net.train()
        total_loss = total_main = total_aux = total_cont = 0.0
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            aux_labels = aux_lookup[labels]
            optimizer.zero_grad()
            main_logits, aux_logits, cont_feat, _ = net(images)
            main_loss = focal_crit(main_logits, labels)
            aux_loss = aux_crit(aux_logits, aux_labels)
            # cont_feat 可能为 None (新模型已移除 contrastive head); 此时跳过对比损失
            if cont_feat is not None:
                cont_loss = contrastive_loss(cont_feat, labels, pair_table)
                loss = main_loss + AUX_WEIGHT * aux_loss + CONTRASTIVE_WEIGHT * cont_loss
            else:
                cont_loss = torch.zeros((), device=DEVICE)
                loss = main_loss + AUX_WEIGHT * aux_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_main += main_loss.item()
            total_aux += aux_loss.item()
            total_cont += cont_loss.item()
            correct += (main_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        # 评估
        net.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                main_logits, _, _, _ = net(images)
                test_correct += (main_logits.argmax(1).cpu() == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total
        train_acc = correct / total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {key: value.cpu().clone() for key, value in net.state_dict().items()}
        n_batches = len(train_loader)
        done = epoch + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        print("\r  Ep%2d/%d [%s] main=%.3f aux=%.3f cnt=%.3f train=%.3f test=%.4f best=%.4f" %
              (done, args.epochs, bar, total_main / n_batches,
               total_aux / n_batches, total_cont / n_batches, train_acc, test_acc, best_test_acc),
              end="", flush=True)
    print()

    net.load_state_dict(best_state)

    # 混淆对评估 (复用 training_utils, 包装取主头)
    class _MainHeadWrapper(nn.Module):
        def __init__(self, hybrid_net):
            super().__init__()
            self.hybrid_net = hybrid_net

        def forward(self, x):
            return self.hybrid_net(x)[0]

    pair_acc = compute_pair_accuracy(_MainHeadWrapper(net), test_loader, i2l)
    print("Confusable pairs:")
    for k, v in pair_acc.items():
        print("  %s: %.3f" % (k, v))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "output/hybrid_%s.pth" % timestamp
    torch.save(net.state_dict(), save_path)
    print("Saved best (test=%.4f): %s" % (best_test_acc, save_path))


if __name__ == "__main__":
    main()
