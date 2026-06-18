"""监督微调: 加载 FCMAE 预训练权重, 强增强 + mixup/cutmix + EMA."""
import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import (NUM_CLASSES, PAIR_NUM_CLASSES, labels_to_pair,
                   load_combined_train_data, load_random_split_data,
                   load_split_data, make_loader)
from model import ConvNeXtV2UNet, count_params


class LabelSmoothing(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_prob = functional.log_softmax(pred, -1)
        with torch.no_grad():
            smooth = torch.full_like(log_prob, self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), self.confidence)
        return (-smooth * log_prob).sum(-1).mean()


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1 - d)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)


def mixup_cutmix(images, labels, mixup_a=0.2, cutmix_a=1.0):
    """随机选 mixup 或 cutmix 混合 batch."""
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    y_a, y_b = labels, labels[perm]
    if torch.rand(1).item() < 0.5:  # cutmix
        lam = float(torch.distributions.Beta(cutmix_a, cutmix_a).sample().item())
        _, _, height, width = images.shape
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        cy = int(torch.randint(0, height, (1,)).item())
        cx = int(torch.randint(0, width, (1,)).item())
        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
        return mixed, y_a, y_b, lam
    # mixup
    lam = float(torch.distributions.Beta(mixup_a, mixup_a).sample().item())
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, y_a, y_b, lam


def warmup_cosine(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def layer_lr_groups(model, base_lr, decay=0.75):
    """decay=1.0 时退化为单一 lr 组 (纯监督无 LLRD)."""
    """Layer-wise LR decay: 浅层 lr 小, 深层 lr 大. 微调标准技巧.

    层级 (深 -> 浅):
      cls_head(0) -> dec1(1) -> dec2(2) -> dec3(3) -> bottleneck(4)
      -> stage3(5) -> stage2(6) -> stage1(7) -> stem(8)
    lr_i = base_lr * decay**i
    """
    groups = []
    name_to_layer = {
        'stem': 8, 'stage1': 7, 'down1': 7,
        'stage2': 6, 'down2': 6,
        'stage3': 5, 'down3': 5,
        'bottleneck': 4,
        'up3': 3, 'dec3': 3,
        'up2': 2, 'dec2': 2,
        'up1': 1, 'dec1': 1, 'final_up': 1, 'recon_head': 1,
        'cls_norm': 0, 'cls_fc': 0,
    }
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prefix = name.split('.')[0]
        layer = name_to_layer.get(prefix, 0)
        lr = base_lr * (decay ** layer)
        buckets.setdefault(lr, []).append(p)
    for lr, params in buckets.items():
        groups.append({'params': params, 'lr': lr})
    return groups


def _main_logits(out):
    """net(x) 在带 pair head 时返回 tuple, 取主分类 logits."""
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        pred = _main_logits(net(images)).argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
    return correct / total


def load_pretrained_encoder(net, ckpt_path):
    """加载 FCMAE 预训练权重, 仅编码器部分严格匹配."""
    state = torch.load(ckpt_path, map_location='cpu')
    encoder_keys = ['stem', 'stage1', 'down1', 'stage2', 'down2',
                    'stage3', 'down3', 'bottleneck']
    own = net.state_dict()
    loaded = 0
    for k, v in state.items():
        prefix = k.split('.')[0]
        if prefix in encoder_keys and k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    net.load_state_dict(own)
    print("Loaded %d encoder tensors from %s" % (loaded, ckpt_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--mix_prob", type=float, default=0.5)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--llrd", type=float, default=0.75,
                        help="Layer-wise LR decay; 1.0=关闭 (纯监督)")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--pretrain", default="output/pretrain.pth",
                        help="预训练权重路径, 'none' 跳过")
    parser.add_argument("--swa_start", type=int, default=0,
                        help="从第 N epoch 起累积 SWA, 0=禁用")
    parser.add_argument("--combine_val", action="store_true",
                        help="train+val 合并为训练集 (3100 张), 不再独立监控 val")
    parser.add_argument("--aux_pair_weight", type=float, default=0.0,
                        help="混淆字符辅助头损失权重; 0=禁用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_split", action="store_true",
                        help="改用随机 7:2:1 划分, 避免 51-55 漂移")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="随机划分 seed (固定划分本身可复现)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # 锁定确定性: 避免 cuDNN 非确定性导致"训练卡壳", 强正则化下尤其重要
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    if args.random_split:
        train_d, val_d, test_d, _, _ = load_random_split_data(
            ratio=(0.7, 0.2, 0.1), seed=args.split_seed)
        print("[random_split seed=%d] 7:2:1" % args.split_seed)
    elif args.combine_val:
        train_d, test_d, _, _ = load_combined_train_data()
        val_d = test_d  # 无独立 val, test 仅监控不参与 best_val 选择
    else:
        train_d, val_d, test_d, _, _ = load_split_data()
    train_loader = make_loader(train_d, batch=args.batch, train=True)
    val_loader = make_loader(val_d, batch=args.batch, train=False)
    test_loader = make_loader(test_d, batch=args.batch, train=False)

    n_pair = PAIR_NUM_CLASSES if args.aux_pair_weight > 0 else 0
    net = ConvNeXtV2UNet(num_classes=NUM_CLASSES,
                          drop_path=args.drop_path,
                          dropout=args.dropout,
                          num_pair_classes=n_pair).to(device)
    mode = "from-scratch" if args.pretrain == "none" else "finetune"
    print("%s | Params: %.2fM | drop_path=%.2f dropout=%.2f llrd=%.2f mix_p=%.2f" %
          (mode, count_params(net) / 1e6, args.drop_path, args.dropout,
           args.llrd, args.mix_prob))
    print("Train/Val/Test: %d/%d/%d" % (len(train_d), len(val_d), len(test_d)))

    if args.pretrain != "none" and os.path.exists(args.pretrain):
        load_pretrained_encoder(net, args.pretrain)
    else:
        print("WARN: pretrain weights not loaded (%s)" % args.pretrain)

    ema = ModelEMA(net, decay=args.ema)
    criterion = LabelSmoothing(NUM_CLASSES, smoothing=0.1)
    pair_criterion = LabelSmoothing(PAIR_NUM_CLASSES, smoothing=0.05) if n_pair > 0 else None
    groups = layer_lr_groups(net, args.lr, decay=args.llrd)
    opt = AdamW(groups, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    sched = LambdaLR(opt, lr_lambda=lambda e: warmup_cosine(e, args.warmup, args.epochs))

    log_path = "output/train_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_acc,val_ema,gap\n")

    best_val = 0.0
    gap_warn = 0
    swa_state = None
    swa_n = 0
    # 暖身阶段关掉 mixup/cutmix, 让模型先破壳学到基本表征 (前 10 epoch)
    warmup_no_mix = max(10, args.warmup * 2)
    for ep in range(args.epochs):
        net.train()
        total_loss = correct = seen = 0
        mix_prob_ep = 0.0 if ep < warmup_no_mix else args.mix_prob
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            do_mix = (args.mixup > 0 or args.cutmix > 0) and torch.rand(1).item() < mix_prob_ep
            if do_mix:
                images, y_a, y_b, lam = mixup_cutmix(images, labels, args.mixup, args.cutmix)
                out = net(images)
                logits = _main_logits(out)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                if n_pair > 0:
                    pa, pb = labels_to_pair(y_a), labels_to_pair(y_b)
                    pair_loss = lam * pair_criterion(out[1], pa) + (1 - lam) * pair_criterion(out[1], pb)
                    loss = loss + args.aux_pair_weight * pair_loss
            else:
                out = net(images)
                logits = _main_logits(out)
                loss = criterion(logits, labels)
                if n_pair > 0:
                    pair_loss = pair_criterion(out[1], labels_to_pair(labels))
                    loss = loss + args.aux_pair_weight * pair_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            ema.update(net)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        sched.step()

        train_acc = correct / seen
        val_acc = evaluate(net, val_loader, device)
        val_ema = evaluate(ema.ema, val_loader, device)
        gap = train_acc - val_ema
        avg_loss = total_loss / len(train_loader)

        if gap > 0.15:
            gap_warn += 1
        else:
            gap_warn = 0
        warn = " OVERFIT!" if gap_warn >= 5 else ""

        if val_ema > best_val:
            best_val = val_ema
            torch.save(ema.ema.state_dict(), "output/finetune.pth")

        # SWA: 累积 EMA 权重 (running mean)
        if args.swa_start > 0 and (ep + 1) >= args.swa_start:
            swa_n += 1
            cur = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
            if swa_state is None:
                swa_state = cur
            else:
                for k in swa_state:
                    if swa_state[k].dtype.is_floating_point:
                        swa_state[k] = swa_state[k] + (cur[k] - swa_state[k]) / swa_n
                    else:
                        swa_state[k] = cur[k]

        done = ep + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        print("\r  Ep%3d/%d [%s] loss=%.3f tr=%.3f val=%.3f val_ema=%.3f gap=%.3f%s" %
              (done, args.epochs, bar, avg_loss, train_acc, val_acc, val_ema, gap, warn),
              end="", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                    (done, avg_loss, train_acc, val_acc, val_ema, gap))
    print()
    print("Best val EMA: %.4f -> output/finetune.pth" % best_val)

    if swa_state is not None:
        torch.save(swa_state, "output/swa.pth")
        print("SWA averaged over %d epochs -> output/swa.pth" % swa_n)
        ema.ema.load_state_dict(swa_state)
        swa_val = evaluate(ema.ema, val_loader, device)
        swa_test = evaluate(ema.ema, test_loader, device)
        print("SWA val: %.4f | SWA test: %.4f" % (swa_val, swa_test))

    # 最终 test (best EMA, 无 TTA)
    ema.ema.load_state_dict(torch.load("output/finetune.pth", map_location=device))
    test_acc = evaluate(ema.ema, test_loader, device)
    print("Best-EMA test acc (no TTA): %.4f" % test_acc)


if __name__ == "__main__":
    main()
