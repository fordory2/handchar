"""HybridHandCharNet 联合训练 (5-fold CV + MixUp + 强正则).

- 数据策略: 62 类 × 55 张全集做 5-fold StratifiedCV, 每 fold ~2728 train / 682 val
- 损失: Focal (γ=2.0, smoothing=0.1) + MixUp 双标签线性组合
- 正则: dropout↑(0.146→0.25), wd↑(1e-4→5e-4), random erasing, MixUp α=0.2, 50% batch
"""
import argparse
import copy
import datetime
import time
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import CharDataset, collate_with_augment, load_kfold_splits
from losses import FocalLoss, sigreg_loss
from models import (ConvNeXtV2Char, HybridHandCharNet, HybridHandCharNetCnxBypassA,
                    HybridHandCharNetCnxBypassB, HybridHandCharNetCnxStem,
                    HybridHandCharNetR18Stem, ResNet18Pretrained, ResNet50UNetChar,
                    TransferBackbone, TransferBackboneMS, TransferBackboneAdapter)
from project_constants import (BATCH_SIZE, CONFUSABLE_PAIRS, DEVICE, LEARNING_RATE,
                               NUM_CLASSES, TRAIN_EPOCHS)
from training_utils import compute_pair_accuracy


class ModelEMA:
    """EMA 权重. short training (100 ep) 用 decay=0.99, val 评估走 net/ema 双路 max."""
    def __init__(self, model, decay=0.99):
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


@torch.no_grad()
def _eval_acc(net, loader, device):
    net.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        pred = net(images)[0].argmax(1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def _build_net(args):
    if args.arch == "transfer_ms":
        return TransferBackboneMS(
            model_name=args.transfer_model, num_classes=NUM_CLASSES, pretrained=True,
            input_size=args.transfer_input, ms_stages=args.transfer_ms_stages,
        ).to(DEVICE)
    if args.arch == "disentangled":
        from models import DisentangledNet
        return DisentangledNet(
            model_name=args.transfer_model, num_classes=NUM_CLASSES, pretrained=True,
            input_size=args.transfer_input,
            shape_dim=args.shape_dim, geo_dim=args.geo_dim,
            ista_steps=args.ista_steps,
        ).to(DEVICE)
    if args.arch == "transfer_adapter":
        return TransferBackboneAdapter(
            model_name=args.transfer_model, num_classes=NUM_CLASSES, pretrained=True,
            input_size=args.transfer_input, adapt_dim=args.adapt_dim,
            branches=tuple(b for b in args.adapter_branches.split(",") if b),
            intensity_mode=args.intensity_mode, weber_norm=args.weber_norm,
            use_intensity=not args.no_intensity, head=args.adapter_head,
            arcface_scale=args.arcface_scale, code_length=args.ecoc_length,
            recon=args.recon_lambda > 0.0,
        ).to(DEVICE)
    if args.arch == "transfer":
        return TransferBackbone(
            model_name=args.transfer_model,
            num_classes=NUM_CLASSES,
            pretrained=True,
            input_size=args.transfer_input,
            gray_to_3ch=True,
            imagenet_norm=True,
        ).to(DEVICE)
    if args.arch == "convnextv2p":
        return ConvNeXtV2Char(
            num_classes=NUM_CLASSES,
            pretrained=True,
            input_size=args.convnextv2p_input,
        ).to(DEVICE)
    if args.arch == "resnet18p":
        return ResNet18Pretrained(
            num_classes=NUM_CLASSES,
            pretrained=True,
            input_size=args.resnet18p_input,
        ).to(DEVICE)
    if args.arch == "hybrid_r18stem":
        return HybridHandCharNetR18Stem(
            num_classes=NUM_CLASSES,
            pretrained=True,
            use_rnn=not args.no_rnn,
            rnn_cell=args.rnn_cell,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            rnn_proj_dim=args.rnn_proj_dim,
        ).to(DEVICE)
    if args.arch == "hybrid_cnxstem":
        return HybridHandCharNetCnxStem(
            num_classes=NUM_CLASSES,
            pretrained=True,
            use_rnn=not args.no_rnn,
            rnn_cell=args.rnn_cell,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            rnn_proj_dim=args.rnn_proj_dim,
        ).to(DEVICE)
    if args.arch == "hybrid_cnxbypassA":
        return HybridHandCharNetCnxBypassA(
            num_classes=NUM_CLASSES,
            pretrained=True,
            input_size=args.convnextv2p_input,
        ).to(DEVICE)
    if args.arch == "hybrid_cnxbypassB":
        return HybridHandCharNetCnxBypassB(
            num_classes=NUM_CLASSES,
            pretrained=True,
            input_size=args.convnextv2p_input,
        ).to(DEVICE)
    if args.arch == "resnet50_unet":
        return ResNet50UNetChar(
            num_classes=NUM_CLASSES,
            use_rnn=not args.no_rnn,
            rnn_hidden=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            rnn_proj_dim=args.rnn_proj_dim,
        ).to(DEVICE)
    return HybridHandCharNet(
        num_classes=NUM_CLASSES,
        use_rnn=not args.no_rnn,
        rnn_cell=args.rnn_cell,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_proj_dim=args.rnn_proj_dim,
    ).to(DEVICE)


def _make_optimizer(net, base_lr, args):
    """判别式学习率: net 支持 param_groups 且 backbone_lr_mult!=1 → 分两组 (backbone 小 lr);
    否则退回单组 (保持原行为)。"""
    if hasattr(net, "param_groups") and args.backbone_lr_mult != 1.0:
        return AdamW(net.param_groups(base_lr, args.backbone_lr_mult, weight_decay=5e-4))
    return AdamW([p for p in net.parameters() if p.requires_grad],
                 lr=base_lr, weight_decay=5e-4)


def train_one_fold(fold_idx, train_data, val_data, args, i2l, pair_idx=None):
    """单 fold 训练: 复用 MixUp collate + EMA dual-path + cosine LR."""
    train_ds = CharDataset(train_data, train=True)
    val_ds = CharDataset(val_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net = _build_net(args)
    if args.pretrained:
        encoder_state = torch.load(args.pretrained, map_location=DEVICE)
        load_state = {k: v for k, v in encoder_state.items()
                      if not k.startswith(("head_", "lstm", "lstm_proj"))}
        missing, unexpected = net.load_state_dict(load_state, strict=False)
        print("  Loaded MAE: %d keys (missing=%d, unexpected=%d)" %
              (len(load_state), len(missing), len(unexpected)))

    # 阶段 1: 冻 stem 让 tail 先适应数据, 避免 ImageNet 权重被强增强污染
    stem_frozen = False
    if args.freeze_stem_epochs > 0 and hasattr(net, 'stem'):
        for p in net.stem.parameters():
            p.requires_grad_(False)
        stem_frozen = True
        n_frozen = sum(p.numel() for p in net.stem.parameters())
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("  [freeze stem] frozen=%d trainable=%d (first %d ep)" %
              (n_frozen, n_train, args.freeze_stem_epochs))

    # 渐进解冻: 前 N ep 冻结整个预训练 backbone, 只训新建 head (迁移微调防早期污染)
    backbone_frozen = False
    if args.freeze_backbone_epochs > 0 and hasattr(net, "freeze_backbone"):
        net.freeze_backbone()
        backbone_frozen = True
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("  [freeze backbone] 仅训 head trainable=%d (first %d ep)" %
              (n_train, args.freeze_backbone_epochs))

    # 混淆矩阵感知软标签: 把 CONFUSABLE_PAIRS 的字符对映射成类别索引对
    l2i = {v: k for k, v in i2l.items()}
    pair_idx = [(l2i[a], l2i[b]) for a, b in CONFUSABLE_PAIRS
                if a in l2i and b in l2i]
    focal_crit = FocalLoss(gamma=2.0, smoothing=0.1, num_classes=NUM_CLASSES,
                           pair_smoothing=args.pair_smoothing,
                           confusable_pairs_idx=pair_idx,
                           margin=args.cos_margin, margin_scale=args.arcface_scale)
    # wd: 1e-4 → 5e-4 (强正则)
    optimizer = _make_optimizer(net, LEARNING_RATE, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Snapshot Ensemble: 余弦重启, 每个周期末 (LR 低谷) 存一个快照
    snapshots = []
    snap_cycle_len = 0
    if args.snapshot_cycles > 0:
        snap_cycle_len = max(1, args.epochs // args.snapshot_cycles)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=snap_cycle_len)
        print("  [snapshot] 余弦重启 cycles=%d cycle_len=%d (每周期末存快照)" %
              (args.snapshot_cycles, snap_cycle_len))
    ema = ModelEMA(net, decay=0.99)
    # SWA: 训练后段权重等权平均, 更接近 loss basin center. swa_start=0 表示禁用
    swa_model = None
    swa_scheduler = None
    if args.swa_start > 0 and args.snapshot_cycles == 0:
        swa_model = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr,
                              anneal_epochs=5, anneal_strategy='linear')

    best_val_acc = 0.0
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    best_tag = 'net'
    for epoch in range(args.epochs):
        # 渐进解冻: 到 freeze_backbone_epochs 解冻整个 backbone, 用判别式 lr 重建 optim
        if backbone_frozen and epoch == args.freeze_backbone_epochs:
            net.unfreeze_backbone()
            backbone_frozen = False
            optimizer = _make_optimizer(net, args.unfreeze_lr, args)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            if swa_model is not None:
                swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr,
                                      anneal_epochs=5, anneal_strategy="linear")
            print("\n  [unfreeze backbone at ep %d] base_lr=%g (backbone×%g), T_max=%d" %
                  (epoch, args.unfreeze_lr, args.backbone_lr_mult, args.epochs - epoch))

        # 阶段 2: 到 freeze_stem_epochs 解冻 stem, 重建 optim + cosine + swa_sched
        if stem_frozen and epoch == args.freeze_stem_epochs:
            for p in net.stem.parameters():
                p.requires_grad_(True)
            stem_frozen = False
            optimizer = AdamW(net.parameters(), lr=args.unfreeze_lr, weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            if swa_model is not None:
                swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr,
                                      anneal_epochs=5, anneal_strategy='linear')
            print("\n  [unfreeze stem at ep %d] lr=%g, cosine T_max=%d" %
                  (epoch, args.unfreeze_lr, args.epochs - epoch))

        net.train()
        total_main = 0.0
        correct = total = 0
        for images, y_a, y_b, lam in train_loader:
            # MixUp 4 元组: lam=1.0 时 y_b=y_a, 等价无 MixUp
            images = images.to(DEVICE)
            y_a = y_a.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            out = net(images)
            if args.arch == "disentangled":
                main_logits, unified, logits_shape = out
                # focal_crit 用 main_logits, 残差损失用 logits_shape
                loss_main = lam * focal_crit(main_logits, y_a) + (1.0 - lam) * focal_crit(main_logits, y_b)
                from losses import residual_loss
                # 取 y_a 为主目标 (MixUp 时 y_a 权重≥0.5)
                loss_res, res_terms = residual_loss(
                    logits_shape, main_logits, main_logits - logits_shape, y_a,
                    confusable_pairs_idx=pair_idx,
                    lambda_sparse=args.sparse_lambda, lambda_diff=args.diff_lambda)
                loss = loss_main + loss_res
                decoder_out = None
            else:
                main_logits, unified, decoder_out = out
                loss = lam * focal_crit(main_logits, y_a) + (1.0 - lam) * focal_crit(main_logits, y_b)
            # SIGReg (LeJEPA): 强制 unified 特征空间高斯; λ=0 时跳过
            if args.sigreg_lambda > 0.0:
                loss = loss + args.sigreg_lambda * sigreg_loss(unified, n_projections=args.sigreg_proj)
            # 自监督重建正则: 重建灰度输入 (生成式辅助任务, 限定数据集内增强特征学习)
            if args.recon_lambda > 0.0 and decoder_out is not None:
                target = nn.functional.interpolate(images, size=decoder_out.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                loss = loss + args.recon_lambda * nn.functional.mse_loss(decoder_out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            ema.update(net)
            total_main += loss.item()
            # train acc 按 y_a 算 (近似, MixUp 时 lam 偏 1 多数)
            correct += (main_logits.argmax(1) == y_a).sum().item()
            total += y_a.size(0)

        # LR 调度: SWA 阶段 (epoch >= swa_start) 走 SWALR 固定/退火; 否则 cosine
        if swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()
        # Snapshot: 周期末 (LR 退到低谷) 抓一张当集成成员
        if snap_cycle_len and (epoch + 1) % snap_cycle_len == 0:
            snapshots.append({k: v.cpu().clone() for k, v in net.state_dict().items()})

        train_acc = correct / total
        val_net = _eval_acc(net, val_loader, DEVICE)
        val_ema = _eval_acc(ema.ema, val_loader, DEVICE)
        # SWA: 累积阶段才评估; update_bn 单 batch 仅供训练监控, 最终 update_bn 在循环外做
        val_swa = -1.0
        if swa_model is not None and epoch >= args.swa_start:
            swa_model.eval()
            val_swa = _eval_acc(swa_model, val_loader, DEVICE)
        candidates = [('net', val_net), ('ema', val_ema)]
        if val_swa >= 0:
            candidates.append(('swa', val_swa))
        tag, val_acc = max(candidates, key=lambda kv: kv[1])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tag = tag
            if tag == 'ema':
                src = ema.ema
            elif tag == 'swa':
                src = swa_model.module   # AveragedModel 把原 net 包在 .module 下, 取出来取 state_dict 才兼容
            else:
                src = net
            best_state = {k: v.cpu().clone() for k, v in src.state_dict().items()}
        done = epoch + 1
        bar = "#" * (done * 20 // args.epochs) + "-" * (20 - done * 20 // args.epochs)
        swa_str = "  swa=%.3f" % val_swa if val_swa >= 0 else ""
        print("\r  F%d Ep%2d/%d [%s] loss=%.3f tr=%.3f net=%.3f ema=%.3f%s best=%.4f(%s)" %
              (fold_idx, done, args.epochs, bar,
               total_main / len(train_loader),
               train_acc, val_net, val_ema, swa_str, best_val_acc, best_tag),
              end="", flush=True)
    print()

    # SWA: 训练完用 train loader 重算 BN running stats (collate 输出 4 元组, 取 [0])
    # 仅当 SWA 真正累积过 (epoch 到过 swa_start) 才做; 否则 (如 epochs<swa_start 的短训练)
    # 跳过, 避免对未累积的空 SWA 模型 update_bn 产生垃圾数。
    if swa_model is not None and int(swa_model.n_averaged) > 0:
        def _bn_iter():
            for batch in train_loader:
                yield batch[0]
        update_bn(_bn_iter(), swa_model, device=DEVICE)
        val_swa_final = _eval_acc(swa_model, val_loader, DEVICE)
        print("  SWA post-update_bn: val=%.4f" % val_swa_final)
        if val_swa_final > best_val_acc:
            best_val_acc = val_swa_final
            best_tag = 'swa_final'
            best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}

    # 用 best 状态做 val + pair 评估
    net.load_state_dict(best_state)
    final_val_acc = _eval_acc(net, val_loader, DEVICE)

    class _MainHeadWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)[0]

    pair_acc = compute_pair_accuracy(_MainHeadWrapper(net), val_loader, i2l)
    return best_state, final_val_acc, pair_acc, snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--arch", type=str, default="hybrid",
                        choices=["hybrid", "resnet50_unet", "resnet18p", "convnextv2p",
                                 "hybrid_r18stem", "hybrid_cnxstem",
                                 "hybrid_cnxbypassA", "hybrid_cnxbypassB",
                                 "transfer", "transfer_ms", "transfer_adapter", "disentangled"],
                        help="hybrid=HybridHandCharNet; resnet50_unet=ResNet50+UNet+LSTM; "
                             "resnet18p=ImageNet 预训练 ResNet18 + 1ch + 前层 Dropout2d; "
                             "convnextv2p=FCMAE+IN1k 预训练 ConvNeXtV2 atto + 1ch; "
                             "hybrid_r18stem=Hybrid 全保留, stem 换预训练 R18 conv1+layer1 (压 64→32); "
                             "hybrid_cnxstem=Hybrid 全保留, stem 换预训练 ConvNeXtV2 stem+s0+s1, "
                             "三分支主路化吃 80ch 全量 (血统对齐, 无信息瓶颈); "
                             "hybrid_cnxbypassA=完整 ConvNeXtV2 backbone + 三分支并联 stage1 concat 融合; "
                             "hybrid_cnxbypassB=完整 ConvNeXtV2 backbone + 三分支残差 add (1×1 init=0)")
    parser.add_argument("--resnet18p_input", type=int, default=128,
                        help="resnet18p 上采样目标分辨率 (默认 128; 128 折中 64×48→ResNet 224 spec)")
    parser.add_argument("--convnextv2p_input", type=int, default=96,
                        help="convnextv2p 上采样目标分辨率 (默认 96; atto stem stride=4 适配)")
    parser.add_argument("--transfer_model", type=str, default="convnextv2_femto",
                        help="arch=transfer 时的 timm 骨干名 (resnet50/convnext_tiny/tf_efficientnet_b0 等; 见 dl-vision-backbones-ssl 选型)")
    parser.add_argument("--transfer_input", type=int, default=160,
                        help="arch=transfer 上采样目标分辨率 (默认 160; 更接近 ImageNet 原生)")
    parser.add_argument("--transfer_ms_stages", type=int, default=2,
                        help="transfer_ms: 聚合最后 N 个 stage 的多尺度特征 (默认 2)")
    parser.add_argument("--adapt_dim", type=int, default=64,
                        help="transfer_adapter: 三分支适配工作维度 (默认 64)")
    parser.add_argument("--no_intensity", action="store_true",
                        help="transfer_adapter: 关掉强度统计分支 (做消融用)")
    parser.add_argument("--adapter_branches", type=str, default="spatial,channel,multiscale",
                        help="transfer_adapter 特征分支 (逗号分隔): "
                             "spatial/channel/multiscale/gabor/morph 任意组合; 用于特征消融")
    parser.add_argument("--adapter_head", type=str, default="linear",
                        choices=["linear", "cosine", "ecoc", "hier"],
                        help="transfer_adapter 分类头: linear/cosine(度量)/ecoc(纠错输出编码)/hier(分层路由)")
    parser.add_argument("--intensity_mode", type=str, default="stats",
                        choices=["stats", "moment", "both", "none"],
                        help="强度特征: stats(std/max) / moment(强度加权几何矩,经典图像矩) / "
                             "both / none; 用于强度模块消融")
    parser.add_argument("--weber_norm", action="store_true",
                        help="强度路径加 Weber 除性归一化前端 (对墨色深浅鲁棒)")
    parser.add_argument("--cos_margin", type=float, default=0.0,
                        help="CosFace 加性间隔 (仅配 --adapter_head cosine 用; 真类 cos 减 m "
                             "把混淆类在超球面推开; 建议 0.1~0.3; 0=禁用)")
    parser.add_argument("--arcface_scale", type=float, default=30.0,
                        help="余弦头/间隔的缩放 scale (默认 30)")
    parser.add_argument("--ecoc_length", type=int, default=127,
                        help="ECOC 头码字长度 (head=ecoc 时; 评估须与训练一致, 建议保持默认)")
    parser.add_argument("--sparse_lambda", type=float, default=1e-4,
                        help="disentangled: L1(Δ) 稀疏惩罚权重 (默认 1e-4)")
    parser.add_argument("--diff_lambda", type=float, default=0.1,
                        help="disentangled: 歧义对 Δ 差异鼓励权重 (默认 0.1)")
    parser.add_argument("--shape_dim", type=int, default=192,
                        help="disentangled: 形状流特征维度 (默认 192)")
    parser.add_argument("--geo_dim", type=int, default=192,
                        help="disentangled: 结构流特征维度 (默认 192)")
    parser.add_argument("--ista_steps", type=int, default=2,
                        help="disentangled: ChannelISTA 迭代步数 (默认 2; 0=关)")
    parser.add_argument("--recon_lambda", type=float, default=0.0,
                        help="transfer_adapter 辅助重建解码器的损失权重 (自监督重建正则; "
                             "0=禁用; 建议 0.1~0.5)")
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0,
                        help="判别式学习率: 预训练 backbone 用 lr*该系数, 新建 head 用 lr "
                             "(默认 1.0=不区分; 迁移微调建议 0.1)")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="渐进解冻: 前 N ep 冻结整个预训练 backbone 只训 head, 之后解冻 "
                             "(需 net 实现 freeze_backbone/unfreeze_backbone; 0=禁用)")
    parser.add_argument("--snapshot_cycles", type=int, default=0,
                        help="Snapshot Ensemble (TextCaps 同款): >0 时用余弦重启, 把 epochs "
                             "均分成 N 个周期, 每周期低谷存一个快照当集成成员 (一次训练多成员); "
                             "0=禁用。开启后自动关 SWA 以免调度冲突。建议 epochs 为 cycles 整数倍")
    parser.add_argument("--pair_smoothing", type=float, default=0.0,
                        help="混淆矩阵感知软标签: 把这部分平滑质量定向分给混淆伙伴 "
                             "(o/O,s/S,1/l 等 CONFUSABLE_PAIRS); 0=退回普通均匀平滑。建议 0.05~0.1")
    parser.add_argument("--pretrained", type=str, default="",
                        help="MAE 预训练编码器路径 (output/pretrain_encoder.pth), 空=从头训")
    parser.add_argument("--no_rnn", action="store_true", help="禁用 RNN 分支")
    parser.add_argument("--rnn_cell", type=str, default="lstm", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--rnn_hidden", type=int, default=96)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_proj_dim", type=int, default=192)
    parser.add_argument("--n_folds", type=int, default=5, help="CV fold 数")
    parser.add_argument("--folds_to_run", type=str, default="",
                        help="逗号分隔 fold idx (如 '0,1'), 空 = 全跑")
    parser.add_argument("--seed", type=int, default=42, help="KFold 切分随机种子")
    parser.add_argument("--holdout_test", action="store_true",
                        help="干净留出协议: 写手 051-055 整块排除出五折, 只在 001-050 上训练; "
                             "训练完用 ensemble.py 在 051-055 上做无泄漏 bagging 集成评估")
    parser.add_argument("--sigreg_lambda", type=float, default=0.0,
                        help="SIGReg 高斯正则权重 (0=禁用; 论文严格版无 warmup 全程恒定; "
                             "实测 init focal=4 vs sigreg=0.008, 监督场景 λ=5 → 后期 sigreg 项 ≤ 40% focal)")
    parser.add_argument("--sigreg_proj", type=int, default=1024,
                        help="SIGReg 投影方向数 (论文默认 1024, 仅 λ>0 时生效)")
    parser.add_argument("--swa_start", type=int, default=75,
                        help="SWA 起始 epoch (0=禁用); 默认 75 = 100ep 的后 25%")
    parser.add_argument("--swa_lr", type=float, default=5e-4,
                        help="SWA 阶段固定 LR (默认 5e-4, AdamW base lr 的 ~1/2)")
    parser.add_argument("--freeze_stem_epochs", type=int, default=0,
                        help="前 N ep 冻结 net.stem (仅 hybrid_r18stem 有意义, 0=禁用); "
                             "防止 ImageNet 预训练 stem 被强增强污染")
    parser.add_argument("--unfreeze_lr", type=float, default=5e-4,
                        help="解冻 stem 后整体微调起始 lr (默认 5e-4, 进余下 cosine 衰减)")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    exclude_min = 51 if args.holdout_test else None
    folds, all_labels, l2i = load_kfold_splits(
        n_splits=args.n_folds, random_state=args.seed, exclude_writer_min=exclude_min)
    # 歧义对索引 (供 disentangled 残差损失用)
    _pair_idx = [(l2i[a], l2i[b]) for a, b in CONFUSABLE_PAIRS if a in l2i and b in l2i]
    i2l = {v: k for k, v in l2i.items()}

    # 报模型规格 (用 fold 0 临时建一个数 params)
    tmp = _build_net(args)
    n_params = sum(p.numel() for p in tmp.parameters())
    del tmp
    rnn_desc = "off" if args.no_rnn else "%s/h%d/L%d/d%d" % (
        args.rnn_cell, args.rnn_hidden, args.rnn_layers, args.rnn_proj_dim)
    print("=" * 70)
    print("Arch: %s | Params: %d | Epochs: %d | RNN: %s" %
          (args.arch, n_params, args.epochs, rnn_desc))
    print("CV: %d-fold StratifiedKFold (seed=%d) | %d train / %d val per fold" %
          (args.n_folds, args.seed, len(folds[0][0]), len(folds[0][1])))
    if args.holdout_test:
        print("Holdout: 写手 051-055 已排除, 仅 001-050 进 CV (留出集留给 ensemble.py)")
    sigreg_desc = ("off" if args.sigreg_lambda == 0.0
                   else "λ=%g, K=%d" % (args.sigreg_lambda, args.sigreg_proj))
    swa_desc = ("off" if args.swa_start == 0
                else "start=ep%d, lr=%g" % (args.swa_start, args.swa_lr))
    print("Reg: dropout=0.25, wd=5e-4, MixUp α=0.2 p=0.5, RandomErasing p=0.5, SIGReg %s" %
          sigreg_desc)
    print("SWA: %s" % swa_desc)
    print("=" * 70)

    if args.folds_to_run:
        run_set = {int(s) for s in args.folds_to_run.split(",")}
    else:
        run_set = set(range(args.n_folds))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_results = []  # list of (fold_idx, val_acc, ckpt_path)
    pair_accum = {}    # pair -> list of acc
    n_run = len([i for i in range(args.n_folds) if i in run_set])  # 实际要跑的折数
    fold_times = []
    for fold_idx, (tr, val) in enumerate(folds):
        if fold_idx not in run_set:
            continue
        print("\n[Fold %d/%d] train=%d val=%d" % (fold_idx, args.n_folds, len(tr), len(val)))
        _t_fold = time.time()
        best_state, val_acc, pair_acc, snapshots = train_one_fold(fold_idx, tr, val, args, i2l, _pair_idx)
        ckpt_path = "output/%s_cv%d_f%d_%s.pth" % (args.arch, args.n_folds, fold_idx, timestamp)
        torch.save(best_state, ckpt_path)
        for si, snap in enumerate(snapshots):
            snap_path = "output/%s_cv%d_f%d_s%d_%s.pth" % (
                args.arch, args.n_folds, fold_idx, si, timestamp)
            torch.save(snap, snap_path)
        if snapshots:
            print("  → 另存 %d 个 snapshot 集成成员" % len(snapshots))
        print("  → val=%.4f, saved %s" % (val_acc, ckpt_path))
        fold_times.append(time.time() - _t_fold)
        _done = len(fold_times)
        _remain = n_run - _done
        _avg = sum(fold_times) / _done
        print("  [T] fold time %.1f min | elapsed %.1f min | ETA ~%.1f min (%d folds left)" % (
            fold_times[-1] / 60, sum(fold_times) / 60, _avg * _remain / 60, _remain))
        for k, v in pair_acc.items():
            pair_accum.setdefault(k, []).append(v)
        fold_results.append((fold_idx, val_acc, ckpt_path))

    if len(fold_results) >= 2:
        accs = [r[1] for r in fold_results]
        mean = sum(accs) / len(accs)
        var = sum((a - mean) ** 2 for a in accs) / len(accs)
        std = var ** 0.5
        print("\n" + "=" * 70)
        print("CV Summary (%d folds run): val acc mean=%.4f ± %.4f" %
              (len(fold_results), mean, std))
        for fi, acc, p in fold_results:
            print("  fold %d: %.4f  (%s)" % (fi, acc, os.path.basename(p)))
        print("Confusable pairs (mean over folds):")
        for k in sorted(pair_accum):
            vs = pair_accum[k]
            print("  %s: %.3f ± %.3f  (n=%d)" %
                  (k, sum(vs) / len(vs), (sum((v - sum(vs) / len(vs)) ** 2 for v in vs) / len(vs)) ** 0.5, len(vs)))
        print("=" * 70)


if __name__ == "__main__":
    main()
