"""折模型评估脚本, 支持两种模式:
  [留出集 bagging 集成] (默认, 需 --holdout_test 训练的 ckpt)
    各折模型在独立的留出集(写手 051-055)上做温度标定+加权融合, 评估无泄漏泛化。
    解决三个融合问题: 数据泄漏/概率尺度不一致/等权拖累。

  [全量 CV 评估] (--cv_eval, 对应不带 --holdout_test 训练的 ckpt)
    每折模型只预测自己的验证折, 汇聚全部样本预测后算完整指标。
    这是课设最常用的评估协议——全量数据五折交叉验证, 无留出集。
    注意: CV 模式下不做温度标定和加权(每样本只被预测一次)。

用法 (留出集 bagging):
  python ensemble.py --member "arch=output/*.pth"

用法 (全量 CV):
  python ensemble.py --cv_eval --member "arch=output/*.pth"

前提: 留出集模式需 --holdout_test 训练; CV 模式需不带 --holdout_test 训练;
      --n_folds/--seed 必须与训练时一致。
"""
import argparse
import csv
import glob
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import CharDataset, load_holdout_data, load_kfold_splits
from evaluate_cv import _build_net, _main_logits
from project_constants import BATCH_SIZE, DEVICE, CONFUSABLE_PAIRS
from tta import TTA_VIEWS, _translate
from metrics import classification_metrics


# ---------------------------------------------------------------- logits 收集
@torch.no_grad()
def _collect_logits(net, loader, tta=False):
    """返回 (logits[N,C], labels[N])。tta=True 时对 9 视图 logit 求平均 (标定前)。"""
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        if tta:
            acc = None
            for dx, dy in TTA_VIEWS:
                v = _translate(images, dx, dy) if (dx or dy) else images
                lg = _main_logits(net(v))
                acc = lg if acc is None else acc + lg
            logits = acc / len(TTA_VIEWS)
        else:
            logits = _main_logits(net(images))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ---------------------------------------------------------------- 温度标定
def _fit_temperature(logits, labels):
    """在验证折 logits 上最小化 NLL 拟合标量温度 T (LBFGS)。返回 clamp 到 [0.5, 5] 的 T。"""
    t = torch.nn.Parameter(torch.ones(1))
    nll = torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = nll(logits / t.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(0.5, 5.0).item())


# ---------------------------------------------------------------- 工具
def _collect_ckpts(pattern):
    """返回 [(fold_idx, path), ...]。保留同一折下的多个文件 (Snapshot Ensemble 每折多快照),
    fold_idx 取文件名里的 _f(N)_, 用于在该折的 val 上做温度标定。"""
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"_f(\d+)_", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return out


def _acc(probs, labels):
    return (probs.argmax(1) == labels).float().mean().item()


def _case_insensitive_acc(probs, labels, i2l):
    """大小写不敏感准确率: Cc/Oo/Ss 等天生歧义对折叠后再判对错 (准确率天花板参考)。"""
    pred = probs.argmax(1)
    correct = sum(i2l[int(p)].lower() == i2l[int(t)].lower()
                  for p, t in zip(pred, labels))
    return correct / len(labels)


def _pair_report(probs, labels, i2l):
    pred = probs.argmax(1)
    lines = []
    for a, b in CONFUSABLE_PAIRS:
        c = t = 0
        for p, y in zip(pred, labels):
            tl, pl = i2l[int(y)], i2l[int(p)]
            if tl in (a, b) and pl in (a, b):
                t += 1
                c += int(pl == tl)
        if t:
            lines.append("  %s vs %s: %.3f (%d/%d)" % (a, b, c / t, c, t))
    return "\n".join(lines)



# ---------------------------------------------------------------- 全量 CV 评估
def _run_cv_eval(args):
    """全量五折 CV 评估, 支持单架构和多架构异构集成.

    单架构 (--member 一次): 每折模型预测自己的验证折 → 汇聚全部预测 → 完整指标.
    多架构 (--member 多次): 每架构各自产出 OOF 预测 → 按样本对齐 → 概率平均 → 完整指标.
    同架构同折的多个 ckpt (快照集成) 先在该折内平均.
    """
    import numpy as np
    folds, all_labels, l2i = load_kfold_splits(n_splits=args.n_folds, random_state=args.seed)
    i2l = {v: k for k, v in l2i.items()}
    num_classes = len(all_labels)
    tta = not args.no_tta

    # Step 1: 每个架构产出 OOF 预测 (样本顺序与数据集一致)
    arch_oof_probs = []  # list of (arch_name, probs[N,C])
    arch_fold_accs = []  # list of (arch_name, [fold_accs])

    for spec in args.member:
        if "=" not in spec:
            raise ValueError('--member 需为 "arch=glob" 格式, 收到: %s' % spec)
        arch, pattern = spec.split("=", 1)
        ckpts = _collect_ckpts(pattern)
        if not ckpts:
            raise FileNotFoundError("no ckpt matched %s" % pattern)
        print("\n[成员] arch=%s | 匹配 %d 个折 ckpt" % (arch, len(ckpts)))

        # 按折分组 (同折多个 ckpt = 快照, 先平均)
        fold_ckpts = {}
        for fi, ckpt in sorted(ckpts):
            fold_ckpts.setdefault(fi, []).append(ckpt)

        fold_probs = {}  # fi -> probs tensor
        fold_accs_list = []

        for fi in sorted(fold_ckpts.keys()):
            _, val_data = folds[fi]
            val_loader = DataLoader(CharDataset(val_data, train=False), BATCH_SIZE,
                                    shuffle=False, num_workers=0)
            snap_probs = []
            snap_labels = None
            for ckpt in fold_ckpts[fi]:
                state = torch.load(ckpt, map_location=DEVICE)
                net = _build_net(state, arch)
                logits, labels = _collect_logits(net, val_loader, tta=tta)
                snap_probs.append(F.softmax(logits, dim=1))
                snap_labels = labels
                del net
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            # 同折多快照 → 平均概率
            if len(snap_probs) == 1:
                probs = snap_probs[0]
            else:
                probs = torch.stack(snap_probs).mean(dim=0)
            acc = _acc(probs, snap_labels)
            fold_accs_list.append(acc)
            fold_probs[fi] = probs
            n_ckpt = len(fold_ckpts[fi])
            print("  fold %d (n=%d): %d ckpt → val_acc=%.4f %s"
                  % (fi, len(snap_labels), n_ckpt, acc, "(TTA)" if tta else ""))

        # 按数据集原始顺序拼接 OOF 预测
        oof_probs = []
        for fi in range(args.n_folds):
            if fi in fold_probs:
                oof_probs.append(fold_probs[fi])
        oof_probs = torch.cat(oof_probs)
        arch_oof_probs.append((arch, oof_probs))
        arch_fold_accs.append((arch, fold_accs_list, np.mean(fold_accs_list)))

    # Step 2: 单架构报告 (各自 CV 成绩)
    print()
    print("=" * 78)
    print("全量 %d 折 CV 评估" % args.n_folds)
    for arch, accs, mean_acc in arch_fold_accs:
        print("  %s: fold accs=%s  →  mean=%.4f ± %.4f"
              % (arch.split(":")[0],
                 " / ".join("%.4f" % a for a in accs),
                 mean_acc, np.std(accs)))

    # Step 3: 多架构集成 (概率平均)
    if len(arch_oof_probs) == 1:
        # 单架构: 直接用 OOF
        final_probs = arch_oof_probs[0][1]
    else:
        # 多架构: 验证标签对齐后平均概率
        # (同一 seed 的 fold 划分保证各架构验证集完全一致, 所以直接平均)
        stacked = torch.stack([p for _, p in arch_oof_probs])
        final_probs = stacked.mean(dim=0)
        print("  集成方式: %d 架构 OOF 概率平均" % len(arch_oof_probs))

    # Step 4: 最终指标
    # 重新获取标签 (从 folds 拼接, 保证样本顺序与 oof_probs 一致)
    all_lbls = []
    for fi in range(args.n_folds):
        _, val_data = folds[fi]
        for _, lbl in val_data:
            all_lbls.append(lbl)
    final_lbls = torch.tensor(all_lbls)
    print("  汇聚 n=%d 样本, %d 类" % (len(final_lbls), num_classes))
    print("=" * 78)

    met, report = classification_metrics(final_probs, final_lbls, i2l, worst_k=10)
    print(report)
    print("混淆对:")
    print(_pair_report(final_probs, final_lbls, i2l))
    print("=" * 78)

    if args.results_csv:
        cols = ["tag", "members", "accuracy", "balanced_accuracy",
                "macro_f1", "macro_auc", "case_insensitive_accuracy", "tta"]
        row = {
            "tag": args.tag or ("+".join(a.split("=")[0] for a in args.member)),
            "members": len(args.member),
            "accuracy": "%.4f" % met["accuracy"],
            "balanced_accuracy": "%.4f" % met["balanced_accuracy"],
            "macro_f1": "%.4f" % met["macro_f1"],
            "macro_auc": "%.4f" % met["macro_auc"],
            "case_insensitive_accuracy": ("%.4f" % met["case_insensitive_accuracy"]
                                          if met["case_insensitive_accuracy"] is not None else ""),
            "tta": 0 if args.no_tta else 1,
        }
        import csv
        exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print("✔ 指标已追加到 %s (tag=%s)" % (args.results_csv, row["tag"]))

# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member", action="append", required=True,
                   help='集成成员, 格式 "arch=glob模式" (arch 可为 transfer:resnet50:160), 可重复')
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--writer_min", type=int, default=51,
                   help="留出集起始写手编号 (默认 51 = 051-055)")
    p.add_argument("--no_tta", action="store_true", help="跳过 9 视图 TTA")
    p.add_argument("--tag", type=str, default="",
                   help="本次配置标签 (写入 --results_csv 那一行, 如 'femto' / 'adapter_full')")
    p.add_argument("--results_csv", type=str, default="",
                   help="把本次最终指标追加到该 CSV (供 collect_results.py 汇总成表)")
    p.add_argument("--cv_eval", action="store_true",
                   help="全量 CV 评估模式: 每折模型预测自己的验证折, 汇聚后算指标 (无需留出集)")
    p.add_argument("--weight_power", type=float, default=1.0,
                   help="验证准确率加权指数 (1=线性; 越大越偏向强模型; 0=等权)")
    args = p.parse_args()

    if args.cv_eval:
        _run_cv_eval(args)
        return

    # 干净留出集 (写手 >=51) + 与训练一致的折划分 (写手 <51), 用于各折温度标定/权重
    holdout, all_labels, l2i = load_holdout_data(writer_min=args.writer_min)
    i2l = {v: k for k, v in l2i.items()}
    folds, _, _ = load_kfold_splits(n_splits=args.n_folds, random_state=args.seed,
                                    exclude_writer_min=args.writer_min)
    hold_loader = DataLoader(CharDataset(holdout, train=False), BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print("=" * 78)
    print("Clean-holdout bagging ensemble | holdout=写手%03d-055 (n=%d)"
          % (args.writer_min, len(holdout)))
    print("=" * 78)

    # 累加器: 留出集上 [N, C] 概率
    n = len(holdout)
    num_classes = len(all_labels)
    sum_naive = torch.zeros(n, num_classes)   # 等权 + 未标定
    sum_cal = torch.zeros(n, num_classes)     # 加权 + 温度标定
    sum_naive_t = torch.zeros(n, num_classes)
    sum_cal_t = torch.zeros(n, num_classes)
    total_w = 0.0
    hold_labels = None

    for spec in args.member:
        if "=" not in spec:
            raise ValueError('--member 需为 "arch=glob" 格式, 收到: %s' % spec)
        arch, pattern = spec.split("=", 1)
        ckpts = _collect_ckpts(pattern)
        if not ckpts:
            raise FileNotFoundError("no ckpt matched %s" % pattern)
        print("\n[成员] arch=%s | 匹配 %d 个折 ckpt" % (arch, len(ckpts)))

        for fi, ckpt in sorted(ckpts):
            state = torch.load(ckpt, map_location=DEVICE)
            net = _build_net(state, arch)

            # 验证折 (clean): 拟合温度 + 计算权重
            _, val_data = folds[fi]
            val_loader = DataLoader(CharDataset(val_data, train=False), BATCH_SIZE,
                                    shuffle=False, num_workers=0)
            v_logits, v_labels = _collect_logits(net, val_loader, tta=False)
            temp = _fit_temperature(v_logits, v_labels)
            val_acc = _acc(v_logits, v_labels)
            w = val_acc ** args.weight_power

            # 留出集: single + (可选) TTA, 标定前后各取一份
            h_logits, h_labels = _collect_logits(net, hold_loader, tta=False)
            if hold_labels is None:
                hold_labels = h_labels
            sum_naive += F.softmax(h_logits, dim=1)
            sum_cal += w * F.softmax(h_logits / temp, dim=1)
            if not args.no_tta:
                ht_logits, _ = _collect_logits(net, hold_loader, tta=True)
                sum_naive_t += F.softmax(ht_logits, dim=1)
                sum_cal_t += w * F.softmax(ht_logits / temp, dim=1)
            total_w += w

            h_acc = _acc(F.softmax(h_logits, dim=1), h_labels)
            print("  fold %d: val=%.4f T=%.2f w=%.3f | holdout(单模型)=%.4f  [%s]"
                  % (fi, val_acc, temp, w, h_acc, os.path.basename(ckpt)))
            del net
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    lbl = hold_labels
    print("\n" + "=" * 78)
    print("留出集集成结果 (n=%d, %d 类)" % (n, num_classes))
    print("=" * 78)
    print("等权 + 未标定 (single)        : %.4f" % _acc(sum_naive, lbl))
    print("加权 + 温度标定 (single)      : %.4f" % _acc(sum_cal, lbl))
    if not args.no_tta:
        print("等权 + 未标定 (TTA)           : %.4f" % _acc(sum_naive_t, lbl))
        print("加权 + 温度标定 (TTA)  ★最终  : %.4f" % _acc(sum_cal_t, lbl))
    best = sum_cal_t if not args.no_tta else sum_cal
    # 完整指标 (macro-F1 / macro-AUC / balanced acc / 最差类 / 大小写不敏感)
    met, report = classification_metrics(best, lbl, i2l, worst_k=10)
    print(report)
    print("混淆对 (最终集成):")
    print(_pair_report(best, lbl, i2l))
    print("=" * 78)

    # 追加一行到结果 CSV (供 collect_results.py 汇总)
    if args.results_csv:
        cols = ["tag", "members", "accuracy", "balanced_accuracy",
                "macro_f1", "macro_auc", "case_insensitive_accuracy", "tta"]
        row = {
            "tag": args.tag or ("+".join(a.split("=")[0] for a in args.member)),
            "members": len(args.member),
            "accuracy": "%.4f" % met["accuracy"],
            "balanced_accuracy": "%.4f" % met["balanced_accuracy"],
            "macro_f1": "%.4f" % met["macro_f1"],
            "macro_auc": "%.4f" % met["macro_auc"],
            "case_insensitive_accuracy": ("%.4f" % met["case_insensitive_accuracy"]
                                          if met["case_insensitive_accuracy"] is not None else ""),
            "tta": 0 if args.no_tta else 1,
        }
        exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print("✔ 指标已追加到 %s (tag=%s)" % (args.results_csv, row["tag"]))


if __name__ == "__main__":
    main()
