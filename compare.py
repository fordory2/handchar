"""五折交叉验证 + 消融对比"""

import datetime
import os
from collections import defaultdict

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from cli_utils import parse_training_args
from data_utils import CharDataset, collate_with_augment, load_split_data
from models import (
    CRNNCharNet,
    DenseNetChar,
    FPNCharNet,
    HandCharNet,
    HandCharNetCA,
    HandCharNetCBAM,
    HandCharNetECA,
    HandCharNetGroupSE,
    HandCharNetNoDirECA,
    HandCharNetNoDirHybrid,
    HandCharNetNoDirection,
    HandCharNetNoDirectionEca,
    HandCharNetNoMultiScale,
    HandCharNetNoSe,
    HandCharNetResGELU,
    HybridHandCharNet,
    RefinedHybridNet,
    ConvNeXtV2Char,
    MobileNetV4Char,
    ResNet18Char,
)
from project_constants import BATCH_SIZE, CONFUSABLE_PAIRS, DEVICE, NUM_CLASSES
from training_utils import compute_pair_accuracy, evaluate, evaluate_per_class, fit_best_model


class _HybridForCompare(nn.Module):
    """把 HybridHandCharNet 包成单输出的 model_factory(num_classes) 接口."""
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.hybrid = HybridHandCharNet(num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.hybrid(x)[0]


def hybrid_full_factory(num_classes):
    return _HybridForCompare(num_classes, use_rnn=True)


def hybrid_no_rnn_factory(num_classes):
    return _HybridForCompare(num_classes, use_rnn=False)


def hybrid_tiny_rnn_factory(num_classes):
    """轻量 BiGRU 版本: 减重抗过拟合, 保留 RNN 笔顺特征."""
    return _HybridForCompare(num_classes, use_rnn=True,
                              rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                              rnn_proj_dim=128)


def hybrid_transformer_factory(num_classes):
    """TransformerEncoder 替代 RNN: 自注意力建模笔顺位置间长依赖."""
    return _HybridForCompare(num_classes, use_rnn=True,
                              rnn_cell='transformer', rnn_layers=1,
                              rnn_proj_dim=128)


class _RefinedHybridForCompare(nn.Module):
    """RefinedHybridNet 包成单输出的 model_factory(num_classes) 接口."""
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__()
        self.refined = RefinedHybridNet(num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.refined(x)[0]


def refined_full_factory(num_classes):
    """RefinedHybrid + Full BiLSTM. RNN 接 bottleneck (CRNN 风格, 文献首选)."""
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='lstm', rnn_hidden=128, rnn_layers=2,
                                     rnn_proj_dim=256, rnn_attach_to='bottleneck')


def refined_no_rnn_factory(num_classes):
    return _RefinedHybridForCompare(num_classes, use_rnn=False)


def refined_tiny_factory(num_classes):
    """RefinedHybrid + TinyGRU. RNN 接 bottleneck."""
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='bottleneck')


def refined_trans_factory(num_classes):
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='transformer', rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='bottleneck')


def refined_grn_full_factory(num_classes):
    """RefinedHybrid + Full BiLSTM + GRN. 阶段 2 候选."""
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='lstm', rnn_hidden=128, rnn_layers=2,
                                     rnn_proj_dim=256, rnn_attach_to='bottleneck',
                                     use_grn=True)


def refined_grn_tiny_factory(num_classes):
    """RefinedHybrid + TinyGRU + GRN. 阶段 2 候选."""
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='bottleneck',
                                     use_grn=True)


def refined_grn_no_rnn_factory(num_classes):
    """RefinedHybrid (no RNN) + GRN. 阶段 2 候选."""
    return _RefinedHybridForCompare(num_classes, use_rnn=False, use_grn=True)


# RNN 挂载点消融: TinyGRU 在 bottleneck / dec3 / dec2 / dec1 4 个位置
def refined_tiny_at_bottleneck(num_classes):
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='bottleneck')


def refined_tiny_at_dec3(num_classes):
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='dec3')


def refined_tiny_at_dec2(num_classes):
    return _RefinedHybridForCompare(num_classes, use_rnn=True,
                                     rnn_cell='gru', rnn_hidden=64, rnn_layers=1,
                                     rnn_proj_dim=128, rnn_attach_to='dec2')


def stratified_k_fold(labels, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    class_idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_idx[lbl].append(i)
    fold_list = [[] for _ in range(n_splits)]
    for class_indices in class_idx.values():
        rng.shuffle(class_indices)
        per_fold = len(class_indices) // n_splits
        remainder = len(class_indices) % n_splits
        start = 0
        for split_idx in range(n_splits):
            extra = 1 if split_idx < remainder else 0
            end = start + per_fold + extra
            fold_list[split_idx].extend(class_indices[start:end])
            start = end
    all_idx = list(range(len(labels)))
    pairs = []
    for split_idx in range(n_splits):
        validation_indices = np.array(sorted(fold_list[split_idx]))
        training_indices = np.array(sorted(set(all_idx) - set(fold_list[split_idx])))
        pairs.append((training_indices, validation_indices))
    return pairs


def train_one_fold(model_factory, train_loader, val_loader, test_dl, epochs, tag="",
                   mixup_alpha=0.0, cutmix_alpha=0.0):
    model = model_factory(NUM_CLASSES).to(DEVICE)
    best_v, _ = fit_best_model(model, train_loader, val_loader, epochs,
                                progress_label=tag,
                                mixup_alpha=mixup_alpha,
                                cutmix_alpha=cutmix_alpha)
    t_acc = evaluate(model, test_dl)
    return model, best_v, t_acc


def main():
    def configure_parser(parser):
        parser.add_argument("--attention", action="store_true")
        parser.add_argument("--ablation", action="store_true")
        parser.add_argument("--sota", action="store_true")
        parser.add_argument("--hybrid", action="store_true",
                            help="对比 Hybrid+RNN / Hybrid-RNN 与 baseline Full+SE")
        parser.add_argument("--refined", action="store_true",
                            help="对比 RefinedHybrid 4 个变体 (UNet 4 级骨架)")
        parser.add_argument("--attach", action="store_true",
                            help="对比 TinyGRU 在 bottleneck/dec3/dec2/dec1 4 个挂载点")
        parser.add_argument("--models", type=str, default="")
        parser.add_argument("--pairs", action="store_true")
        parser.add_argument("--folds", type=str, default="all")

    args = parse_training_args(configure_parser)

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Device: %s | 5-Fold CV | %d epochs | seed=%d" %
          (DEVICE, args.epochs, args.seed))

    train_data, _, test_data, all_labels, label_to_idx = load_split_data()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    holdout_ds = CharDataset(test_data, train=False)
    holdout_loader = DataLoader(holdout_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    print("CV pool: %d  Holdout(051-055): %d  Classes: %d" %
          (len(train_data), len(test_data), len(all_labels)))

    all_models = [
        (HandCharNetResGELU, "ResGELU"), (HandCharNet, "Full+SE"),
        (HandCharNetGroupSE, "Full+GSE"), (HandCharNetECA, "Full+ECA"),
        (HandCharNetCBAM, "Full+CBAM"),
        (HandCharNetCA, "Full+CA"),
        (HandCharNetNoDirECA, "NoDir+ECA"), (HandCharNetNoDirHybrid, "NoDir+Hybrid"),
        (FPNCharNet, "FPN+Skip"),
        (CRNNCharNet, "AllInOne"), (HandCharNetNoDirection, "-Dir+SE"),
        (HandCharNetNoDirectionEca, "-Dir+ECA"),
        (HandCharNetNoSe, "-SE"), (HandCharNetNoMultiScale, "-MultiScale"),
        (ResNet18Char, "ResNet18"), (DenseNetChar, "DenseNet121"),
        (ConvNeXtV2Char, "ConvNeXtV2"), (MobileNetV4Char, "MobileNetV4"),
        (hybrid_full_factory, "Hybrid+RNN"),
        (hybrid_no_rnn_factory, "Hybrid-RNN"),
        (hybrid_tiny_rnn_factory, "Hybrid+TinyGRU"),
        (hybrid_transformer_factory, "Hybrid+Trans"),
        (refined_full_factory, "Refined+RNN"),
        (refined_no_rnn_factory, "Refined-RNN"),
        (refined_tiny_factory, "Refined+TinyGRU"),
        (refined_trans_factory, "Refined+Trans"),
        (refined_tiny_at_bottleneck, "TinyGRU@Bottleneck"),
        (refined_tiny_at_dec3, "TinyGRU@Dec3"),
        (refined_tiny_at_dec2, "TinyGRU@Dec2"),
    ]

    selected = set()
    if args.models:
        selected = {int(x.strip()) for x in args.models.split(",")}
    elif args.attention:
        selected |= {1, 2, 3, 4, 5, 6}  # SE, GSE, ECA, CBAM, CA, NoDir+ECA
    if args.ablation:
        selected |= {1, 10, 11, 12, 13}
    if args.sota:
        selected |= {14, 15, 16, 17}
    if args.hybrid:
        selected |= {1, 6, 18, 19, 20, 21}  # Full+SE + NoDir+ECA + Hybrid 4 版 (RNN/无RNN/TinyGRU/Trans)
    if args.refined:
        selected |= {18, 22, 23, 24, 25}  # Hybrid+RNN (50ep 冠军) + Refined 4 版
    if args.attach:
        selected |= {24, 26, 27, 28}  # TinyGRU @ dec1/bottleneck/dec3/dec2 挂载点消融
    if not selected:
        selected = set(range(len(all_models)))
    models = [all_models[i] for i in sorted(selected)]
    print("Models: %s" % ", ".join(name for _, name in models))

    k_fold_pairs = stratified_k_fold([d[1] for d in train_data], n_splits=5, seed=args.seed)
    if args.folds == "all":
        folds_to_run = list(enumerate(k_fold_pairs))
    else:
        requested_fold_indices = [int(x.strip()) - 1 for x in args.folds.split(",")]
        folds_to_run = [(i, k_fold_pairs[i]) for i in requested_fold_indices if 0 <= i < 5]
    print("Folds: %s\n" % ", ".join(str(i + 1) for i, _ in folds_to_run))

    results = {name: [] for _, name in models}
    class_results = {name: {c: [] for c in range(NUM_CLASSES)} for _, name in models}
    pair_results = {name: {key: [] for key in
                           ["%s/%s" % (a, b) for a, b in CONFUSABLE_PAIRS]}
                    for _, name in models} if args.pairs else {}

    for fold_index, (train_indices, validation_indices) in folds_to_run:
        print("=" * 55)
        print("Fold %d/5" % (fold_index + 1))
        print("=" * 55)

        # 每个 fold 只创建一次数据集，所有模型复用（缓存热了之后不再读盘）
        train_ds = CharDataset([train_data[i] for i in train_indices], train=True)
        val_ds = CharDataset([train_data[i] for i in validation_indices], train=False)

        for model_factory, model_name in models:
            # 每个模型新建 DataLoader（shuffle 需要独立迭代器）
            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_with_augment, num_workers=0)
            val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

            trained_model, val_acc, test_acc = train_one_fold(
                model_factory, train_loader, val_loader,
                holdout_loader, args.epochs, tag=model_name,
                mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha)
            results[model_name].append(test_acc)
            per_class = evaluate_per_class(trained_model, holdout_loader)
            for cls_idx in range(NUM_CLASSES):
                class_results[model_name][cls_idx].append(
                    per_class.get(cls_idx, 0.0))
            print("  best_val=%.4f  test(051-055)=%.4f" % (val_acc, test_acc))

            if args.pairs:
                pair_acc = compute_pair_accuracy(trained_model, holdout_loader, idx_to_label)
                for key, acc in pair_acc.items():
                    pair_results[model_name][key].append(acc)

    # ---- Summary ----
    sep = "=" * 70
    lines = [
        sep,
        "5-Fold CV (test=051-055, seed=%d) - %s" % (args.seed, timestamp),
        sep,
        "%-30s %-22s %-10s %s" % ("Model", "Acc(mean+/-std)", "Best", "Fold Accuracies"),
        "-" * 80,
    ]

    def model_row(row_model_name):
        row_accuracies = results[row_model_name]
        row_mean_accuracy = float(np.mean(row_accuracies))
        row_std_accuracy = float(np.std(row_accuracies))
        row_best = float(np.max(row_accuracies))
        return "%-30s %.4f +/- %.4f  %.4f    %s" % (
            row_model_name, row_mean_accuracy, row_std_accuracy, row_best,
            "  ".join("%.4f" % x for x in row_accuracies)
        )

    for _, model_name in models:
        lines.append(model_row(model_name))

    # Per-class breakdown (取最佳模型)
    best_model_name = max(results, key=lambda n: np.mean(results[n]))
    lines.append("\n" + sep + "\nPer-Class Accuracy (best model: %s, mean+best across folds)\n" % best_model_name + sep)
    top10, bottom10 = [], []
    for cls_idx in range(NUM_CLASSES):
        fold_values = class_results[best_model_name][cls_idx]
        mean_val = np.mean(fold_values)
        best_val = np.max(fold_values)
        top10.append((cls_idx, mean_val, best_val))
        bottom10.append((cls_idx, mean_val, best_val))
    top10.sort(key=lambda x: -x[1])
    bottom10.sort(key=lambda x: x[1])
    for label, items in [("Best", top10[:10]), ("Worst", bottom10[:10])]:
        parts = []
        for i, m, b in items:
            parts.append("%s=%.3f/%.3f" % (idx_to_label[i], m, b))
        lines.append("%-8s %s" % (label + " 10:", "  ".join(parts)))

    if args.ablation and "Full+SE" in results:
        lines.append("\n" + sep + "\n4-Layer Ablation\n" + sep)
        full_mean = float(np.mean(results["Full+SE"]))
        for ab_name, desc in [("-Dir+SE", "Layer2 DirectionConv"),
                               ("-Dir+ECA", "Layer2 Direction(ECA)"),
                               ("-SE", "Layer3 SE Attention"),
                               ("-MultiScale", "Layer4 MultiScale")]:
            if ab_name not in results: continue
            ablation_mean = float(np.mean(results[ab_name]))
            delta = full_mean - ablation_mean
            lines.append("  Remove %-25s: %.4f (delta=%+.4f, %+.1f%%)" %
                         (desc, ablation_mean, delta, delta * 100))

    if args.pairs:
        pair_keys_list = ["%s/%s" % (a, b) for a, b in CONFUSABLE_PAIRS]
        short_names = [k.replace("/", " vs ") for k in pair_keys_list]
        lines.append("\n" + sep + "\nConfusable Pair Accuracy (mean across folds)\n" + sep)
        lines.append("%-22s %s" % ("Model", " ".join("%-7s" % s for s in short_names)))
        lines.append("-" * 90)
        for _, model_name in models:
            row_data = [model_name[:21]]
            for key in pair_keys_list:
                pair_accuracies = pair_results[model_name][key]
                row_data.append("%.3f" % float(np.mean(pair_accuracies)) if pair_accuracies else "-")
            lines.append("%-22s %s" % (row_data[0], " ".join("%-7s" % v for v in row_data[1:])))

    for line in lines:
        print(line)

    with open("output/cv_results_%s.txt" % timestamp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open("output/cv_folds_%s.csv" % timestamp, "w", encoding="utf-8-sig") as f:
        fold_names = ",".join("Fold%d" % (fi + 1) for fi, _ in folds_to_run)
        f.write("Model,%s,Mean,Std\n" % fold_names)
        for _, model_name in models:
            csv_accuracies = results[model_name]
            mean_accuracy = float(np.mean(csv_accuracies))
            std_accuracy = float(np.std(csv_accuracies))
            f.write("%s,%s,%.4f,%.4f\n" %
                    (model_name, ",".join("%.4f" % x for x in csv_accuracies), mean_accuracy, std_accuracy))
    print("\nFiles: output/cv_results_%s.{txt,csv}" % timestamp)


if __name__ == "__main__":
    main()
