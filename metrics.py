"""多分类评估指标 —— 以 sklearn 为准, 缺失时回退纯 numpy 实现。

任务为 62 类单标签多分类; macro-F1 / macro-AUC 用 one-vs-rest 推广。
accuracy 会被大量易类掩盖, 故同时报 balanced accuracy / macro-F1 / 逐类 / 最差 K 类,
显式暴露 s/S、o/O 等混淆类弱点; macro-AUC 反映排序/置信度质量。

依赖: pip install scikit-learn (推荐)。eval 阶段离线只读, 不影响训练数据管线。
"""
import numpy as np

try:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, roc_auc_score)
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------- numpy 兜底 ----------------
def _average_ranks(sorted_vals):
    n = len(sorted_vals)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auc_ovr_np(scores, pos_mask):
    n = len(scores); n_pos = int(pos_mask.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = scores.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = _average_ranks(scores[order])
    return (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _per_class_prf_np(pred, labels, c):
    eps = 1e-12
    p = np.zeros(c); r = np.zeros(c); f = np.zeros(c); sup = np.zeros(c)
    for cls in range(c):
        gt = labels == cls; pr = pred == cls
        tp = np.sum(gt & pr); fp = np.sum(~gt & pr); fn = np.sum(gt & ~pr)
        p[cls] = tp / (tp + fp + eps); r[cls] = tp / (tp + fn + eps)
        f[cls] = 2 * p[cls] * r[cls] / (p[cls] + r[cls] + eps); sup[cls] = gt.sum()
    return p, r, f, sup


def classification_metrics(probs, labels, i2l=None, worst_k=10):
    """probs: [N,C] 概率; labels: [N] 真类索引。返回 (dict, 可打印字符串)。"""
    probs = _to_numpy(probs).astype(np.float64)
    labels = _to_numpy(labels).astype(np.int64)
    n, c = probs.shape
    pred = probs.argmax(1)
    all_labels = list(range(c))

    if _HAS_SKLEARN:
        accuracy = accuracy_score(labels, pred)
        balanced = balanced_accuracy_score(labels, pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, pred, average="macro", labels=all_labels, zero_division=0)
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            labels, pred, average=None, labels=all_labels, zero_division=0)
        try:
            macro_auc = roc_auc_score(labels, probs, multi_class="ovr",
                                      average="macro", labels=all_labels)
        except Exception:
            macro_auc = np.nan
        backend = "sklearn"
    else:
        per_p, per_r, per_f1, support = _per_class_prf_np(pred, labels, c)
        present = support > 0
        accuracy = float(np.mean(pred == labels))
        balanced = float(np.mean(per_r[present]))
        macro_p = float(np.mean(per_p[present]))
        macro_r = float(np.mean(per_r[present]))
        macro_f1 = float(np.mean(per_f1[present]))
        aucs = np.array([_auc_ovr_np(probs[:, k], labels == k) for k in range(c)])
        macro_auc = float(np.nanmean(aucs))
        backend = "numpy-fallback"

    support = np.asarray(support)
    per_f1 = np.asarray(per_f1)
    ci_acc = None
    if i2l is not None:
        ci_acc = float(np.mean([i2l[int(p)].lower() == i2l[int(t)].lower()
                                for p, t in zip(pred, labels)]))

    present_idx = np.where(support > 0)[0]
    worst = present_idx[np.argsort(per_f1[present_idx])][:worst_k]

    metrics = dict(accuracy=accuracy, balanced_accuracy=balanced,
                   macro_precision=macro_p, macro_recall=macro_r,
                   macro_f1=macro_f1, macro_auc=macro_auc,
                   case_insensitive_accuracy=ci_acc, backend=backend,
                   per_class_f1=per_f1, per_class_precision=np.asarray(per_p),
                   per_class_recall=np.asarray(per_r), support=support)

    lines = ["-" * 60, "评估指标 (n=%d, %d 类) [backend=%s]" % (n, c, backend), "-" * 60,
             "Accuracy            : %.4f" % accuracy,
             "Balanced Accuracy   : %.4f" % balanced,
             "Macro Precision     : %.4f" % macro_p,
             "Macro Recall        : %.4f" % macro_r,
             "Macro F1            : %.4f" % macro_f1,
             "Macro AUC (OvR)     : %.4f" % macro_auc]
    if ci_acc is not None:
        lines.append("大小写不敏感 Accuracy: %.4f  (天花板参考)" % ci_acc)
    lines.append("最差 %d 类 (按 F1):" % worst_k)
    for idx in worst:
        name = i2l[int(idx)] if i2l is not None else str(idx)
        lines.append("    %s : F1=%.3f  P=%.3f R=%.3f (n=%d)" %
                     (name, per_f1[idx], per_p[idx], per_r[idx], int(support[idx])))
    lines.append("-" * 60)
    return metrics, "\n".join(lines)
