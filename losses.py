"""Focal Loss + SIGReg (LeJEPA 高斯正则): 自动聚焦困难样本 + 各向同性特征空间"""
import math
import os, sys, datetime
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HandCharNet
from data_utils import load_split_data, CharDataset, collate_with_augment
from project_constants import DEVICE, NUM_CLASSES, BATCH_SIZE, CONFUSABLE_PAIRS
from training_utils import evaluate, compute_pair_accuracy

EPOCHS = 60
FOCAL_GAMMA = 2.0  # 聚焦强度: 越大越关注困难样本


class FocalLoss:
    """Focal Loss + (可选) 混淆矩阵感知软标签。

    标准模式 (pair_smoothing=0): 均匀 label smoothing, 平滑质量平摊给其余 n-1 类。
    混淆感知模式 (pair_smoothing>0 且给出 confusable_pairs_idx):
        承认 o/O, s/S, 1/l 这类天生歧义 —— 把 pair_smoothing 的质量【定向】分给
        该类的混淆伙伴, 而非均匀撒开。等价于在标签的混淆对位置设非零相似度 ε
        (dl-architecture-fusion 法则 185)。预计算 [n,n] 软标签矩阵, 训练时按真类索引取行。
    """

    def __init__(self, gamma=2.0, smoothing=0.1, num_classes=None,
                 pair_smoothing=0.0, confusable_pairs_idx=None,
                 margin=0.0, margin_scale=30.0):
        self.gamma = gamma
        self.smoothing = smoothing
        self.pair_smoothing = pair_smoothing
        # CosFace 加性间隔: 配合余弦头 (logit=scale*cos), 真类减 margin*scale = scale*(cos-m)
        self.margin = margin
        self.margin_scale = margin_scale
        self.soft_matrix = None
        if pair_smoothing > 0.0 and confusable_pairs_idx and num_classes:
            self.soft_matrix = self._build_soft_matrix(
                num_classes, smoothing, pair_smoothing, confusable_pairs_idx)

    @staticmethod
    def _build_soft_matrix(n, smoothing, pair_smoothing, pairs):
        """[n,n] 软标签矩阵: 行 c = 真类为 c 时的目标分布 (每行和=1)。"""
        m = torch.full((n, n), smoothing / (n - 1))
        for c in range(n):
            m[c, c] = 1.0 - smoothing
        partners = {}
        for i, j in pairs:
            partners.setdefault(i, set()).add(j)
            partners.setdefault(j, set()).add(i)
        for c, ps in partners.items():
            k = len(ps)
            if k == 0:
                continue
            m[c, c] -= pair_smoothing          # 从真类挪出 pair_smoothing
            for pidx in ps:
                m[c, pidx] += pair_smoothing / k  # 均分给各混淆伙伴
        return m

    def __call__(self, logits, targets):
        if self.margin > 0.0:
            # CosFace: 真类 logit 减 margin*scale。用独立 offset 张量做非原地减法,
            # 避免对计算图内张量原地 scatter_ 触发 autograd 报错。
            offset = torch.zeros_like(logits)
            offset.scatter_(1, targets.unsqueeze(1), self.margin * self.margin_scale)
            logits = logits - offset
        n = logits.shape[1]
        log_probs = F.log_softmax(logits, -1)
        with torch.no_grad():
            if self.soft_matrix is not None:
                smooth_targets = self.soft_matrix.to(log_probs.device)[targets]
            else:
                smooth_targets = torch.full_like(log_probs, self.smoothing / (n - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
            # 标准 focal: p_t 取真类概率 (不含 label smoothing)
            true_prob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
            focal_weight = (1 - true_prob).pow(self.gamma).unsqueeze(-1)
        return -(smooth_targets * log_probs * focal_weight).sum(-1).mean()




def residual_loss(logits_shape, logits_final, Delta, targets,
                  confusable_pairs_idx=None,
                  lambda_sparse=1e-4, lambda_diff=0.1, diff_tau=0.01):
    """残差损失: CE(shape) + CE(final) + λ₁|Δ|₁ + λ_diff·L_diff.

    设计原理 (ResNet 残差哲学的损失函数版):
      - logits_shape 走 GAP → 对尺寸/位移不变, C/c 天然模糊
      - Δ 来自多尺度结构信息, 只在形状不够时修补
      - L1 惩罚 → Δ 在 50 个非歧义类上自动 →0 (稀疏)
      - L_diff → 歧义对上 Δ 有区分力 (不会退化到 Δ==常数)

    Args:
        logits_shape: [B,62] 形状流 logits (恒等路径)
        logits_final: [B,62] 修正后 logits = logits_shape + Δ
        Delta:        [B,62] 残差修正量 (用于 L1 和 L_diff)
        targets:      [B]    真类索引
        confusable_pairs_idx: list of (i,j) 歧义对索引
        lambda_sparse: L1(Δ) 权重, 默认 1e-4 (需调: 太大会压死 Δ, 太小无稀疏效果)
        lambda_diff:   L_diff 权重, 默认 0.1
        diff_tau:      歧义对 Δ 差异的最小期望阈值 (平方差 < τ 就罚)

    Returns:
        scalar loss, dict of per-term values (for logging)
    """
    import torch.nn as nn
    ce = nn.CrossEntropyLoss()
    L_shape = ce(logits_shape, targets)
    L_final = ce(logits_final, targets)
    L_sparse = Delta.abs().mean()

    L_diff = torch.tensor(0.0, device=Delta.device)
    if confusable_pairs_idx and lambda_diff > 0:
        diffs = []
        for i, j in confusable_pairs_idx:
            mask_i = (targets == i)
            mask_j = (targets == j)
            if mask_i.any() and mask_j.any():
                delta_i = Delta[mask_i].mean(0)  # [62]
                delta_j = Delta[mask_j].mean(0)  # [62]
                sq_dist = (delta_i - delta_j).pow(2).mean()
                diffs.append(torch.relu(diff_tau - sq_dist))
        if diffs:
            L_diff = torch.stack(diffs).mean()

    total = L_shape + L_final + lambda_sparse * L_sparse + lambda_diff * L_diff

    terms = {
        'L_shape': L_shape.item(),
        'L_final': L_final.item(),
        'L_sparse': L_sparse.item(),
        'L_diff': L_diff.item(),
        'total': total.item(),
    }
    return total, terms

def sigreg_loss(features, n_projections=1024, num_points=17, t_max=2.0):
    """SIGReg (Sketched Isotropic Gaussian Regularization, LeJEPA arXiv:2511.08544).

    论文严格版: 在 num_points 个 t 点离散评估经验特征函数 (ECF) 与 N(0,1) 特征
    函数 exp(-t²/2) 的平方距离, 梯度/曲率有界 (论文 §3, 表 2).
    机理: 把 [B, D] 随机单位投影到 K 条 1D 射线 [B, K], 每条投影计算
    |φ_emp(t) - φ_N01(t)|² 在 num_points 个 t 上的均值, 再对 K 条投影取均值.
    单超参 λ, 无 warmup / 无 schedule, 训练全程恒定加权.

    Args:
        features: [B, D] 监督特征 (unified). 监督场景 λ 推荐 0.05~0.1.
        n_projections: K 条投影方向 (论文默认 1024).
        num_points: ECF 离散评估点数 (论文 README 推荐 17).
        t_max: 评估点对称分布在 [-t_max, t_max] (覆盖 N(0,1) 特征函数主体).

    Returns:
        scalar ≥ 0, 完美高斯 → 0.
    """
    b, d = features.shape
    if b < 2:
        return features.new_zeros(())
    # 随机单位投影方向 (高斯采样 + 行单位化)
    w = torch.randn(d, n_projections, device=features.device, dtype=features.dtype)
    w = w / (w.norm(dim=0, keepdim=True) + 1e-6)
    proj = features @ w                                              # [B, K]
    # per-projection 标准化 (mean=0, std=1), 让对照分布固定为 N(0,1)
    proj = (proj - proj.mean(dim=0, keepdim=True)) / (
        proj.std(dim=0, keepdim=True) + 1e-6)
    # 评估点: 对称分布在 [-t_max, t_max] 上 num_points 个 (论文默认 17)
    t = torch.linspace(-t_max, t_max, num_points,
                       device=features.device, dtype=features.dtype)  # [P]
    # ECF 实部/虚部: φ_emp(t) = (1/B) Σ_j exp(i·t·proj_j)
    angles = proj.unsqueeze(-1) * t                                  # [B, K, P]
    re_emp = torch.cos(angles).mean(dim=0)                           # [K, P]
    im_emp = torch.sin(angles).mean(dim=0)                           # [K, P]
    # N(0,1) 特征函数: φ_N01(t) = exp(-t²/2) (实数, 虚部 = 0)
    re_ref = torch.exp(-t.pow(2) / 2.0)                              # [P]
    return ((re_emp - re_ref).pow(2) + im_emp.pow(2)).mean()


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    train_data, _, test_data, all_labels, l2i = load_split_data()
    i2l = {v: k for k, v in l2i.items()}

    train_ds = CharDataset(train_data, train=True)
    test_ds = CharDataset(test_data, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_augment, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    net = HandCharNet(NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    opt = AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        net.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        test_acc = evaluate(net, test_loader)
        done = ep + 1
        bar = "#" * (done * 20 // EPOCHS) + "-" * (20 - done * 20 // EPOCHS)
        print("\r  Ep%2d/%d [%s] loss=%.3f test=%.4f" %
              (done, EPOCHS, bar, total_loss / len(train_loader), test_acc),
              end="", flush=True)
    print()

    # Per-pair eval (复用 training_utils)
    pair_acc = compute_pair_accuracy(net, test_loader, i2l)
    print("Confusable pairs:")
    for k, v in pair_acc.items():
        print("  %s: %.3f" % (k, v))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), "output/focal_%s.pth" % timestamp)
    print("Saved: output/focal_%s.pth" % timestamp)
