"""本地自检 (在装好 torch+timm 的 conda 环境里跑: python selfcheck.py)。

不需要 GPU、不读数据集、不下载预训练权重(pretrained=False, 仅验证结构/名字/前向)。
逐项打印 [PASS]/[FAIL]:
  1. torch / timm 就绪
  2. 推荐骨干名能否被 timm 解析
  3. transfer / transfer_ms / transfer_adapter 前向 + 输出形状 [B,62]
  4. 全部特征分支 + 强度模式 + 四种头(linear/cosine/ecoc/hier)构建+前向
  5. 损失 (focal + 混淆软标签 + CosFace) 前向
  6. 一步反传(确认可训练)
  7. count_params / metrics
真训练请用 train.py(需 GPU + 数据集 + pretrained=True)。
"""
import sys
import traceback

OK, BAD = 0, 0


def check(name, fn):
    global OK, BAD
    try:
        fn()
        print("[PASS] %s" % name); OK += 1
    except Exception as e:
        print("[FAIL] %s\n       %s" % (name, e)); BAD += 1
        if "-v" in sys.argv:
            traceback.print_exc()


def main():
    check("1. import torch / timm", lambda: (__import__("torch"), __import__("timm")))
    import torch
    from project_constants import NUM_CLASSES
    import models
    x = torch.randn(2, 1, 64, 48)

    # 2. 推荐骨干名解析 (pretrained=False, 不下载)
    def names():
        import timm
        for n, (feat_ok, _) in models.RECOMMENDED_BACKBONES.items():
            timm.create_model(n, pretrained=False, num_classes=0,
                              **({"features_only": True} if feat_ok else {}))
    check("2. 推荐骨干名 timm 解析", names)

    # 3. 三种 transfer arch 前向
    def fwd(cls, **kw):
        net = cls(model_name="convnextv2_nano", num_classes=NUM_CLASSES,
                  pretrained=False, **kw).eval()
        out = net(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        assert logits.shape == (2, NUM_CLASSES), "logits 形状 %s" % (logits.shape,)
    check("3a. TransferBackbone 前向", lambda: fwd(models.TransferBackbone))
    check("3b. TransferBackboneMS 前向", lambda: fwd(models.TransferBackboneMS))
    check("3c. TransferBackboneAdapter 前向", lambda: fwd(models.TransferBackboneAdapter))

    # 3d. DisentangledNet 前向 (双流 + 残差)
    def fwd_disentangled():
        net = models.DisentangledNet(model_name="convnextv2_nano", num_classes=NUM_CLASSES,
                                     pretrained=False, input_size=160).eval()
        logits_final, fused, logits_shape = net(x)
        assert logits_final.shape == (2, NUM_CLASSES), "logits_final shape %s" % (logits_final.shape,)
        assert logits_shape.shape == (2, NUM_CLASSES), "logits_shape shape %s" % (logits_shape.shape,)
        Delta = logits_final - logits_shape
        assert Delta.abs().mean() < 0.1, "Δ should be near-zero at init (scale=0), got %.4f" % Delta.abs().mean()
    check("3d. DisentangledNet 前向 (Δ≈0 at init)", fwd_disentangled)

    # 4. 全分支 + 强度 both + weber + 四种头
    def adapter(head, intensity="both"):
        net = models.TransferBackboneAdapter(
            model_name="convnextv2_nano", num_classes=NUM_CLASSES, pretrained=False,
            branches=("spatial", "channel", "multiscale", "gabor", "morph"),
            intensity_mode=intensity, weber_norm=True, head=head).eval()
        out = net(x)[0]
        assert out.shape == (2, NUM_CLASSES)
    for h in ["linear", "cosine", "ecoc", "hier"]:
        check("4. Adapter 全分支 + head=%s" % h, (lambda hh: (lambda: adapter(hh)))(h))

    # 5. 损失前向 (focal + 软标签 + CosFace)
    def loss_fwd():
        from losses import FocalLoss
        logits = torch.randn(8, NUM_CLASSES, requires_grad=True)
        y = torch.randint(0, NUM_CLASSES, (8,))
        pair_idx = [(0, 36), (10, 36)]  # 假混淆对
        crit = FocalLoss(gamma=2.0, smoothing=0.1, num_classes=NUM_CLASSES,
                         pair_smoothing=0.08, confusable_pairs_idx=pair_idx,
                         margin=0.2, margin_scale=30.0)
        l = crit(logits, y)
        l.backward()
        assert logits.grad is not None and torch.isfinite(l)
    check("5. FocalLoss(软标签+CosFace) 前向+反传", loss_fwd)

    # 5b. 残差损失
    def res_loss_fwd():
        from losses import residual_loss
        logits_shape = torch.randn(8, NUM_CLASSES)
        Delta = torch.randn(8, NUM_CLASSES) * 0.1
        logits_final = logits_shape + Delta
        y = torch.randint(0, NUM_CLASSES, (8,))
        pair_idx = [(0, 36), (10, 36)]
        loss, terms = residual_loss(logits_shape, logits_final, Delta, y,
                                    confusable_pairs_idx=pair_idx,
                                    lambda_sparse=1e-4, lambda_diff=0.1)
        assert torch.isfinite(loss)
        assert all(k in terms for k in ['L_shape', 'L_final', 'L_sparse', 'L_diff'])
    check("5b. residual_loss 前向", res_loss_fwd)

    # 6. 一步反传 (Adapter + 重建)
    def train_step():
        net = models.TransferBackboneAdapter(
            model_name="convnextv2_nano", num_classes=NUM_CLASSES,
            pretrained=False, recon=True).train()
        from losses import FocalLoss
        crit = FocalLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        logits, _, recon = net(x)
        loss = crit(logits, torch.randint(0, NUM_CLASSES, (2,)))
        if recon is not None:
            tgt = torch.nn.functional.interpolate(x, size=recon.shape[-2:])
            loss = loss + 0.3 * torch.nn.functional.mse_loss(recon, tgt)
        loss.backward(); opt.step()
        assert torch.isfinite(loss)
    check("6. Adapter+重建 一步反传", train_step)

    # 7. count_params / metrics
    def cp():
        from count_params import count_parameters
        net = models.TransferBackboneAdapter(model_name="convnextv2_nano",
                                             num_classes=NUM_CLASSES, pretrained=False)
        d = count_parameters(net)
        assert d["from_scratch"] > 0 and d["backbone"] > 0
    check("7a. count_params", cp)

    def met():
        import numpy as np
        from metrics import classification_metrics
        probs = np.random.dirichlet([1] * NUM_CLASSES, 50)
        labels = np.random.randint(0, NUM_CLASSES, 50)
        _, txt = classification_metrics(probs, labels)
        assert "Macro F1" in txt
    check("7b. metrics", met)

    print("\n==== 自检结果: %d 通过 / %d 失败 ====" % (OK, BAD))
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
