"""参数量统计工具: 打印任意模型配置的 总参数 / 冻结 / 可训练 / 骨干(预训练) / 从零新增。

过拟合看的是"从零新增的可训练容量 相对 训练集大小", 不是参数总量 ——
预训练骨干参数虽多, 但是优质初始化, 配判别式 lr/冻结只小幅移动。
本工具帮你用数字判断每个配置(加哪个分支、换哪个骨干、冻不冻)的有效容量。

用法示例:
  python count_params.py --arch transfer_adapter --transfer_model convnextv2_nano \\
    --adapter_branches spatial,channel,multiscale,gabor,morph --intensity_mode both
  python count_params.py --arch transfer --transfer_model resnet50 --freeze   # 模拟冻结骨干
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import _build_net

TRAIN_SIZE = 3100  # 干净留出协议下的训练集大小 (写手 001-050, 62×50)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = 0
    if hasattr(model, "backbone"):
        backbone = sum(p.numel() for p in model.backbone.parameters())
    return dict(total=total, trainable=trainable, frozen=total - trainable,
                backbone=backbone, from_scratch=total - backbone)


def _fmt(n):
    return "%s (%.2fM)" % ("{:,}".format(n), n / 1e6)


def _add_model_args(p):
    """镜像 train 里 _build_net 用到的全部参数 (默认值保持一致)。"""
    p.add_argument("--arch", type=str, default="transfer_adapter")
    p.add_argument("--transfer_model", type=str, default="convnextv2_femto")
    p.add_argument("--transfer_input", type=int, default=160)
    p.add_argument("--transfer_ms_stages", type=int, default=2)
    p.add_argument("--adapt_dim", type=int, default=64)
    p.add_argument("--adapter_branches", type=str, default="spatial,channel,multiscale")
    p.add_argument("--intensity_mode", type=str, default="stats")
    p.add_argument("--weber_norm", action="store_true")
    p.add_argument("--no_intensity", action="store_true")
    p.add_argument("--adapter_head", type=str, default="linear")
    p.add_argument("--arcface_scale", type=float, default=30.0)
    p.add_argument("--ecoc_length", type=int, default=127)
    p.add_argument("--recon_lambda", type=float, default=0.0)
    p.add_argument("--shape_dim", type=int, default=192)
    p.add_argument("--geo_dim", type=int, default=192)
    p.add_argument("--sparse_lambda", type=float, default=1e-4)
    p.add_argument("--diff_lambda", type=float, default=0.1)
    p.add_argument("--convnextv2p_input", type=int, default=96)
    p.add_argument("--resnet18p_input", type=int, default=128)
    p.add_argument("--no_rnn", action="store_true")
    p.add_argument("--rnn_cell", type=str, default="lstm")
    p.add_argument("--rnn_hidden", type=int, default=96)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--rnn_proj_dim", type=int, default=192)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_model_args(p)
    p.add_argument("--freeze", action="store_true",
                   help="模拟冻结骨干 (等价 --freeze_backbone_epochs 阶段), 看可训练量")
    args = p.parse_args()

    net = _build_net(args)
    if args.freeze and hasattr(net, "freeze_backbone"):
        net.freeze_backbone()

    d = count_parameters(net)
    print("=" * 60)
    print("参数量统计 | arch=%s | backbone=%s%s" %
          (args.arch, args.transfer_model, "  [骨干已冻结]" if args.freeze else ""))
    print("=" * 60)
    print("总参数        : %s" % _fmt(d["total"]))
    print("  ├ 预训练骨干: %s" % _fmt(d["backbone"]))
    print("  └ 从零新增  : %s   ← 过拟合主要看它" % _fmt(d["from_scratch"]))
    print("可训练参数    : %s" % _fmt(d["trainable"]))
    print("冻结参数      : %s" % _fmt(d["frozen"]))
    print("-" * 60)
    ratio = d["from_scratch"] / TRAIN_SIZE
    print("从零新增 / 训练集(%d) = %.1f 参数/样本" % (TRAIN_SIZE, ratio))
    print("可训练 / 训练集(%d)   = %.1f 参数/样本" % (TRAIN_SIZE, d["trainable"] / TRAIN_SIZE))
    print("(经验: '从零新增/样本' 越小越不易过拟合; 骨干冻结/小 lr 时可训练量更接近从零新增)")
    print("=" * 60)


if __name__ == "__main__":
    main()
