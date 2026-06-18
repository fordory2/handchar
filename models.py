# noinspection SpellCheckingInspection
"""模型定义: 组件 + 工厂 + 14模型 + SOTA基线 + HybridHandCharNet"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import timm
from torchvision.models import densenet121, resnet18, resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1,
                 activation_cls: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = activation_cls()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, activation_cls: type[nn.Module] = nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = activation_cls()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(channels, channels // reduction), nn.ReLU(),
                                nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class GroupedSEBlock(nn.Module):
    """分组SE: 每个方向组独立注意力, 不跨组压缩. 解耦 Direction+SE 互斥."""
    def __init__(self, channels, groups=4, reduction=4):
        super().__init__()
        c_per = channels // groups
        hidden = max(4, c_per // reduction)
        self.groups = groups
        self.c_per = c_per
        self.fc1 = nn.ModuleList([nn.Linear(c_per, hidden) for _ in range(groups)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden, c_per) for _ in range(groups)])
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, channels = x.shape[0], x.shape[1]
        pooled = self.gap(x).view(batch_size, self.groups, self.c_per)
        weights = []
        for g in range(self.groups):
            weight = self.fc2[g](self.act(self.fc1[g](pooled[:, g])))
            weights.append(weight.unsqueeze(1))
        return x * self.sigmoid(
            torch.cat(weights, dim=1).view(batch_size, channels, 1, 1))


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(np.log2(channels) / gamma + b / gamma))
        k = max(t if t % 2 == 1 else t + 1, 3)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.attn_dropout = nn.Dropout2d(0.1)  # 正则化: 防止 attention 过拟合
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        weights = self.sigmoid(self.conv(y).transpose(-1, -2).unsqueeze(-1))
        return x * self.attn_dropout(weights)


class GRN(nn.Module):
    """Global Response Normalization, ConvNeXt V2 (Woo 2023).
    抑制通道维度的 feature collapse, 残差形式 (γ 初始 0 → 训练时自学是否使用).
    输入: [B, C, H, W], 输出: [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ArcMarginProduct(nn.Module):
    """ArcFace cosine head: 返回 scale*cos(θ) logits, 不在 forward 加 margin.

    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.

    设计上不在 model.forward 里加 margin (避免线路里穿 labels),
    margin 由配套的 ArcFaceCrossEntropy 在训练时施加.
    Inference: argmax(scale*cos) == argmax(scale*cos+margin), 不影响结果.
    """
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = functional.linear(functional.normalize(features, dim=1),
                                    functional.normalize(self.weight, dim=1))
        return cosine * self.scale


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ch_mlp = nn.Sequential(nn.Flatten(), nn.Linear(channels, channels // reduction),
                                    nn.ReLU(), nn.Linear(channels // reduction, channels))
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ch = self.sigmoid(self.ch_mlp(self.ch_avg(x)).unsqueeze(-1).unsqueeze(-1) +
                          self.ch_mlp(self.ch_max(x)).unsqueeze(-1).unsqueeze(-1))
        x_ch = x * ch
        avg_sp, max_sp = x_ch.mean(dim=1, keepdim=True), x_ch.max(dim=1, keepdim=True)[0]
        return x_ch * self.spatial(torch.cat([avg_sp, max_sp], dim=1))


class CABlock(nn.Module):
    def __init__(self, channels, reduction=32):  # reduction 加大: 更强正则化
        super().__init__()
        c_hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, c_hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_hidden)
        self.conv_h = nn.Conv2d(c_hidden, channels, 1)
        self.conv_w = nn.Conv2d(c_hidden, channels, 1)
        self.attn_dropout = nn.Dropout2d(0.15)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

    def forward(self, x):
        _, _, height, width = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        weights = self.sigmoid(self.conv_h(x_h)) * self.sigmoid(self.conv_w(x_w))
        return x * self.attn_dropout(weights)


class DirectionConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        c = out_c // 4
        self.horiz = nn.Conv2d(in_c // 4, c, (1, 5), padding=(0, 2), bias=False)
        self.vert = nn.Conv2d(in_c // 4, c, (5, 1), padding=(2, 0), bias=False)
        self.diag = nn.Conv2d(in_c // 4, c, 3, padding=1, bias=False)
        self.anti = nn.Conv2d(in_c // 4, c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        c = x.size(1) // 4
        x0, x1 = x[:, :c], x[:, c:2 * c]
        x2, x3 = x[:, 2 * c:3 * c], x[:, 3 * c:]
        return functional.relu(self.bn(torch.cat(
            [self.horiz(x0), self.vert(x1), self.diag(x2), self.anti(x3)], 1)))


class MultiScalePool(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        c = out_c // 4
        self.bn = nn.BatchNorm2d(out_c)
        self.conv_in = nn.Conv2d(in_c, c, 1, bias=False)
        self.conv_p1, self.pool1 = nn.Conv2d(in_c, c, 1, bias=False), nn.AdaptiveAvgPool2d(1)
        self.conv_p2, self.pool2 = nn.Conv2d(in_c, c, 1, bias=False), nn.AdaptiveAvgPool2d(2)
        self.conv_p3, self.pool3 = nn.Conv2d(in_c, c, 1, bias=False), nn.AdaptiveAvgPool2d(4)

    def forward(self, x):
        sz = x.size()[2:]
        x0 = self.conv_in(x)
        p1 = functional.interpolate(self.conv_p1(self.pool1(x)), size=sz, mode='nearest')
        p2 = functional.interpolate(self.conv_p2(self.pool2(x)), size=sz, mode='nearest')
        p3 = functional.interpolate(self.conv_p3(self.pool3(x)), size=sz, mode='nearest')
        return functional.relu(self.bn(torch.cat([x0, p1, p2, p3], 1)))


class SpatialAttentionBlock(nn.Module):
    """空间注意力: 通道维 avg+max pool, 7×7 conv 学每个位置权重 (CBAM 的空间分支单拎出)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        weights = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * weights


class FrequencyBlock(nn.Module):
    """DCT 频域分支: 2D DCT-II → 频域 1×1 conv 学习滤波 → IDCT → 3×3 conv 投影."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.freq_conv = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.fuse = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    @staticmethod
    def _dct_matrix(n, device, dtype):
        ks = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        ns = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        m = torch.cos(math.pi * (2 * ns + 1) * ks / (2 * n))
        m[0] *= 1.0 / math.sqrt(n)
        m[1:] *= math.sqrt(2.0 / n)
        return m

    def forward(self, x):
        b, c, h, w = x.shape
        mh = self._dct_matrix(h, x.device, x.dtype)
        mw = self._dct_matrix(w, x.device, x.dtype)
        x_freq = mh @ x.reshape(b * c, h, w) @ mw.t()
        x_freq = self.freq_conv(x_freq.reshape(b, c, h, w))
        x_back = mh.t() @ x_freq.reshape(b * c, h, w) @ mw
        return self.act(self.bn(self.fuse(x_back.reshape(b, c, h, w))))


# ====== HandCharNet 工厂 ======
GROUP_SE = 'gse'
SE = 'se'
ECA = 'eca'
CBAM = 'cbam'
CA = 'ca'
_ATTN = {SE: SEBlock, ECA: ECABlock, GROUP_SE: GroupedSEBlock,
         CBAM: CBAMBlock, CA: CABlock}


def make_hand_char(attention: Optional[str] = 'se', use_direction=True, use_multiscale=True,
                   residual=False, activation_cls: type[nn.Module] = nn.ReLU,
                   dropout_rate=0.146):
    activation_factory = activation_cls
    attn_cls = None if attention is None else _ATTN.get(attention)

    class _Net(nn.Module):
        def __init__(self, num_classes=62, dropout=dropout_rate):
            super().__init__()
            self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3, activation_factory), nn.MaxPool2d(2))
            dir_conv = DirectionConv(32, 48) if use_direction else ConvBlock(32, 48, 3, 1, 1, activation_factory)
            dir_layers: list[nn.Module] = [dir_conv]
            if attn_cls:
                dir_layers.append(attn_cls(48))
            dir_layers.append(nn.MaxPool2d(2))
            self.direction = nn.Sequential(*dir_layers)
            if residual:
                self.res1 = ResBlock(48, 80, activation_cls=activation_factory)
                self.res2 = ResBlock(80, 80, activation_cls=activation_factory)
                self.pool_s = nn.MaxPool2d(2)
            else:
                self.shape1 = nn.Sequential(
                    ConvBlock(48, 80, 3, 1, 1, activation_factory), ConvBlock(80, 80, 3, 1, 1, activation_factory), nn.MaxPool2d(2))
            if use_multiscale:
                self.ms = MultiScalePool(80, 128)
                post_ms: list[nn.Module] = [
                    nn.Sequential(nn.Conv2d(128, 96, 1, bias=False), nn.BatchNorm2d(96), activation_factory()),
                    ConvBlock(96, 96, 3, 1, 1, activation_factory)]
            else:
                self.ms = nn.Sequential(ConvBlock(80, 128, 3, 1, 1, activation_factory), nn.MaxPool2d(2))
                post_ms = [ConvBlock(128, 96, 3, 1, 1, activation_factory)]
            if residual:
                self.post_ms = post_ms[0] if len(post_ms) == 1 else nn.Sequential(*post_ms)
                self.res_ms = ResBlock(128, 96, activation_cls=activation_factory) if len(post_ms) == 1 else None
            else:
                self.shape2 = nn.Sequential(self.ms, *post_ms)
            detail_layers: list[nn.Module] = [ConvBlock(96, 160, 3, 1, 1, activation_factory)]
            if attn_cls:
                detail_layers.append(attn_cls(160))
            detail_layers.append(
                ResBlock(160, 160, activation_cls=activation_factory) if residual else ConvBlock(160, 160, 3, 1, 1, activation_factory))
            self.detail = nn.Sequential(*detail_layers)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                      nn.Linear(160, 128), activation_factory(), nn.Dropout(dropout * 0.5),
                                      nn.Linear(128, num_classes))

        def forward(self, x):
            x = self.direction(self.stem(x))
            if residual:
                x = self.res2(self.res1(x))
                x = self.pool_s(x)
                x = self.ms(x)
                x = self.res_ms(x) if self.res_ms else self.post_ms(x)
            else:
                x = self.shape2(self.shape1(x))
            return self.head(self.detail(x))

    return _Net


# 老模型 01 | HandCharNet —— 从零基线: 空间+通道(SE)+多尺度 三分支
class HandCharNet(make_hand_char(SE)):
    pass


# 老模型 02 | 消融: 去方向卷积 (DirectionConv)
class HandCharNetNoDirection(make_hand_char(SE, use_direction=False)):
    pass


# 老模型 03 | 消融: 去 SE 通道注意力
class HandCharNetNoSe(make_hand_char(None)):
    pass


# 老模型 04 | 消融: 去多尺度池化
class HandCharNetNoMultiScale(make_hand_char(SE, use_multiscale=False)):
    pass


# ====== SOTA 基线 ======
# 老模型 05 | SOTA 基线: 从零 ResNet18
class ResNet18Char(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, x):
        return self.model(x)


# 老模型 06 | SOTA 基线: ImageNet 预训练 ResNet18 (原 91% 起点)
class ResNet18Pretrained(nn.Module):
    """ImageNet 预训练 ResNet18 + 1ch 适配 + 前侧 SpatialDropout.

    关键设计 (复现"魔改 ResNet18 → 91%" 的可能路径):
    1. ImageNet 预训练: torchvision.resnet18(weights=IMAGENET1K_V1)
       自然图像低级边缘/纹理先验对 3410 样本是强迁移收益.
    2. conv1 1ch 适配: 把预训练 3ch [64,3,7,7] 在通道维求和 → [64,1,7,7]
       (sum 而非 mean: 灰度图等价 3ch 同值, sum 让响应幅度匹配原 BGR 输入)
    3. 输入上采样 64×48 → 128×128: ResNet18 设计 spec 是 224, 128 折中保速度.
    4. layer1/2/3 后插 Dropout2d(0.1/0.1/0.15): SpatialDropout, 通道级随机置零,
       等价于轻量 DropConnect, 朋友说的 "前侧 dropout" 路线.
    5. forward 返 (logits, unified_512, None) 兼容 train.py 3 元组接口
       unified = layer4 GAP 后的 512-d 向量, 给 SIGReg 用.

    输入: [B, 1, 64, 48]
    输出: (logits[B,62], unified[B,512], None)
    """
    def __init__(self, num_classes=62, pretrained=True, input_size=128,
                 drop1=0.1, drop2=0.1, drop3=0.15):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # 1ch 适配: 预训练 conv1.weight [64,3,7,7] sum 到 [64,1,7,7]
        old_w = backbone.conv1.weight.data
        new_conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(old_w.sum(dim=1, keepdim=True))
        backbone.conv1 = new_conv1

        self.input_size = input_size
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.drop1 = nn.Dropout2d(p=drop1)
        self.drop2 = nn.Dropout2d(p=drop2)
        self.drop3 = nn.Dropout2d(p=drop3)
        # head: 512 → num_classes (轻量 head, dropout 已在前侧)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # upsample 到 input_size (128) — bilinear 保留细节
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = functional.interpolate(x, size=(self.input_size, self.input_size),
                                       mode='bilinear', align_corners=False)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.drop1(x)
        x = self.layer2(x); x = self.drop2(x)
        x = self.layer3(x); x = self.drop3(x)
        x = self.layer4(x)
        unified = self.avgpool(x).flatten(1)        # [B, 512]
        logits = self.fc(unified)
        return logits, unified, None


# 老模型 07 | SOTA 基线: DenseNet121
class DenseNetChar(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = densenet121(num_classes=num_classes)
        self.model.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, x):
        return self.model(x)


# 老模型 08 | SOTA 基线: ConvNeXtV2-atto (FCMAE+IN1k 预训练)
class ConvNeXtV2Char(nn.Module):
    """timm ConvNeXt V2 atto, FCMAE+IN1k 预训练 + 1ch 适配 + 上采样 + dropout.

    设计要点:
    1. timm 加载 'convnextv2_atto.fcmae_ft_in1k': 已在 ImageNet 上做过 FCMAE 自监督
       预训练 + IN1k 有监督微调, 直接含 GRN 模块, 是 ConvNeXt V2 论文标配.
    2. in_chans=1: timm 自动把预训练 conv1 [40,3,4,4] 在 dim=1 平均到 [40,1,4,4].
    3. 输入 64×48 → upsample 到 input_size (默认 96, atto stem stride=4 时
       output 24→12→6→3, GAP 收尾稳定).
    4. drop_path_rate / 头前 dropout: timm 原生 drop_path + 我们加 head_drop,
       前侧正则就靠 ConvNeXt 内置 GRN + DropPath, 不再叠 SpatialDropout.
    5. forward 返 (logits, unified, None) 兼容 train.py 3 元组接口
       unified = forward_features → GAP 后的 320-d 向量.

    输入: [B, 1, 64, 48]
    输出: (logits[B,62], unified[B,320], None)
    """
    def __init__(self, num_classes=62, pretrained=True, input_size=96,
                 drop_path_rate=0.1, head_dropout=0.1):
        super().__init__()
        self.input_size = input_size
        # timm 自动: pretrained 权重的 stem.0 [40,3,4,4] → mean over dim=1 → [40,1,4,4]
        self.backbone = timm.create_model(
            'convnextv2_atto.fcmae_ft_in1k' if pretrained else 'convnextv2_atto',
            pretrained=pretrained,
            num_classes=0,                # 去掉原 head, 我们自己接
            in_chans=1,
            drop_path_rate=drop_path_rate,
            global_pool='avg',
        )
        self.feat_dim = self.backbone.num_features   # 320
        self.head_dropout = nn.Dropout(p=head_dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)
        # fc 显式 trunc_normal init: 修 ConvNeXtV2 + PyTorch 默认 fc init 偶发 dead start
        # (实测 5-fold 中 fold 0 起得来 0.9018 而 fold 1 困在 1/62=0.0161 整 15 epoch)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = functional.interpolate(x, size=(self.input_size, self.input_size),
                                       mode='bilinear', align_corners=False)
        unified = self.backbone(x)                    # [B, 320] (global_pool='avg' 已 GAP)
        logits = self.fc(self.head_dropout(unified))
        return logits, unified, None


# 老模型 09 | SOTA 基线: MobileNetV4
class MobileNetV4Char(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = timm.create_model('mobilenetv4_conv_small', pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.model(x)


# 老模型 10 | SOTA 基线: FPN + 跳跃连接
class FPNCharNet(nn.Module):
    def __init__(self, num_classes=62, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(DirectionConv(32, 48), SEBlock(48, 4), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(ConvBlock(48, 80, 3), ConvBlock(80, 80, 3), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(ConvBlock(80, 160, 3), SEBlock(160, 4), ConvBlock(160, 160, 3),
                                    nn.MaxPool2d(2))
        self.lat2, self.lat3, self.lat4 = (nn.Conv2d(48, 128, 1), nn.Conv2d(80, 128, 1),
                                           nn.Conv2d(160, 128, 1))
        self.smooth3, self.smooth2 = ConvBlock(128, 128, 3), ConvBlock(128, 128, 3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(384, 256), nn.ReLU(),
                                  nn.Dropout(dropout * 0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        s2 = self.stage2(self.stem(x))
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        lat_2, lat_3, lat_4 = self.lat2(s2), self.lat3(s3), self.lat4(s4)
        pyr_4 = lat_4
        pyr_3 = self.smooth3(lat_3 + functional.interpolate(pyr_4, size=lat_3.shape[2:],
                              mode='bilinear', align_corners=False))
        pyr_2 = self.smooth2(lat_2 + functional.interpolate(pyr_3, size=lat_2.shape[2:],
                              mode='bilinear', align_corners=False))
        return self.head(torch.cat([self.gap(pyr_2), self.gap(pyr_3), self.gap(pyr_4)], 1))


# 老模型 11 | SOTA 基线: CNN + RNN 一体
class CRNNCharNet(nn.Module):
    def __init__(self, num_classes=62, hidden_size=128, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3), nn.MaxPool2d((2, 1)))
        self.stage2 = nn.Sequential(DirectionConv(32, 64), SEBlock(64, 4), nn.MaxPool2d((2, 1)))
        self.stage3 = nn.Sequential(ConvBlock(64, 96, 3), SEBlock(96, 4), ConvBlock(96, 96, 3),
                                    nn.MaxPool2d((2, 2)))
        self.lat2, self.lat3 = nn.Conv2d(64, 128, 1), nn.Conv2d(96, 128, 1)
        self.smooth = ConvBlock(256, 128, 3)
        self.se_fuse = SEBlock(128, 4)
        self.lstm_proj = nn.Linear(1024, 256)
        self.lstm = nn.LSTM(256, hidden_size, 2, batch_first=False, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(),
                                  nn.Dropout(dropout * 0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        s2 = self.stage2(self.stem(x))
        s3 = self.stage3(s2)
        lat_2, lat_3 = self.lat2(s2), self.lat3(s3)
        lat_3_up = functional.interpolate(lat_3, size=lat_2.shape[2:],
                                          mode='bilinear', align_corners=False)
        fused = self.se_fuse(self.smooth(torch.cat([lat_2, lat_3_up], 1)))
        batch, channels, height, width = fused.shape
        seq = fused.permute(3, 0, 1, 2).reshape(width, batch, channels * height)
        seq = self.lstm_proj(seq)
        _, (hidden_state, _) = self.lstm(seq)
        return self.head(torch.cat([hidden_state[-2], hidden_state[-1]], 1))


# ====== HybridHandCharNet: 4 分支 + UNet skip + BiLSTM + 多头输出 ======
# 老模型 12 | 原创主模型: 三分支 + UNet 跳连 + BiLSTM (★报告创新点)
class HybridHandCharNet(nn.Module):
    """大一统模型 (中等剪枝版): 空间/通道/多尺度 3 分支 → UNet 跳跃 → 可选 BiLSTM → 主头.

    剪枝记录 (2026-06-10): 删 freq_branch (DCT 在 48 宽度低分辨上贡献薄), 删 head_aux/head_pair
    (训练流程 AUX/PAIR_WEIGHT 已置 0, 死路径). LSTM 实测 +4.84pp 保留.

    输入: [B, 1, H, W] (原生 64×48, 内部 bilinear 升采到 input_size 工作分辨率)
    输出: (main_logits[B,62], unified_feat, decoder_out)

    分辨率正常化 (2026-06-11): 原 64×48 输入下三分支 / ResNet stem / multi-scale pool
    都在"亚设计分辨率"工作 — 多尺度 pool(4) 比 bottleneck (4×3) 还大变纯噪声, stem 7×7
    感受野在 64×48 上几乎 cover 整图. 升到 128×96 (×2 双线性, 保 4:3) 让所有空间模块
    回到设计意图工作点; lstm_proj 入口 64×16=1024 通过对 decoder_out 钳制 H=16 保持不变,
    W 自然 12→24, 序列建模信息量翻倍.
    """
    def __init__(self, num_classes=62, dropout=0.22, use_rnn=True,
                 rnn_cell='lstm', rnn_hidden=96, rnn_layers=1, rnn_proj_dim=192,
                 use_grn=False, use_arcface=False, head_bottleneck=192, arcface_scale=30.0,
                 input_size=(128, 96)):
        super().__init__()
        self.use_rnn = use_rnn
        self.use_grn = use_grn
        self.use_arcface = use_arcface
        self.input_size = input_size  # (H, W) 工作分辨率, encode() 第一步 bilinear 升采

        # Stem: 1/4 下采样
        self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3), nn.MaxPool2d(2))

        # 3 个并联分支 (输入 [B,32,16,12], 每路输出 16 通道) - 频域分支已剪
        branch_c = 16
        stem_c = 32
        self.spatial_branch = nn.Sequential(
            SpatialAttentionBlock(),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.channel_branch = nn.Sequential(
            ECABlock(stem_c),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.multiscale_branch = MultiScalePool(stem_c, branch_c)
        # 3×16 → 64 通道融合
        self.branch_fuse = ConvBlock(branch_c * 3, 64, 1, 1, 0)
        # 输出 feat_fine: [B,64,16,12]

        # Encoder (中档扩容: 64/96/128 → 80/128/160, 回到 baseline 容量验证欠拟合假设)
        self.stage1 = nn.Sequential(ResBlock(64, 80), nn.MaxPool2d(2))   # [B,80,8,6]
        self.stage2 = nn.Sequential(ResBlock(80, 128), nn.MaxPool2d(2))  # [B,128,4,3]
        self.stage3 = ResBlock(128, 160)                                  # [B,160,4,3]

        # UNet 风格 decoder (跟随上游扩容)
        self.up3 = nn.ConvTranspose2d(160, 128, 2, 2)               # → [B,128,8,6]
        self.fuse3 = ConvBlock(128 + 80, 128, 1, 1, 0)              # concat feat_mid (80ch)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)                # → [B,64,16,12]
        self.fuse2 = ConvBlock(64 + 64, 64, 1, 1, 0)                # concat feat_fine (64ch)

        # 可选 GRN: 插在 4 个关键节点 (残差形式, γ=0 初始 → 不破基线)
        #   grn_fuse:   branch_fuse 后, 多尺度融合点 (64ch)
        #   grn_stage3: stage3 后, 最深语义 (128ch)
        #   grn_dec3:   fuse3 后, decoder 中层 (96ch)
        #   grn_dec2:   fuse2 后, decoder_out / RNN 接入点 (48ch)
        if use_grn:
            self.grn_fuse = GRN(64)
            self.grn_stage3 = GRN(160)
            self.grn_dec3 = GRN(128)
            self.grn_dec2 = GRN(64)

        # 可选序列建模: decoder_out [B,64,16,12] → W=12 步, 每步 64*16=1024 维
        # 中档扩容: proj 1024→192, BiLSTM h=96, L=1 (rnn_dim=192)
        self.rnn_cell = rnn_cell.lower() if use_rnn else None
        if use_rnn:
            self.lstm_proj = nn.Linear(64 * 16, rnn_proj_dim)
            if self.rnn_cell == 'transformer':
                # 可学习位置编码: 预留 W_max=32 (升采后 W=24, slice 用前 W 个; 64×48 兼容)
                self.pos_encoding = nn.Parameter(torch.zeros(32, 1, rnn_proj_dim))
                nn.init.normal_(self.pos_encoding, std=0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=rnn_proj_dim, nhead=4,
                    dim_feedforward=rnn_proj_dim * 2,
                    dropout=dropout, batch_first=False,
                )
                self.lstm = nn.TransformerEncoder(encoder_layer, num_layers=rnn_layers)
                rnn_dim = rnn_proj_dim  # 不双向
            else:
                rnn_cls = nn.GRU if self.rnn_cell == 'gru' else nn.LSTM
                self.lstm = rnn_cls(rnn_proj_dim, rnn_hidden, num_layers=rnn_layers,
                                    batch_first=False, bidirectional=True,
                                    dropout=dropout if rnn_layers > 1 else 0.0)
                rnn_dim = rnn_hidden * 2  # 双向
            self.rnn_dropout = nn.Dropout(dropout * 2)  # 抑制序列模块过拟合
        else:
            rnn_dim = 0

        # 全局聚合 + 多头
        self.gap_coarse = nn.AdaptiveAvgPool2d(1)
        self.gap_fine = nn.AdaptiveAvgPool2d(1)
        unified_dim = 160 + 64 + rnn_dim
        self.unified_dim = unified_dim

        # head_main: bottleneck 加宽 128 → head_bottleneck (默认 256), 最后一层可换 ArcFace.
        if use_arcface:
            self.head_main = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(unified_dim, head_bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                ArcMarginProduct(head_bottleneck, num_classes, scale=arcface_scale),
            )
        else:
            self.head_main = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(unified_dim, head_bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_bottleneck, num_classes),
            )
        # head_aux / head_pair / contrastive head 全删: 训练流程 weight 均为 0, 死路径.

        self._init_weights()

    def _init_weights(self):
        """全部走 PyTorch 默认 init (Conv2d: kaiming_uniform(a=√5); Linear: kaiming_uniform).

        切换原因 (2026-06-11): 引入预训练 stem 子类 HybridHandCharNetR18Stem 后,
        显式 Kaiming reset 会在 super().__init__() 期间作用到所有 Conv, 后续注入预训练
        stem 时虽然顺序上能保证 weights 不被覆盖, 但时序依赖脆弱. 改走 PyTorch 默认让
        子类不必依赖构造顺序; 父类 from-scratch 训练时同样可接受 (默认 init 实测可训).
        """
        pass
        # 旧 from-scratch init 策略 (保留备查):
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.trunc_normal_(m.weight, std=0.02)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def encode(self, x):
        """编码到 (feat_coarse, decoder_out), 供 MAE / Proto 复用.

        分辨率正常化: 任意输入 H×W bilinear 升采到 self.input_size (默认 128×96)
        让 stem 7×7 感受野 / 三分支 / multi-scale pool 全部在设计意图分辨率工作.
        """
        if x.shape[-2:] != tuple(self.input_size):
            x = functional.interpolate(x, size=self.input_size,
                                       mode='bilinear', align_corners=False)
        x = self.stem(x)
        branches = [
            self.spatial_branch(x),
            self.channel_branch(x),
            self.multiscale_branch(x),
        ]
        feat_fine = self.branch_fuse(torch.cat(branches, dim=1))
        if self.use_grn:
            feat_fine = self.grn_fuse(feat_fine)
        feat_mid = self.stage1(feat_fine)
        feat_coarse_pre = self.stage2(feat_mid)
        feat_coarse = self.stage3(feat_coarse_pre)
        if self.use_grn:
            feat_coarse = self.grn_stage3(feat_coarse)
        up = self.up3(feat_coarse)
        up = self.fuse3(torch.cat([up, feat_mid], dim=1))
        if self.use_grn:
            up = self.grn_dec3(up)
        up = self.up2(up)
        decoder_out = self.fuse2(torch.cat([up, feat_fine], dim=1))
        if self.use_grn:
            decoder_out = self.grn_dec2(decoder_out)
        return feat_coarse, decoder_out

    def unified_feature(self, x):
        """供 Proto 元学习用: 输出 unified flat 特征向量 [B, unified_dim]."""
        feat_coarse, decoder_out = self.encode(x)
        coarse_pooled = self.gap_coarse(feat_coarse).flatten(1)
        fine_pooled = self.gap_fine(decoder_out).flatten(1)
        if self.use_rnn:
            # 钳制 H=16 让 lstm_proj 入口 (64*16=1024) 与原版兼容;
            # W 保留 (升采后 W=24, 原版 W=12), 序列长度自然适配 LSTM/GRU/Transformer.
            decoder_out_rnn = functional.adaptive_avg_pool2d(decoder_out, (16, None))
            b, c, h, w = decoder_out_rnn.shape
            seq = decoder_out_rnn.permute(3, 0, 1, 2).reshape(w, b, c * h)
            seq = self.lstm_proj(seq)
            if self.rnn_cell == 'transformer':
                seq = seq + self.pos_encoding[:seq.size(0)]
                out = self.lstm(seq)         # [W, B, d]
                rnn_feat = out.mean(dim=0)   # 序列均值池化
            else:
                rnn_out = self.lstm(seq)
                # LSTM 返回 (out, (h, c)); GRU 返回 (out, h)
                hidden = rnn_out[1][0] if isinstance(rnn_out[1], tuple) else rnn_out[1]
                rnn_feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
            rnn_feat = self.rnn_dropout(rnn_feat)
            return torch.cat([coarse_pooled, fine_pooled, rnn_feat], dim=1)
        return torch.cat([coarse_pooled, fine_pooled], dim=1)

    def forward(self, x):
        unified = self.unified_feature(x)
        main_logits = self.head_main(unified)
        # 3 元组: (main_logits, unified_feat, None placeholder).
        # 留第 3 位 None 以便老消费者用 [0] 取主 logits 不出错; pretrain 走 encode() 不走 forward.
        return main_logits, unified, None


# 老模型 13 | Hybrid 变体: 嫁接预训练 ResNet18 stem
class HybridHandCharNetR18Stem(HybridHandCharNet):
    """HybridHandCharNet 子类, 把 stem 换成 ImageNet 预训练 ResNet18 的 conv1+layer1.

    设计要点:
    1. 仅替换 stem, 后续三分支 / stage1-3 / UNet decoder / BiLSTM / head_main 全继承
       — 你的原创架构 100% 保留, 仅低层 conv 借 ImageNet prior.
    2. 嫁接位置: ResNet18.conv1(7×7 s=2) + bn1 + relu + maxpool(s=2) + layer1(no down)
       → [B,64,16,12] (64×48 输入直送, 无需上采样, 完美对齐原 stem 输出空间维度)
       → 1×1 conv 64→32 (通道适配, 对齐原 stem 输出通道)
       → [B,32,16,12] 与原 stem 输出一致, 三分支接口完全兼容.
    3. 父类 _init_weights() 已改为 PyTorch 默认 init (pass), 注入预训练 stem 不会被 reset.
    4. 1ch 适配: ResNet18 预训练 conv1 权重 [64,3,7,7] 在通道维 sum → [64,1,7,7]
       (sum 而非 mean: 灰度等价 3ch 同值, sum 让响应幅度匹配原 RGB).

    输入: [B, 1, 64, 48]
    输出: (main_logits[B,62], unified_feat, decoder_out) — 与父类一致
    """
    def __init__(self, num_classes=62, pretrained=True, freeze_stem=False, **kwargs):
        # 先让父类按默认配置构造 + 跑完 _init_weights
        super().__init__(num_classes=num_classes, **kwargs)

        # 再注入预训练 stem (覆盖父类的 stem, _modules 字典自动重注册)
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # 1ch 适配: 预训练 conv1.weight [64,3,7,7] sum 到 [64,1,7,7]
        new_conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(backbone.conv1.weight.data.sum(dim=1, keepdim=True))
        backbone.conv1 = new_conv1

        adapter = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # adapter 走 Kaiming (跟父类风格一致)
        nn.init.kaiming_normal_(adapter[0].weight, mode='fan_out', nonlinearity='relu')

        self.stem = nn.Sequential(
            backbone.conv1,      # [B,64,32,24]
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,    # [B,64,16,12]  ← H,W 已对齐
            backbone.layer1,     # [B,64,16,12]  ← layer1 stride=1 不下采样
            adapter,             # [B,32,16,12]  ← 通道压到 32 对齐父类原 stem 输出
        )

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad_(False)


class _CnxFeatureWrap(nn.Module):
    """timm features_only 输出 list[Tensor], wrap 成 tensor 直出, 兼容父类 self.stem(x)."""
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return self.body(x)[0]


# 老模型 14 | Hybrid 变体: 嫁接预训练 ConvNeXtV2 stem
class HybridHandCharNetCnxStem(HybridHandCharNet):
    """HybridHandCharNet 子类, 把 stem 换成 ImageNet 预训练 ConvNeXtV2-Atto 的 stem+s1+s2,
    并把三分支扩成吃全量 80ch (主路加工, 非旁路).

    设计要点:
    1. 血统对齐: 父类可选 GRN 模块 (use_grn=True) 正是 ConvNeXtV2 核心创新, 嫁接同源
       预训练 → 路线一致, 不像 R18Stem 那样 ResNet stem + ConvNeXt GRN 冲突.
    2. 嫁接位置: timm features_only=True out_indices=(1,) 取到 stem + stage0 + stage1
       的出口, 在 128×96 输入下输出 [B,80,16,12] — 完美对齐父类三分支期望分辨率.
    3. 三分支主路化: 输入 ch 从父类原 32 → 80, 三路各出 16ch 不变 → concat 48 →
       branch_fuse 1×1 48→64 不变 → 后段 (stage1/2/3 + UNet decoder + RNN + head)
       全继承, 接口零偏移.
    4. 1ch 适配: timm 自动把预训练 conv [40,3,4,4] 在 dim=1 平均到 [40,1,4,4].
    5. 父类 _init_weights() 已改 PyTorch 默认 init (pass), 注入预训练 stem 不会被 reset.

    预训练覆盖: stem+s1+s2 ≈ 0.4M (~25% 总参), 远超 R18Stem 的 9.4%.

    输入: [B, 1, H, W] (encode 内 bilinear 升到 self.input_size, 默认 128×96)
    输出: (main_logits[B,62], unified_feat, decoder_out) — 与父类一致
    """
    def __init__(self, num_classes=62, pretrained=True, freeze_stem=False, **kwargs):
        # 父类按默认配置构造 (stem_c=32, 三分支按 32ch 输入), 之后我们整体覆盖
        super().__init__(num_classes=num_classes, **kwargs)

        # 注入 ConvNeXtV2-Atto stem+s1+s2 (features_only=True 直接到 stage1 出口)
        body = timm.create_model(
            'convnextv2_atto.fcmae_ft_in1k' if pretrained else 'convnextv2_atto',
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            out_indices=(1,),
        )
        self.stem = _CnxFeatureWrap(body)

        # 覆盖三分支: 输入 ch 32 → 80 (吃 ConvNeXt stage1 出口全量), 输出仍 16ch/路
        branch_c = 16
        stem_c_new = 80
        self.spatial_branch = nn.Sequential(
            SpatialAttentionBlock(),
            ConvBlock(stem_c_new, branch_c, 3, 1, 1),
        )
        self.channel_branch = nn.Sequential(
            ECABlock(stem_c_new),
            ConvBlock(stem_c_new, branch_c, 3, 1, 1),
        )
        self.multiscale_branch = MultiScalePool(stem_c_new, branch_c)
        # branch_fuse (48→64) 不变, 父类已构造, 沿用

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad_(False)


class _CnxBypassBase(nn.Module):
    """完整 ConvNeXtV2-Atto backbone + 三分支旁路 (tap from stage0 出口).

    设计要点 (基类, 不直接实例化, 由子类 A/B 决定融合方式):
    1. **完整 backbone 保留**: timm convnextv2_atto.fcmae_ft_in1k, in_chans=1,
       num_classes=0, global_pool='avg'. 拆 stem / stages[0..3] / head 手动 forward,
       预训练覆盖率 ≈ 100% (除三分支 + fuse 层).
    2. **三分支 tap 点**: stage0 出口 [40, 24, 24] (96 输入下), 三路各出 16ch.
    3. **分辨率对齐**: 三分支 concat 出 [48, 24, 24] → MaxPool(2) → [48, 12, 12],
       与 stage1 出口 [80, 12, 12] 同尺寸.
    4. **head**: backbone.head 已含 GAP+norm, num_classes=0 时 fc 是 Identity.
       自带 fc(320→62) 接收 GAP 后特征.
    5. 1ch 适配 timm 自动 mean over dim=1.

    输入: [B,1,H,W] (内部 bilinear 升采到 input_size=96)
    输出: (logits[B,62], unified[B,320], None) 兼容 3 元组接口
    """
    def __init__(self, num_classes=62, pretrained=True, input_size=96,
                 drop_path_rate=0.1, head_dropout=0.1, freeze_stem=False):
        super().__init__()
        self.input_size = input_size
        backbone = timm.create_model(
            'convnextv2_atto.fcmae_ft_in1k' if pretrained else 'convnextv2_atto',
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            drop_path_rate=drop_path_rate,
            global_pool='avg',
        )
        self.stem = backbone.stem
        self.stage0 = backbone.stages[0]
        self.stage1 = backbone.stages[1]
        self.stage2 = backbone.stages[2]
        self.stage3 = backbone.stages[3]
        self.head_pool = backbone.head   # ClassifierHead, num_classes=0 → norm+GAP+Identity

        # 三分支吃 stage0 出口 40ch
        stem_c = 40
        branch_c = 16
        self.spatial_branch = nn.Sequential(
            SpatialAttentionBlock(),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.channel_branch = nn.Sequential(
            ECABlock(stem_c),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.multiscale_branch = MultiScalePool(stem_c, branch_c)
        self.branch_pool = nn.MaxPool2d(2)   # 对齐 stage1 出口分辨率

        self.feat_dim = 320                   # ConvNeXtV2-Atto stage3 通道
        self.head_dropout = nn.Dropout(p=head_dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad_(False)
            for p in self.stage0.parameters():
                p.requires_grad_(False)

    def _branches(self, x):
        """x: stage0 出口 [B,40,24,24] → [B,48,12,12] (concat + pool)."""
        b = torch.cat([
            self.spatial_branch(x),
            self.channel_branch(x),
            self.multiscale_branch(x),
        ], dim=1)
        return self.branch_pool(b)

    def _fuse(self, main, branch):
        raise NotImplementedError

    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = functional.interpolate(x, size=(self.input_size, self.input_size),
                                       mode='bilinear', align_corners=False)
        x = self.stem(x)                # [B,40,24,24]
        x = self.stage0(x)              # [B,40,24,24]
        branch = self._branches(x)      # [B,48,12,12]
        main = self.stage1(x)           # [B,80,12,12]
        fused = self._fuse(main, branch)  # [B,80,12,12]
        h = self.stage2(fused)          # [B,160,6,6]
        h = self.stage3(h)              # [B,320,3,3]
        unified = self.head_pool(h)     # [B,320] (norm+GAP)
        logits = self.fc(self.head_dropout(unified))
        return logits, unified, None


# 老模型 15 | Hybrid 变体: 完整 ConvNeXt 骨干 + 三分支 concat 融合
class HybridHandCharNetCnxBypassA(_CnxBypassBase):
    """方案 A: 三分支与 stage1 并联 concat 融合 (修改 stage2 输入分布).

    fuse: concat([stage1_out 80ch, branch 48ch]) → 1×1 → 80ch → stage2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_concat = ConvBlock(80 + 48, 80, 1, 1, 0)

    def _fuse(self, main, branch):
        return self.fuse_concat(torch.cat([main, branch], dim=1))


# 老模型 16 | Hybrid 变体: 完整 ConvNeXt 骨干 + 三分支残差 add
class HybridHandCharNetCnxBypassB(_CnxBypassBase):
    """方案 B: 三分支旁路残差 add 到 stage1 出口 (1×1 投影 init=0).

    fuse: stage1_out + proj(branch), proj weight/bias 全 0 → 训练开始等价纯 backbone,
    三分支贡献随训练渐进注入, 不扰动 stage2 预训练工作点.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_add = nn.Conv2d(48, 80, 1, 1, 0)
        nn.init.zeros_(self.proj_add.weight)
        nn.init.zeros_(self.proj_add.bias)

    def _fuse(self, main, branch):
        return main + self.proj_add(branch)


# ====== RefinedHybridNet: UNet 4 级骨架, 4 模块分级配置 ======
# 老模型 17 | Hybrid 变体: UNet 4 级骨架 + 分级模块配置
class RefinedHybridNet(nn.Module):
    """以 UNet 4 级编解码为骨架, 把 4 个特殊模块按 inductive bias 分到各 encoder 级:
      L1 (1/1 高分辨率): SpatialAttention   — 笔画局部空间位置
      L2 (1/2 中分辨率): MultiScalePool     — 聚合多尺度上下文
      L3 (1/4 低分辨率): FrequencyBlock     — 提取高层频率模式
      Bottleneck (1/8): ECABlock            — 通道路由精炼
    解码器每级 skip 连接 (concat + 1×1 conv + ResBlock + ECA).
    分类头: GAP(bottleneck) ⊕ GAP(decoder_L1) ⊕ 可选 RNN(笔顺 W=48 步) → 3 头.
    """
    def __init__(self, num_classes=62, aux_classes=3, dropout=0.146, use_rnn=True,
                 base_ch=32, rnn_cell='gru', rnn_hidden=64, rnn_layers=1, rnn_proj_dim=128,
                 rnn_input_ch=16, rnn_attach_to='dec1', use_grn=False,
                 use_arcface=False, head_bottleneck=256, arcface_scale=30.0):
        """rnn_attach_to: 'bottleneck' | 'dec3' | 'dec2' | 'dec1' — RNN 扫描在哪一级特征图.

        默认 'dec1' (全分辨率 W=48), 但参数最多且最易过拟合;
        'bottleneck' (W=6) 参数最少, 看高层语义; 'dec3'/'dec2' 中间.

        use_grn: ConvNeXt V2 GRN 模块, 插入 bottleneck 和每个 decoder block 末尾,
        残差形式 (γ 初始 0), 抑制 feature collapse, 对小数据正则化友好.
        """
        super().__init__()
        self.use_rnn = use_rnn
        self.use_grn = use_grn
        self.rnn_cell = rnn_cell.lower() if use_rnn else None
        self.rnn_attach_to = rnn_attach_to.lower() if use_rnn else None

        c1 = base_ch
        c2 = int(base_ch * 1.5)
        c3 = int(base_ch * 2.25)
        c4 = int(base_ch * 3.0)

        # Stem (不下采样, 通道扩展)
        self.stem = ConvBlock(1, c1, 7, 1, 3)

        # Encoder L1: Spatial Attention
        self.enc1 = ResBlock(c1, c1)
        self.enc1_attn = SpatialAttentionBlock()
        self.pool1 = nn.MaxPool2d(2)

        # Encoder L2: Multi-scale 上下文
        self.enc2 = ResBlock(c1, c2)
        self.enc2_ms = MultiScalePool(c2, c2)
        self.pool2 = nn.MaxPool2d(2)

        # Encoder L3: Frequency 频域
        self.enc3 = ResBlock(c2, c3)
        self.enc3_freq = FrequencyBlock(c3, c3)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck: 双 ResBlock + ECA 通道注意力 (+ 可选 GRN)
        bottleneck_layers = [ResBlock(c3, c4), ECABlock(c4), ResBlock(c4, c4)]
        if use_grn:
            bottleneck_layers.append(GRN(c4))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder L3: Up + skip3 + refine
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        dec3_layers = [ConvBlock(c3 * 2, c3, 1, 1, 0), ResBlock(c3, c3), ECABlock(c3)]
        if use_grn:
            dec3_layers.append(GRN(c3))
        self.dec3 = nn.Sequential(*dec3_layers)
        # Decoder L2: Up + skip2 + refine
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        dec2_layers = [ConvBlock(c2 * 2, c2, 1, 1, 0), ResBlock(c2, c2), ECABlock(c2)]
        if use_grn:
            dec2_layers.append(GRN(c2))
        self.dec2 = nn.Sequential(*dec2_layers)
        # Decoder L1: Up + skip1 + refine
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        dec1_layers = [ConvBlock(c1 * 2, c1, 1, 1, 0), ResBlock(c1, c1), ECABlock(c1)]
        if use_grn:
            dec1_layers.append(GRN(c1))
        self.dec1 = nn.Sequential(*dec1_layers)

        # 可选 RNN 扫描笔顺. 4 个挂载点 (基于 64×48 输入):
        #   bottleneck: c4 ch,  8×6  → W=6  最少参数, 看高层语义
        #   dec3:       c3 ch, 16×12 → W=12 中等抽象
        #   dec2:       c2 ch, 32×24 → W=24 中等细节
        #   dec1:       c1 ch, 64×48 → W=48 最细 (易过拟合)
        if use_rnn:
            attach_info = {
                'bottleneck': (c4, 8, 6),
                'dec3':       (c3, 16, 12),
                'dec2':       (c2, 32, 24),
                'dec1':       (c1, 64, 48),
            }
            if self.rnn_attach_to not in attach_info:
                raise ValueError("rnn_attach_to must be one of %s" % list(attach_info.keys()))
            src_ch, src_h, src_w = attach_info[self.rnn_attach_to]
            self.rnn_src_h = src_h
            self.rnn_src_w = src_w
            self.rnn_chan_reduce = nn.Conv2d(src_ch, rnn_input_ch, 1, bias=False)
            self.lstm_proj = nn.Linear(rnn_input_ch * src_h, rnn_proj_dim)
            if self.rnn_cell == 'transformer':
                self.pos_encoding = nn.Parameter(torch.zeros(src_w, 1, rnn_proj_dim))
                nn.init.normal_(self.pos_encoding, std=0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=rnn_proj_dim, nhead=4,
                    dim_feedforward=rnn_proj_dim * 2,
                    dropout=dropout, batch_first=False,
                )
                self.lstm = nn.TransformerEncoder(encoder_layer, num_layers=rnn_layers)
                rnn_dim = rnn_proj_dim
            else:
                rnn_cls = nn.GRU if self.rnn_cell == 'gru' else nn.LSTM
                self.lstm = rnn_cls(rnn_proj_dim, rnn_hidden, num_layers=rnn_layers,
                                    batch_first=False, bidirectional=True,
                                    dropout=dropout if rnn_layers > 1 else 0.0)
                rnn_dim = rnn_hidden * 2
            self.rnn_dropout = nn.Dropout(dropout * 2)
        else:
            rnn_dim = 0

        # 多尺度 GAP 头: bottleneck + decoder_L1
        self.gap_bottleneck = nn.AdaptiveAvgPool2d(1)
        self.gap_dec1 = nn.AdaptiveAvgPool2d(1)
        unified_dim = c4 + c1 + rnn_dim
        self.unified_dim = unified_dim

        self.use_arcface = use_arcface
        if use_arcface:
            self.head_main = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(unified_dim, head_bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                ArcMarginProduct(head_bottleneck, num_classes, scale=arcface_scale),
            )
        else:
            self.head_main = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(unified_dim, head_bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_bottleneck, num_classes),
            )
        # head_aux: 单线性 → 小 MLP (unified_dim → 64 → 3)
        self.head_aux = nn.Sequential(
            nn.Linear(unified_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, aux_classes),
        )
        # contrastive head 删除: 训练流程未启用, 占参数无收益

    def encode(self, x):
        """编码 + 解码全过, 返回所有中间特征 dict."""
        x = self.stem(x)                                  # 64×48
        e1 = self.enc1_attn(self.enc1(x))                 # 64×48 (skip1)
        e2 = self.enc2_ms(self.enc2(self.pool1(e1)))      # 32×24 (skip2)
        e3 = self.enc3_freq(self.enc3(self.pool2(e2)))    # 16×12 (skip3)
        bottleneck = self.bottleneck(self.pool3(e3))      # 8×6
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))  # 16×12
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))          # 32×24
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))          # 64×48
        return {'bottleneck': bottleneck, 'dec3': d3, 'dec2': d2, 'dec1': d1}

    def unified_feature(self, x):
        feats = self.encode(x)
        bottleneck = feats['bottleneck']
        d1 = feats['dec1']
        b_pool = self.gap_bottleneck(bottleneck).flatten(1)
        d1_pool = self.gap_dec1(d1).flatten(1)
        if self.use_rnn:
            src = feats[self.rnn_attach_to]
            src_reduced = self.rnn_chan_reduce(src)        # [B, rnn_input_ch, H, W]
            b, c, h, w = src_reduced.shape
            seq = src_reduced.permute(3, 0, 1, 2).reshape(w, b, c * h)
            seq = self.lstm_proj(seq)
            if self.rnn_cell == 'transformer':
                seq = seq + self.pos_encoding[:seq.size(0)]
                out = self.lstm(seq)
                rnn_feat = out.mean(dim=0)
            else:
                rnn_out = self.lstm(seq)
                hidden = rnn_out[1][0] if isinstance(rnn_out[1], tuple) else rnn_out[1]
                rnn_feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
            rnn_feat = self.rnn_dropout(rnn_feat)
            return torch.cat([b_pool, d1_pool, rnn_feat], dim=1)
        return torch.cat([b_pool, d1_pool], dim=1)

    def forward(self, x):
        unified = self.unified_feature(x)
        main_logits = self.head_main(unified)
        aux_logits = self.head_aux(unified)
        return main_logits, aux_logits, None, unified


# ====== ResNet50UNetChar: ResNet50 4 级 + 轻量 UNet decoder + 可选 BiLSTM ======
# 老模型 18 | ResNet50 编码器 + 轻量 UNet 解码器 + 可选 BiLSTM
class ResNet50UNetChar(nn.Module):
    """ResNet50 (torchvision) 当 encoder, 删 maxpool 保 32×24 → UNet 3 级 decoder.

    通道分布:
      conv1(s=2)+bn+relu       -> [B, 64,  32, 24]
      layer1 (s=1)             -> [B, 256, 32, 24]  skip1
      layer2 (s=2)             -> [B, 512, 16, 12]  skip2
      layer3 (s=2)             -> [B, 1024, 8,  6]  skip3
      layer4 (s=2)             -> [B, 2048, 4,  3]  bottleneck

    decoder 全部统一到 dec_ch (默认 128) 以控参数; 1×1 lateral + bilinear up + ConvBlock+ResBlock.
    head: GAP(lat4) ⊕ GAP(dec_top) ⊕ optional LSTM(扫 dec_top W=24).
    forward 返回 (main_logits, unified, None) - 跟 HybridHandCharNet 同接口.
    """
    def __init__(self, num_classes=62, dec_ch=128, dropout=0.2, use_rnn=True,
                 rnn_hidden=128, rnn_layers=2, rnn_proj_dim=256, rnn_input_ch=16,
                 head_bottleneck=256):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        backbone.maxpool = nn.Identity()  # 保 32×24 分辨率给 UNet skip
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        c = dec_ch
        # lateral 1×1 投到统一通道
        self.lat4 = nn.Conv2d(2048, c, 1, bias=False)
        self.lat3 = nn.Conv2d(1024, c, 1, bias=False)
        self.lat2 = nn.Conv2d(512, c, 1, bias=False)
        self.lat1 = nn.Conv2d(256, c, 1, bias=False)
        # decoder refine: cat(up, skip) → 2c → ConvBlock+ResBlock → c
        self.dec3 = nn.Sequential(ConvBlock(c * 2, c, 3, 1, 1), ResBlock(c, c))
        self.dec2 = nn.Sequential(ConvBlock(c * 2, c, 3, 1, 1), ResBlock(c, c))
        self.dec1 = nn.Sequential(ConvBlock(c * 2, c, 3, 1, 1), ResBlock(c, c))

        self.use_rnn = use_rnn
        if use_rnn:
            # decoder top [c, 32, 24] → 通道压到 rnn_input_ch=16 → W=24 步, 每步 16*32 = 512 维
            self.rnn_chan_reduce = nn.Conv2d(c, rnn_input_ch, 1, bias=False)
            self.lstm_proj = nn.Linear(rnn_input_ch * 32, rnn_proj_dim)
            self.lstm = nn.LSTM(rnn_proj_dim, rnn_hidden, num_layers=rnn_layers,
                                batch_first=False, bidirectional=True,
                                dropout=dropout if rnn_layers > 1 else 0.0)
            self.rnn_dropout = nn.Dropout(dropout * 2)
            rnn_dim = rnn_hidden * 2
        else:
            rnn_dim = 0

        self.gap_b = nn.AdaptiveAvgPool2d(1)
        self.gap_d = nn.AdaptiveAvgPool2d(1)
        unified_dim = c + c + rnn_dim
        self.unified_dim = unified_dim
        self.head_main = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(unified_dim, head_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_bottleneck, num_classes),
        )
        self._init_added_weights()

    def _init_added_weights(self):
        """只初始化 decoder + LSTM + head; ResNet backbone 用 torchvision 默认 init."""
        backbone_prefixes = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
        for name, m in self.named_modules():
            if name.startswith(backbone_prefixes):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # [B, 64, 32, 24]
        e1 = self.layer1(x)                       # [B, 256, 32, 24]
        e2 = self.layer2(e1)                      # [B, 512, 16, 12]
        e3 = self.layer3(e2)                      # [B, 1024, 8, 6]
        e4 = self.layer4(e3)                      # [B, 2048, 4, 3]
        l4 = self.lat4(e4); l3 = self.lat3(e3)
        l2 = self.lat2(e2); l1 = self.lat1(e1)
        d3 = self.dec3(torch.cat([functional.interpolate(
            l4, size=l3.shape[2:], mode='bilinear', align_corners=False), l3], dim=1))
        d2 = self.dec2(torch.cat([functional.interpolate(
            d3, size=l2.shape[2:], mode='bilinear', align_corners=False), l2], dim=1))
        d1 = self.dec1(torch.cat([functional.interpolate(
            d2, size=l1.shape[2:], mode='bilinear', align_corners=False), l1], dim=1))
        return l4, d1

    def forward(self, x):
        bottleneck_lat, dec_top = self.encode(x)
        b_pool = self.gap_b(bottleneck_lat).flatten(1)   # [B, c]
        d_pool = self.gap_d(dec_top).flatten(1)          # [B, c]
        if self.use_rnn:
            r = self.rnn_chan_reduce(dec_top)            # [B, rnn_input_ch, 32, 24]
            b, c, h, w = r.shape
            seq = r.permute(3, 0, 1, 2).reshape(w, b, c * h)  # [W=24, B, c*32]
            seq = self.lstm_proj(seq)
            _, (h_n, _) = self.lstm(seq)
            rnn_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            rnn_feat = self.rnn_dropout(rnn_feat)
            unified = torch.cat([b_pool, d_pool, rnn_feat], dim=1)
        else:
            unified = torch.cat([b_pool, d_pool], dim=1)
        main_logits = self.head_main(unified)
        # 3 元组接口兼容 train / evaluate
        return main_logits, unified, None


# ====== TransferBackbone: 统一迁移学习骨干 (3ch + ImageNet 归一化 + 强 backbone) ======
# ---- 推荐骨干清单 + 友好报错 ----


# 新模型 04 | DisentangledNet —— 尺寸感知双流残差网络
# 形状流(GAP, 刻意尺寸盲) + 结构流(多尺度GeM, 尺寸敏感) → 残差 logits
# logits_final = logits_shape + Δ,  Δ L1 稀疏 → 只在歧义对上非零
class DisentangledNet(nn.Module):
    """双流解耦 + 残差分类头.

    Stream S「形状 = 恒等路径」: backbone stage4 → GAP → logits_shape [B,62]
        GAP 抹掉空间位置和绝对尺寸, C/c 在此路上天然接近.
    Stream G「结构 = 残差路径」: backbone stage1-4 → 多尺度 GeM → f_geo
        → [f_shape ⊕ f_geo] → Δ [B,62]. 多尺度保留尺寸信号, 区分歧义对.
    logits_final = logits_shape + Δ, L1(Δ) 强制 Δ 稀疏.

    返回 (logits_final, pooled_feat, logits_shape)
        pooled_feat = concat(f_shape, f_geo) 供 SIGReg/freeze 等下游.
    """
    def __init__(self, model_name='convnextv2_femto', num_classes=62, pretrained=True,
                 input_size=160, shape_dim=192, geo_dim=192):
        super().__init__()
        import timm
        in_chans = 3
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, in_chans=in_chans,
                features_only=True, out_indices=(0, 1, 2, 3))
        except Exception as e:
            raise RuntimeError(
                "DisentangledNet needs backbone with features_only. "
                "ViT/DINOv2 not supported, use CNN (convnextv2_femto/efficientnet/resnet). "
                "Error: %s" % str(e)) from e

        # 预处理: 灰度→3ch + resize + ImageNet 归一化
        self.input_size = input_size
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        self.register_buffer('_mean', torch.tensor(imagenet_mean).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(imagenet_std).view(1, 3, 1, 1))

        # 取各 stage 输出通道数 (一次假前向)
        dummy = torch.randn(1, in_chans, input_size, input_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
        stage_dims = [f.shape[1] for f in feats]  # [s1_dim, s2_dim, s3_dim, s4_dim]

        # --- Stream S: 形状流 ---
        self.shape_pool = nn.AdaptiveAvgPool2d(1)       # GAP → 尺寸盲
        self.shape_proj = nn.Sequential(
            nn.Linear(stage_dims[3], shape_dim),
            nn.LayerNorm(shape_dim),
            nn.GELU(),
        )
        self.shape_head = nn.Linear(shape_dim, num_classes)

        # --- Stream G: 结构流 (多尺度 GeM) ---
        self.geo_gems = nn.ModuleList([GeMPool() for _ in stage_dims])
        self.geo_concat_dim = sum(stage_dims)  # s1+s2+s3+s4 通道和
        self.geo_proj = nn.Sequential(
            nn.Linear(self.geo_concat_dim, geo_dim),
            nn.LayerNorm(geo_dim),
            nn.GELU(),
        )

        # --- 残差融合 ---
        self.residual_fc = nn.Sequential(
            nn.Linear(shape_dim + geo_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))  # 0-init: 初始 Δ=0

    def forward(self, x):
        # 预处理: resize → 灰度1→3ch → ImageNet 归一化
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = torch.nn.functional.interpolate(x, size=(self.input_size, self.input_size),
                                                mode='bilinear', align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self._mean) / self._std

        feats = self.backbone(x)  # [s1,s2,s3,s4]

        # Stream S: 形状
        s4 = feats[3]                                  # [B, C4, H4, W4]
        f_shape = self.shape_proj(
            self.shape_pool(s4).flatten(1))            # [B, shape_dim]
        logits_shape = self.shape_head(f_shape)        # [B, 62]

        # Stream G: 结构
        geo_parts = []
        for gem, f in zip(self.geo_gems, feats):
            geo_parts.append(gem(f))                   # each [B, C_i]
        f_geo = self.geo_proj(torch.cat(geo_parts, 1)) # [B, geo_dim]

        # 残差修正
        fused = torch.cat([f_shape, f_geo], dim=1)     # [B, shape_dim+geo_dim]
        Delta = self.residual_scale * self.residual_fc(fused)  # [B, 62]
        logits_final = logits_shape + Delta

        # pooled_feat 给 SIGReg / freeze 等下游; 第3位放 logits_shape 给残差损失
        return logits_final, fused, logits_shape

    # ---- 判别式LR / 冻结接口 (复用 _TransferCommon 的规范) ----
    def param_groups(self, base_lr, backbone_lr_mult=0.1, weight_decay=5e-4):
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        backbone_p, head_p = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (backbone_p if id(p) in backbone_ids else head_p).append(p)
        groups = [
            {'params': head_p, 'lr': base_lr, 'weight_decay': weight_decay},
        ]
        if backbone_p:
            groups.append({'params': backbone_p, 'lr': base_lr * backbone_lr_mult,
                           'weight_decay': weight_decay})
        return groups

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

# name: (是否支持 features_only(可用于 transfer_ms/transfer_adapter), 备注)
RECOMMENDED_BACKBONES = {
    "convnextv2_femto":         (True,  "推荐核心: 小数据甜点 ~5M (默认)"),
    "convnextv2_pico":          (True,  "中档备选 ~9M, 想要更多容量时用"),
    "convnextv2_atto":          (True,  "最激进轻量 ~3.4M (可能欠拟合, 作消融对照)"),
    "tf_efficientnet_b0":       (True,  "高效异构成员 ~4M"),
    "resnet18":                 (True,  "稳健基线/异构成员 ~11M (原 91% 骨干)"),
    "convnextv2_nano":          (True,  "偏大 ~15M, 小数据可能过拟合, 慎用"),
    "vit_small_patch14_dinov2": (False, "DINOv2 冻结特征强; 仅 --arch transfer; 输入须 14 的倍数(如 224)"),
}


def _friendly_backbone_error(model_name, msg, features_only):
    rec = ", ".join(sorted(RECOMMENDED_BACKBONES))
    if features_only and "feature" in msg.lower():
        return RuntimeError(
            "骨干 '%s' 不支持 features_only (transfer_ms / transfer_adapter 需要多尺度特征)。"
            "ViT/DINOv2 这类单尺度骨干请改用 --arch transfer。推荐骨干: %s。原始错误: %s"
            % (model_name, rec, msg))
    return RuntimeError(
        "创建 timm 骨干 '%s' 失败 (名字是否正确? 是否 pip install timm?)。"
        "推荐骨干: %s。原始错误: %s" % (model_name, rec, msg))


# 新组件 00 | _TransferCommon —— 迁移骨干公共底座 (预处理 + 判别式LR/冻结, 被下面共用)
class _TransferCommon(nn.Module):
    """灰度3ch + ImageNet 归一化预处理, 判别式学习率 / 渐进解冻。

    子类需: __init__ 内创建 self.backbone 并调用 self._setup_norm(...);
            实现 _backbone_param_ids() 返回"预训练骨干"参数 id 集合 (小 lr / 冻结的那部分)。
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _setup_norm(self, input_size, gray_to_3ch, imagenet_norm):
        self.input_size = input_size
        self.gray_to_3ch = gray_to_3ch
        self.imagenet_norm = imagenet_norm
        self.register_buffer('_mean', torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x):
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = functional.interpolate(x, size=(self.input_size, self.input_size),
                                       mode='bilinear', align_corners=False)
        if self.gray_to_3ch and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.imagenet_norm:
            x = (x - self._mean) / self._std
        return x

    def _backbone_param_ids(self):
        raise NotImplementedError

    def param_groups(self, base_lr, backbone_mult=0.1, weight_decay=5e-4):
        """两组: 新建 head 用 base_lr, 预训练骨干用 base_lr*backbone_mult。"""
        bb_ids = self._backbone_param_ids()
        head, backbone = [], []
        for prm in self.parameters():
            if not prm.requires_grad:
                continue
            (backbone if id(prm) in bb_ids else head).append(prm)
        groups = [{'params': head, 'lr': base_lr, 'weight_decay': weight_decay}]
        if backbone:
            groups.append({'params': backbone, 'lr': base_lr * backbone_mult,
                           'weight_decay': weight_decay})
        return groups

    def freeze_backbone(self):
        bb_ids = self._backbone_param_ids()
        for prm in self.parameters():
            prm.requires_grad_(id(prm) not in bb_ids)

    def unfreeze_backbone(self):
        for prm in self.parameters():
            prm.requires_grad_(True)


# 新模型 01 | TransferBackbone —— 迁移骨干: 3ch+ImageNet归一化+任意timm骨干
class TransferBackbone(_TransferCommon):
    """timm 任意 ImageNet 预训练骨干, 按迁移学习最佳实践配置。

    依据 (F:/skills):
      dl-pretraining-alignment / dl-lightweight-multilabel 铁律: 小数据 (<5K) 默认用
      强通用预训练源 (ImageNet), "特征质量 > 域匹配"; 同域低分辨率 EMNIST 当预训练源反而弱。
      所以这里走 ImageNet 预训练骨干, 并修正三处常见迁移损耗:

    1. 灰度复制成 3 通道 (gray_to_3ch): 不再把预训练 conv1 在通道维 sum/mean 压成 1ch
       (那样会破坏第一层滤波器统计), 而是把 [B,1,H,W] 复制成 [B,3,H,W], 预训练 conv1 原样保留。
    2. ImageNet 归一化 (imagenet_norm): 预训练模型期望 ImageNet mean/std 输入分布,
       直接喂 [0,1] 会让迁移先验对不上。注意原始管线是 ink=1/paper=0 反相图,
       复制 3ch 后按 ImageNet 统计标准化即可。
    3. 分辨率可配 (input_size): ResNet/ConvNeXt 等设计分辨率远高于 64×48, 升采到 128/160/192
       让预训练特征工作在接近原生的分辨率。(ResNet/ConvNeXt 无位置编码, 直接 resize 即可;
       若换 ViT 需按 dl-vision-backbones-ssl 铁律 2 插值位置编码。)

    判别式学习率 / 渐进解冻接口:
      param_groups(base_lr, backbone_mult): backbone 用 base_lr*backbone_mult, 新建 head 用 base_lr。
      freeze_backbone() / unfreeze_backbone(): 配合 train --freeze_backbone_epochs 先冻后解。

    forward 返回 (logits[B,num_classes], pooled_feat, None), 兼容 train 3 元组接口。
    预处理/判别式LR/冻结继承自 _TransferCommon。
    """
    def __init__(self, model_name='resnet50', num_classes=62, pretrained=True,
                 input_size=128, gray_to_3ch=True, imagenet_norm=True,
                 head_dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        in_chans = 3 if gray_to_3ch else 1
        # 部分骨干 (resnet) 不支持 drop_path_rate, 故 try 两段式创建; 外层包友好报错
        try:
            try:
                self.backbone = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=num_classes,
                    in_chans=in_chans, drop_rate=head_dropout, drop_path_rate=drop_path_rate)
            except TypeError:
                self.backbone = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=num_classes,
                    in_chans=in_chans, drop_rate=head_dropout)
        except Exception as e:
            raise _friendly_backbone_error(model_name, str(e), features_only=False) from e
        self._setup_norm(input_size, gray_to_3ch, imagenet_norm)

    def forward(self, x):
        x = self._preprocess(x)
        feats = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # [B, feat_dim]
        logits = self.backbone.forward_head(feats)                   # [B, num_classes]
        return logits, pooled, None

    def _backbone_param_ids(self):
        # 分类模式: "骨干" = self.backbone 除分类头外的参数 (分类头算新建 head, 用大 lr)
        clf = {id(p) for p in self.backbone.get_classifier().parameters()}
        return {id(p) for p in self.backbone.parameters()} - clf


# ====== 架构优化: GeM 池化 + 多尺度头 + 三分支 Adapter (均接在预训练特征之上) ======
# 新组件 02 | GeMPool —— 广义均值池化 (强度敏感, 可学习 p)
class GeMPool(nn.Module):
    """Generalized Mean Pooling: (mean(x^p))^(1/p), 可学习 p。

    强度敏感池化: p=1 退化为 GAP; p 越大越突出"强响应/深墨迹"区域, p→∞ 趋近 max pool。
    对手写字符这种"笔画强度"有判别意义的任务, 比纯 GAP 常 +0.5~1。
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):  # [B,C,H,W] -> [B,C]
        p = self.p.clamp(min=1.0)
        pooled = functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1)
        return pooled.pow(1.0 / p).flatten(1)


def _intensity_stats(x):
    """强度统计池化: 对空间维取 std + max, 捕捉激活强度的分布 (浓淡/对比/峰值)。

    与 (GeM≈强度加权均值) 互补: std 描述强度离散度 (笔画粗细/均匀度),
    max 描述峰值强度 (最深墨迹)。返回 [B, 2C]。
    """
    b, c, _, _ = x.shape
    flat = x.flatten(2)                       # [B,C,HW]
    std = flat.std(dim=2)                     # [B,C]
    mx = flat.max(dim=2)[0]                   # [B,C]
    return torch.cat([std, mx], dim=1)        # [B,2C]


# 新组件 01 | _TransferBase —— 迁移骨干公共底座 (features_only + 预处理 + 判别式LR/冻结)
class _TransferBase(_TransferCommon):
    """预训练 features_only 骨干 (多尺度特征); 预处理/判别式LR/冻结继承自 _TransferCommon。"""

    def __init__(self, model_name, pretrained, input_size, gray_to_3ch, imagenet_norm):
        super().__init__()
        in_chans = 3 if gray_to_3ch else 1
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, in_chans=in_chans, features_only=True)
        except Exception as e:
            raise _friendly_backbone_error(model_name, str(e), features_only=True) from e
        self._setup_norm(input_size, gray_to_3ch, imagenet_norm)

    def _backbone_param_ids(self):
        # features_only 模式: 整个 self.backbone 都是预训练骨干 (head 是外挂的 self.head)
        return {id(p) for p in self.backbone.parameters()}


# 新模型 02 | TransferBackboneMS —— 多尺度 GeM 头 (FPN 式聚合)
class TransferBackboneMS(_TransferBase):
    """多尺度特征聚合头 + GeM 池化 (架构优化方案 1)。

    取预训练骨干最后 ms_stages 个 stage 的特征图 (浅层抓笔画细节 / 深层抓整体字形),
    各自 GeM 池化后拼接 → LayerNorm → 分类头。FPN 式多尺度, 对"一笔之差"的字符对症。
    输出 (logits, pooled_feat, None)。
    """
    def __init__(self, model_name='resnet50', num_classes=62, pretrained=True,
                 input_size=160, ms_stages=2, gray_to_3ch=True, imagenet_norm=True,
                 head_dropout=0.1):
        super().__init__(model_name, pretrained, input_size, gray_to_3ch, imagenet_norm)
        chans = self.backbone.feature_info.channels()
        self.ms_stages = min(ms_stages, len(chans))
        feat_dim = sum(chans[-self.ms_stages:])
        self.gem = GeMPool()
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(feat_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        feats = self.backbone(self._preprocess(x))
        sel = feats[-self.ms_stages:]
        pooled = self.norm(torch.cat([self.gem(f) for f in sel], dim=1))
        return self.fc(self.drop(pooled)), pooled, None


# ---- 特征分支工厂 (按名字组装, 便于自由搭配 + 消融) ----
# 新组件 05 | GaborBranch —— 方向选择性频率特征 (Gabor 核初始化)
class GaborBranch(nn.Module):
    """方向选择性频率特征 (替代被剪的 DCT 分支)。首层 conv 以 Gabor 核初始化, 可学习微调。

    Gabor 对笔画"方向 + 频率"双选择, 比 DCT 更贴合字符 (笔画是有向条纹)。
    """
    def __init__(self, in_c, out_c, n_orient=4, ksize=7):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, ksize, padding=ksize // 2, bias=False)
        self._init_gabor(in_c, out_c, ksize, n_orient)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def _init_gabor(self, in_c, out_c, ksize, n_orient):
        with torch.no_grad():
            coords = torch.arange(ksize).float() - ksize // 2
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            lam, sigma, gamma = 3.0, 2.0, 0.5
            for o in range(out_c):
                theta = math.pi * (o % n_orient) / n_orient
                x_t = xx * math.cos(theta) + yy * math.sin(theta)
                y_t = -xx * math.sin(theta) + yy * math.cos(theta)
                g = torch.exp(-(x_t ** 2 + (gamma ** 2) * y_t ** 2) / (2 * sigma ** 2)) \
                    * torch.cos(2 * math.pi * x_t / lam)
                self.conv.weight[o] = g.unsqueeze(0).repeat(in_c, 1, 1) / in_c

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 新组件 06 | MorphBranch —— 笔画/结构特征 (形态学梯度)
class MorphBranch(nn.Module):
    """笔画/结构特征: 形态学梯度 (dilate - erode) 高亮笔画轮廓与端点/交叉等拓扑结构。全可微。"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = ConvBlock(in_c, out_c, 3, 1, 1)

    def forward(self, x):
        dilate = functional.max_pool2d(x, 3, 1, 1)
        erode = -functional.max_pool2d(-x, 3, 1, 1)
        return self.proj(dilate - erode)


def make_branch(name, in_c, out_c):
    """特征分支工厂: 按名字返回一个 [B,in_c,H,W] -> [B,out_c,H,W] 的分支模块。"""
    name = name.lower()
    if name == "spatial":
        return nn.Sequential(SpatialAttentionBlock(), ConvBlock(in_c, out_c, 3, 1, 1))
    if name == "channel":
        return nn.Sequential(ECABlock(in_c), ConvBlock(in_c, out_c, 3, 1, 1))
    if name == "multiscale":
        return MultiScalePool(in_c, out_c)
    if name == "gabor":
        return GaborBranch(in_c, out_c)
    if name in ("morph", "stroke", "structural"):
        return MorphBranch(in_c, out_c)
    raise ValueError("unknown branch: %s (可选 spatial/channel/multiscale/gabor/morph)" % name)


def _make_codebook(num_classes, code_length, n_trials=300, seed=42):
    """生成 {-1,+1} 码本 [num_classes, code_length], 贪心最大化最小汉明距 (随机多次取最优)。"""
    g = torch.Generator().manual_seed(seed)
    best, best_d = None, -1.0
    for _ in range(n_trials):
        cand = (torch.randint(0, 2, (num_classes, code_length), generator=g) * 2 - 1).float()
        sim = cand @ cand.t()
        ham = (code_length - sim) / 2.0
        ham.fill_diagonal_(code_length)            # 忽略自身
        d = ham.min().item()
        if d > best_d:
            best_d, best = d, cand
    return best


# 新组件 07 | ECOCHead —— 纠错输出编码头 (编码理论)
class ECOCHead(nn.Module):
    """纠错输出编码头 (来自编码理论/通信; Dietterich & Bakiri 1995)。

    把"分类"当成"带纠错的信道传输": 每个类分配一个汉明距尽量大的 {-1,+1} 码字,
    fc 输出 code_length 个软比特 (tanh), 与码本做相关 = 到各码字的"接近度", argmax 即最近码字解码。
    混淆类码字距离大 -> 个别比特错了也能纠回, 等价信道编码纠噪。输出 [B, num_classes] 兼容 CE/focal。
    """
    def __init__(self, in_features, num_classes, code_length=127, scale=4.0, seed=42):
        super().__init__()
        self.fc = nn.Linear(in_features, code_length)
        self.register_buffer("codebook", _make_codebook(num_classes, code_length, seed=seed))
        self.scale = scale
        self.code_length = code_length

    def forward(self, feat):
        bits = torch.tanh(self.fc(feat))                       # [B, L] 软比特
        return bits @ self.codebook.t() * (self.scale / self.code_length)


# 新组件 08 | HierRoutingHead —— 分层路由头 (网络分层寻址)
class HierRoutingHead(nn.Module):
    """分层路由头 (来自网络分层寻址 / hierarchical softmax)。

    先判大类组 (数字/大写/小写, 标准 62 类 sorted 布局: 0-9 / 10-35 / 36-61),
    再路由到细类: 最终 logit = 细类 logit + 所属组 logit。大小写混淆 (C/c) 正好跨组,
    分层结构逼模型先做"大小写/数字"决策, 专治跨组混淆。
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        groups = self._default_groups(num_classes)
        self.num_groups = int(groups.max().item()) + 1
        self.register_buffer("class_to_group", groups)
        self.group_fc = nn.Linear(in_features, self.num_groups)
        self.fine_fc = nn.Linear(in_features, num_classes)

    @staticmethod
    def _default_groups(n):
        g = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            g[i] = 0 if i < 10 else (1 if i < 36 else 2)
        return g

    def forward(self, feat):
        group_logits = self.group_fc(feat)                     # [B, G]
        fine_logits = self.fine_fc(feat)                       # [B, C]
        return fine_logits + group_logits[:, self.class_to_group]


def make_head(name, in_features, num_classes, scale=30.0, code_length=127):
    """分类头工厂: linear / cosine(度量学习) / ecoc(纠错输出编码) / hier(分层路由)。"""
    name = name.lower()
    if name == "linear":
        head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        return head
    if name in ("cosine", "arcface"):
        return ArcMarginProduct(in_features, num_classes, scale=scale)
    if name == "ecoc":
        return ECOCHead(in_features, num_classes, code_length=code_length)
    if name == "hier":
        return HierRoutingHead(in_features, num_classes)
    raise ValueError("unknown head: %s (可选 linear/cosine/ecoc/hier)" % name)


# 新组件 03 | MomentPool —— 强度加权几何矩 (经典图像矩, 广义化 GAP)
class MomentPool(nn.Module):
    """强度加权几何矩池化 (源自经典图像矩, Hu 1962)。

    把每个通道的激活强度 I(x,y) 看作一个二维"质量分布", 提取其低阶矩:
        M00 (总质量) / 强度质心 (mu_x, mu_y) / 二阶中心矩 (展布 var_x, var_y)。
    经典图像矩定义 M_pq = Σ_{x,y} I(x,y)·x^p·y^q —— GAP 只是 M00 这一个特例;
    本模块把"强度落在哪、铺多开"显式编码出来 (位置 + 形状)。
    总质量按 log1p 压缩, 呼应 Weber-Fechner 的对数强度感知。返回 [B, C*5]。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):  # [B,C,H,W] -> [B, C*5]
        b, c, h, w = x.shape
        mass_map = functional.relu(x)                      # 强度非负 (墨迹质量)
        total = mass_map.sum(dim=(2, 3))                   # [B,C] = M00
        prob = mass_map / (total[..., None, None] + self.eps)
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        mu_x = (prob * xs).sum(dim=(2, 3))                 # 强度质心
        mu_y = (prob * ys).sum(dim=(2, 3))
        var_x = (prob * (xs - mu_x[..., None, None]) ** 2).sum(dim=(2, 3))  # 二阶展布
        var_y = (prob * (ys - mu_y[..., None, None]) ** 2).sum(dim=(2, 3))
        mass = torch.log1p(total)                          # Weber-Fechner 对数压缩
        feats = torch.stack([mass, mu_x, mu_y, var_x, var_y], dim=-1)       # [B,C,5]
        return feats.flatten(1)


# 新组件 04 | WeberNorm —— Weber 除性归一化 (对墨色深浅鲁棒)
class WeberNorm(nn.Module):
    """Weber-Fechner / 除性归一化前端: 响应 ∝ ΔI/I, 对全局墨色深浅 (不同笔/扫描) 鲁棒。

    x' = (x - localmean) / (|localmean| + eps), localmean 用 k×k 平均池化估计局部背景强度。
    """
    def __init__(self, ksize=5, eps=1e-3):
        super().__init__()
        self.ksize = ksize
        self.eps = eps

    def forward(self, x):
        local = functional.avg_pool2d(x, self.ksize, 1, self.ksize // 2)
        return (x - local) / (local.abs() + self.eps)


def _intensity_dim(mode, c):
    return {"stats": 2 * c, "moment": 5 * c, "both": 7 * c, "none": 0}[mode]


# 新组件 09 | AuxReconDecoder —— 辅助重建解码器 (自监督正则)
class AuxReconDecoder(nn.Module):
    """轻量重建解码器: 从适配特征上采样重建灰度输入。

    用作自监督重建正则 (生成式辅助任务): 要重建得好, 编码器必须学到完整的笔画结构,
    在限定数据集内 (不引入外部数据) 增强特征学习。每层 ConvTranspose 2x 上采。
    """
    def __init__(self, in_c, ups=3):
        super().__init__()
        layers = []
        c = in_c
        for _ in range(ups):
            out_c = max(c // 2, 16)
            layers += [nn.ConvTranspose2d(c, out_c, 4, 2, 1),
                       nn.BatchNorm2d(out_c), nn.ReLU()]
            c = out_c
        layers += [nn.Conv2d(c, 1, 3, 1, 1), nn.Sigmoid()]
        self.dec = nn.Sequential(*layers)

    def forward(self, x):
        return self.dec(x)


# 新模型 03 | TransferBackboneAdapter —— 三分支/多特征 Adapter (★特征优先核心)
class TransferBackboneAdapter(_TransferBase):
    """三分支/多特征 Adapter (工厂式) 接在预训练特征之上 (方案 3 + 特征优先)。

    预训练骨干出最后一层特征图 -> 1x1 适配到 adapt_dim -> 按 branches 列表并联多条特征分支
    (空间/通道/多尺度/Gabor频率/笔画结构) -> 融合。head 输入 =
        GeM(分支融合) ⊕ GAP(骨干) ⊕ 强度特征。
    强度特征由 intensity_mode 决定:
        stats  = std/max 统计池化
        moment = 强度加权几何矩 (MomentPool, 经典图像矩, 广义化 GAP)
        both   = 两者拼接
        none   = 不加
    weber_norm=True 时对强度路径先做 Weber 除性归一化 (对墨色深浅鲁棒)。
    分类头由 make_head 决定 (linear 或 cosine 度量学习头)。branches/intensity 均可消融。
    """
    def __init__(self, model_name="resnet50", num_classes=62, pretrained=True,
                 input_size=160, adapt_dim=64,
                 branches=("spatial", "channel", "multiscale"),
                 intensity_mode="stats", weber_norm=False, use_intensity=True,
                 head="linear", gray_to_3ch=True, imagenet_norm=True,
                 head_dropout=0.1, arcface_scale=30.0, code_length=127, recon=False):
        super().__init__(model_name, pretrained, input_size, gray_to_3ch, imagenet_norm)
        # 向后兼容: use_intensity=False 等价 intensity_mode="none"
        if not use_intensity:
            intensity_mode = "none"
        self.intensity_mode = intensity_mode
        c_last = self.backbone.feature_info.channels()[-1]
        self.adapt = ConvBlock(c_last, adapt_dim, 1, 1, 0)
        bc = adapt_dim // 4
        self.branches = nn.ModuleList([make_branch(n, adapt_dim, bc) for n in branches])
        self.branch_fuse = ConvBlock(bc * len(self.branches), adapt_dim, 1, 1, 0)
        self.gem = GeMPool()
        self.weber = WeberNorm() if weber_norm else None
        self.moment = MomentPool() if intensity_mode in ("moment", "both") else None
        feat_dim = adapt_dim + c_last + _intensity_dim(intensity_mode, adapt_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(head_dropout)
        self.head = make_head(head, feat_dim, num_classes, scale=arcface_scale,
                              code_length=code_length)
        self.decoder = AuxReconDecoder(adapt_dim) if recon else None

    def forward(self, x):
        feat = self.backbone(self._preprocess(x))[-1]
        a = self.adapt(feat)
        fused = self.branch_fuse(torch.cat([br(a) for br in self.branches], dim=1))
        parts = [self.gem(fused), functional.adaptive_avg_pool2d(feat, 1).flatten(1)]
        if self.intensity_mode != "none":
            src = self.weber(fused) if self.weber is not None else fused
            if self.intensity_mode in ("stats", "both"):
                parts.append(_intensity_stats(src))
            if self.intensity_mode in ("moment", "both"):
                parts.append(self.moment(src))
        pooled = self.norm(torch.cat(parts, dim=1))
        recon_out = self.decoder(a) if self.decoder is not None else None
        return self.head(self.drop(pooled)), pooled, recon_out
