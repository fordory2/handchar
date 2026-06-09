# noinspection SpellCheckingInspection
"""模型定义: 组件 + 工厂 + 14模型 + SOTA基线 + HybridHandCharNet"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import timm
from torchvision.models import densenet121, resnet18


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


class HandCharNet(make_hand_char(SE)):
    pass


class HandCharNetECA(make_hand_char(ECA)):
    pass


class HandCharNetGroupSE(make_hand_char(GROUP_SE)):
    pass


class HandCharNetCBAM(make_hand_char(CBAM)):
    pass


class HandCharNetCA(make_hand_char(CA)):
    pass


class HandCharNetCARegularized(make_hand_char(CA, dropout_rate=0.4)):
    pass


class HandCharNetNoDirection(make_hand_char(SE, use_direction=False)):
    pass


class HandCharNetNoDirectionEca(make_hand_char(ECA, use_direction=False)):
    pass


class HandCharNetNoSe(make_hand_char(None)):
    pass


class HandCharNetNoMultiScale(make_hand_char(SE, use_multiscale=False)):
    pass


class HandCharNetResGELU(make_hand_char(SE, residual=True, activation_cls=nn.GELU)):
    pass


class HandCharNetNoDirECA(make_hand_char(ECA, use_direction=False)):
    """去 DirectionConv + ECA — 消融最强单点"""
    pass


class HandCharNetNoDirHybrid(nn.Module):
    """去Direction + ECA通道 + CBAM空间 + 加宽"""
    def __init__(self, num_classes=62, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3), nn.MaxPool2d(2))
        self.stage1 = nn.Sequential(ConvBlock(32, 64, 3), ECABlock(64), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ConvBlock(64, 96, 3), ConvBlock(96, 96, 3),
                                    CBAMBlock(96, 8), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(MultiScalePool(96, 160),
                                    ConvBlock(160, 128, 1, 0), ConvBlock(128, 128, 3),
                                    ECABlock(128))
        self.stage4 = nn.Sequential(ConvBlock(128, 192, 3), CBAMBlock(192, 8),
                                    ConvBlock(192, 192, 3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(192, 128), nn.ReLU(), nn.Dropout(dropout*0.5),
                                  nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))


# ====== SOTA 基线 ======
class ResNet18Char(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, x):
        return self.model(x)


class DenseNetChar(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = densenet121(num_classes=num_classes)
        self.model.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

    def forward(self, x):
        return self.model(x)


class ConvNeXtV2Char(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.model(x)


class MobileNetV4Char(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.model = timm.create_model('mobilenetv4_conv_small', pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.model(x)


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
class HybridHandCharNet(nn.Module):
    """大一统模型: 空间/频率/通道/多尺度 4 分支 → UNet 跳跃 → 可选 BiLSTM → 主+aux+contrastive 三头.

    输入: [B, 1, 64, 48] (H=64, W=48)
    输出: (main_logits[B,62], aux_logits[B,3], cont_feat[B,128], unified_feat)
    """
    def __init__(self, num_classes=62, aux_classes=3, dropout=0.146, use_rnn=True,
                 rnn_cell='lstm', rnn_hidden=128, rnn_layers=2, rnn_proj_dim=256):
        super().__init__()
        self.use_rnn = use_rnn

        # Stem: 1/4 下采样
        self.stem = nn.Sequential(ConvBlock(1, 32, 7, 2, 3), nn.MaxPool2d(2))

        # 4 个并联分支 (输入 [B,32,16,12], 每路输出 16 通道)
        branch_c = 16
        stem_c = 32
        self.spatial_branch = nn.Sequential(
            SpatialAttentionBlock(),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.frequency_branch = FrequencyBlock(stem_c, branch_c)
        self.channel_branch = nn.Sequential(
            ECABlock(stem_c),
            ConvBlock(stem_c, branch_c, 3, 1, 1),
        )
        self.multiscale_branch = MultiScalePool(stem_c, branch_c)
        # 4×16 → 64 通道融合
        self.branch_fuse = ConvBlock(branch_c * 4, 64, 1, 1, 0)
        # 输出 feat_fine: [B,64,16,12]

        # Encoder
        self.stage1 = nn.Sequential(ResBlock(64, 80), nn.MaxPool2d(2))   # [B,80,8,6]
        self.stage2 = nn.Sequential(ResBlock(80, 128), nn.MaxPool2d(2))  # [B,128,4,3]
        self.stage3 = ResBlock(128, 160)                                  # [B,160,4,3]

        # UNet 风格 decoder
        self.up3 = nn.ConvTranspose2d(160, 128, 2, 2)               # → [B,128,8,6]
        self.fuse3 = ConvBlock(128 + 80, 128, 1, 1, 0)              # concat feat_mid (80ch)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)                # → [B,64,16,12]
        self.fuse2 = ConvBlock(64 + 64, 64, 1, 1, 0)                # concat feat_fine (64ch)

        # 可选序列建模: decoder_out [B,64,16,12] → W=12 步, 每步 64*16=1024 维
        # rnn_cell: 'lstm' | 'gru' | 'transformer'
        self.rnn_cell = rnn_cell.lower() if use_rnn else None
        if use_rnn:
            self.lstm_proj = nn.Linear(64 * 16, rnn_proj_dim)
            if self.rnn_cell == 'transformer':
                # 可学习位置编码 (W=12 固定)
                self.pos_encoding = nn.Parameter(torch.zeros(12, 1, rnn_proj_dim))
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

        self.head_main = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(unified_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self.head_aux = nn.Linear(unified_dim, aux_classes)
        self.head_contrastive = nn.Linear(unified_dim, 128)

    def encode(self, x):
        """编码到 (feat_coarse, decoder_out), 供 MAE / Proto 复用."""
        x = self.stem(x)
        branches = [
            self.spatial_branch(x),
            self.frequency_branch(x),
            self.channel_branch(x),
            self.multiscale_branch(x),
        ]
        feat_fine = self.branch_fuse(torch.cat(branches, dim=1))
        feat_mid = self.stage1(feat_fine)
        feat_coarse_pre = self.stage2(feat_mid)
        feat_coarse = self.stage3(feat_coarse_pre)
        up = self.up3(feat_coarse)
        up = self.fuse3(torch.cat([up, feat_mid], dim=1))
        up = self.up2(up)
        decoder_out = self.fuse2(torch.cat([up, feat_fine], dim=1))
        return feat_coarse, decoder_out

    def unified_feature(self, x):
        """供 Proto 元学习用: 输出 unified flat 特征向量 [B, unified_dim]."""
        feat_coarse, decoder_out = self.encode(x)
        coarse_pooled = self.gap_coarse(feat_coarse).flatten(1)
        fine_pooled = self.gap_fine(decoder_out).flatten(1)
        if self.use_rnn:
            b, c, h, w = decoder_out.shape
            seq = decoder_out.permute(3, 0, 1, 2).reshape(w, b, c * h)
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
        aux_logits = self.head_aux(unified)
        cont_feat = functional.normalize(self.head_contrastive(unified), dim=1)
        return main_logits, aux_logits, cont_feat, unified
