"""ConvNeXt V2 U-Net (Woo et al. CVPR 2023, FCMAE).

输入 1×48×64 -> 编码器 4 级 (24×32 -> 12×16 -> 6×8 -> 3×4) -> 解码器对称.
分类头从 bottleneck (GAP), 重建头从解码器顶层.
GRN 关键: 防止 vanilla ConvNeXt + MAE 特征坍塌.
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional


class LayerNorm2d(nn.Module):
    """Channels-first LN."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2 公式 3)."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x):
        # x: [B, H, W, C]  (channels-last)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # [B, 1, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask.floor_()
        return x.div(keep) * mask


class ConvNeXtV2Block(nn.Module):
    """DwConv 7×7 -> LN -> Linear 4× -> GELU -> GRN -> Linear -> DropPath + skip."""
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * channels)
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        skip = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return skip + self.drop_path(x)


def _make_stage(channels, n_blocks, drop_path):
    return nn.Sequential(*[ConvNeXtV2Block(channels, drop_path) for _ in range(n_blocks)])


class ConvNeXtV2UNet(nn.Module):
    """1.2M 参数版: 通道 24/48/96/192, blocks 2/2/2/2.

    Forward:
        forward(x)                          -> logits   [B, 62]
        forward(x, mask=m, return_recon=True) -> recon  [B, 1, H, W]
    """
    def __init__(self, num_classes=62, drop_path=0.1, dropout=0.4,
                 num_pair_classes=0):
        """num_pair_classes>0 时启用混淆字符辅助头 (11-way)."""
        super().__init__()
        self.num_pair_classes = num_pair_classes
        # Stem: stride=2, 48×64 -> 24×32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(24),
        )
        self.stage1 = _make_stage(24, 2, drop_path)

        self.down1 = nn.Sequential(LayerNorm2d(24), nn.Conv2d(24, 48, 2, 2))
        self.stage2 = _make_stage(48, 2, drop_path)

        self.down2 = nn.Sequential(LayerNorm2d(48), nn.Conv2d(48, 96, 2, 2))
        self.stage3 = _make_stage(96, 2, drop_path)

        self.down3 = nn.Sequential(LayerNorm2d(96), nn.Conv2d(96, 192, 2, 2))
        self.bottleneck = _make_stage(192, 2, drop_path)

        # Decoder
        self.up3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.dec3_proj = nn.Conv2d(192, 96, 1)  # concat skip3(96) + up(96) = 192
        self.dec3 = _make_stage(96, 2, drop_path)

        self.up2 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.dec2_proj = nn.Conv2d(96, 48, 1)
        self.dec2 = _make_stage(48, 2, drop_path)

        self.up1 = nn.ConvTranspose2d(48, 24, 2, 2)
        self.dec1_proj = nn.Conv2d(48, 24, 1)
        self.dec1 = _make_stage(24, 2, drop_path)

        # Final upsample 24×32 -> 48×64
        self.final_up = nn.ConvTranspose2d(24, 12, 2, 2)
        self.recon_head = nn.Conv2d(12, 1, 1)

        # 分类头 (from bottleneck)
        self.cls_norm = nn.LayerNorm(192)
        self.cls_drop = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(192, num_classes)

        # 混淆字符辅助头 (可选): 直击 0/O/o, 1/I/l, 5/S, C/c 这种 worst-class.
        if num_pair_classes > 0:
            self.pair_fc = nn.Linear(192, num_pair_classes)

        self._init_weights()

    def _init_weights(self):
        # Conv 用 Kaiming (适合 ReLU/GELU 激活, 小数据上更易破壳);
        # Linear/分类头保留 trunc_normal_(0.02) 的 transformer 惯例.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x, mask=None):
        """编码器 forward, 返回 (s1, s2, s3, bot). mask 用于 FCMAE 稀疏化."""
        if mask is not None:
            x = x * mask  # 输入 mask 置零
        x = self.stem(x)
        if mask is not None:
            x = x * functional.interpolate(mask, size=x.shape[-2:], mode='nearest')
        s1 = self.stage1(x)
        if mask is not None:
            s1 = s1 * functional.interpolate(mask, size=s1.shape[-2:], mode='nearest')

        x = self.down1(s1)
        s2 = self.stage2(x)
        if mask is not None:
            s2 = s2 * functional.interpolate(mask, size=s2.shape[-2:], mode='nearest')

        x = self.down2(s2)
        s3 = self.stage3(x)
        if mask is not None:
            s3 = s3 * functional.interpolate(mask, size=s3.shape[-2:], mode='nearest')

        x = self.down3(s3)
        bot = self.bottleneck(x)
        if mask is not None:
            bot = bot * functional.interpolate(mask, size=bot.shape[-2:], mode='nearest')
        return s1, s2, s3, bot

    def decode(self, s1, s2, s3, bot):
        x = self.up3(bot)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3_proj(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2_proj(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1_proj(x)
        x = self.dec1(x)

        x = self.final_up(x)
        return self.recon_head(x)

    def classify(self, bot):
        x = bot.mean(dim=[2, 3])  # GAP
        x = self.cls_norm(x)
        x = self.cls_drop(x)
        logits = self.cls_fc(x)
        if self.num_pair_classes > 0:
            return logits, self.pair_fc(x)
        return logits

    def forward(self, x, mask=None, return_recon=False):
        s1, s2, s3, bot = self.encode(x, mask=mask)
        if return_recon:
            return self.decode(s1, s2, s3, bot)
        return self.classify(bot)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = ConvNeXtV2UNet()
    n = count_params(net)
    print("Params: %.2fM" % (n / 1e6))
    x = torch.randn(2, 1, 48, 64)
    logits = net(x)
    print("Logits:", logits.shape)
    mask = torch.ones(2, 1, 48, 64)
    mask[:, :, :24, :32] = 0
    recon = net(x, mask=mask, return_recon=True)
    print("Recon:", recon.shape)
