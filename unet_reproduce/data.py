"""数据加载: 45/5/5 划分 + 强增强 + FCMAE mask collate."""
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _resolve_data_dir():
    """优先级: 环境变量 > AutoDL 标准路径 > 项目根目录."""
    env = os.environ.get("HANDCHAR_DATA_DIR")
    if env and os.path.exists(os.path.join(env, "english.csv")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "/root/autodl-tmp/English-Handwritten-Characters-Dataset",
        os.path.normpath(os.path.join(here, "..", "English-Handwritten-Characters-Dataset")),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "english.csv")):
            return path
    return candidates[-1]


DATA_DIR = _resolve_data_dir()
IMG_SIZE = (48, 64)  # (W, H) PIL convention
NUM_CLASSES = 62

# 混淆字符辅助头: 治 worst-10 中的 0/O/o, 1/I/l, 5/S, C/c. 11-way = 10 + "other".
CONFUSABLE_UNION = ("0", "O", "o", "1", "I", "l", "5", "S", "C", "c")
PAIR_NUM_CLASSES = len(CONFUSABLE_UNION) + 1  # 11
PAIR_OTHER_IDX = len(CONFUSABLE_UNION)        # 10
_PAIR_IDX_MAP = {}  # 全 62 类索引 -> 11-way 索引, 在 _register_sensitive 时建立

SENSITIVE_CHARS = {
    'rotate':    'bdpq69MWNZun',
    'erode':     'il1j.',
    'dilate':    'Oo0',
    'close':     'cu',
    'open':      'eB',
    'erase':     'il1.',
}
_SENSITIVE_IDX = {k: set() for k in SENSITIVE_CHARS}
_IMAGE_CACHE = {}


def _register_sensitive(l2i):
    for group, chars in SENSITIVE_CHARS.items():
        _SENSITIVE_IDX[group] = {l2i[c] for c in chars if c in l2i}
    _PAIR_IDX_MAP.clear()
    for cls_idx in range(NUM_CLASSES):
        _PAIR_IDX_MAP[cls_idx] = PAIR_OTHER_IDX
    for i, ch in enumerate(CONFUSABLE_UNION):
        if ch in l2i:
            _PAIR_IDX_MAP[l2i[ch]] = i


def labels_to_pair(labels):
    """[B] 62-class -> [B] 11-class (在 CONFUSABLE_UNION 外的全归 'other')."""
    return torch.tensor([_PAIR_IDX_MAP[int(l)] for l in labels],
                        dtype=torch.long, device=labels.device)


def _load_image_tensor(file_name):
    cached = _IMAGE_CACHE.get(file_name)
    if cached is not None:
        return cached
    img = Image.open(f'{DATA_DIR}/Img/{file_name}').convert('L')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = 1.0 - np.array(img, dtype=np.float32) / 255.0  # 反色: 笔画=1, 背景=0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    _IMAGE_CACHE[file_name] = tensor
    return tensor


def load_split_data():
    """按编号划分: 001-045 train, 046-050 val, 051-055 test."""
    with open(f'{DATA_DIR}/english.csv') as f:
        label_map = {r['image']: r['label'] for r in csv.DictReader(f)}
    all_labels = sorted(set(label_map.values()))
    l2i = {l: i for i, l in enumerate(all_labels)}
    train_d, val_d, test_d = [], [], []
    for img_path, label in label_map.items():
        file_name = os.path.basename(img_path)
        num = int(file_name.split('-')[1].split('.')[0])
        idx = l2i[label]
        if num <= 45:
            train_d.append((file_name, idx))
        elif num <= 50:
            val_d.append((file_name, idx))
        else:
            test_d.append((file_name, idx))
    _register_sensitive(l2i)
    return train_d, val_d, test_d, all_labels, l2i


def load_all_data():
    """自监督预训练用: 全 3410 张, 不分割."""
    with open(f'{DATA_DIR}/english.csv') as f:
        label_map = {r['image']: r['label'] for r in csv.DictReader(f)}
    all_labels = sorted(set(label_map.values()))
    l2i = {l: i for i, l in enumerate(all_labels)}
    data = []
    for img_path, label in label_map.items():
        data.append((os.path.basename(img_path), l2i[label]))
    _register_sensitive(l2i)
    return data


def _sensitive_mask(group, labels, device):
    idx_set = _SENSITIVE_IDX[group]
    if not idx_set:
        return torch.zeros(len(labels), dtype=torch.bool, device=device)
    return torch.tensor(
        [int(l.item()) in idx_set for l in labels],
        dtype=torch.bool, device=device,
    )


def _elastic_deform(batch, alpha=8.0, sigma=4.0):
    """弹性形变: 高斯模糊的随机位移场 + grid_sample. 手写最自然扰动."""
    batch_size, _, height, width = batch.shape
    device = batch.device
    dx = torch.randn(batch_size, 1, height, width, device=device)
    dy = torch.randn(batch_size, 1, height, width, device=device)
    # 高斯模糊: 用 separable conv
    radius = int(sigma * 3)
    kernel_size = 2 * radius + 1
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - radius
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kx = g.view(1, 1, 1, kernel_size)
    ky = g.view(1, 1, kernel_size, 1)
    dx = functional.conv2d(dx, kx, padding=(0, radius))
    dx = functional.conv2d(dx, ky, padding=(radius, 0)) * alpha
    dy = functional.conv2d(dy, kx, padding=(0, radius))
    dy = functional.conv2d(dy, ky, padding=(radius, 0)) * alpha
    # 构造 base grid
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij',
    )
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    delta = torch.stack([dx.squeeze(1) / width * 2, dy.squeeze(1) / height * 2], dim=-1)
    grid = base + delta
    return functional.grid_sample(batch, grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)


class StrongAugment:
    """监督微调用强增强: 仿射 + 弹性 + 噪声 + 形态学 + 擦除."""
    @staticmethod
    def apply(batch, labels):
        batch_size, _, height, width = batch.shape
        device = batch.device

        # 仿射 ±7° 旋转 + ±8px 平移 + 0.85-1.15 缩放
        angles = torch.empty(batch_size, device=device).uniform_(-7, 7) * 3.14159 / 180
        angles = angles.masked_fill(_sensitive_mask('rotate', labels, device), 0.0)
        cos_a, sin_a = angles.cos(), angles.sin()
        dx = torch.empty(batch_size, device=device).uniform_(-8, 8) / width * 2
        dy = torch.empty(batch_size, device=device).uniform_(-8, 8) / height * 2
        scale = torch.empty(batch_size, device=device).uniform_(0.85, 1.15)
        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0] = scale * cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = dx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = scale * cos_a
        theta[:, 1, 2] = dy
        grid = functional.affine_grid(theta, batch.shape, align_corners=False)
        batch = functional.grid_sample(batch, grid, mode='bilinear',
                                        padding_mode='border', align_corners=False)

        # 弹性形变 30%
        if torch.rand(1).item() < 0.3:
            batch = _elastic_deform(batch, alpha=8.0, sigma=4.0)

        # 高斯噪声 50%
        noise_m = torch.rand(batch_size, device=device) > 0.5
        if noise_m.any():
            noise = torch.randn_like(batch[noise_m]) * 0.05
            batch[noise_m] = torch.clamp(batch[noise_m] + noise, 0.0, 1.0)

        # 形态学 30%
        morph_m = torch.rand(batch_size, device=device) < 0.3
        if morph_m.any():
            op = random.choices(['dilate', 'erode', 'open', 'close'],
                                weights=[0.4, 0.4, 0.1, 0.1])[0]
            applicable = morph_m & ~_sensitive_mask(op, labels, device)
            if applicable.any():
                sub = batch[applicable]
                if op == 'dilate':
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                elif op == 'erode':
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                elif op == 'close':
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                else:
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                batch[applicable] = sub

        # RandomErasing 5×5, 20%
        erase_m = torch.rand(batch_size, device=device) < 0.2
        erase_m = erase_m & ~_sensitive_mask('erase', labels, device)
        if erase_m.any():
            ys = torch.randint(0, height - 5, (batch_size,), device=device)
            xs = torch.randint(0, width - 5, (batch_size,), device=device)
            for i in torch.where(erase_m)[0]:
                y, x = int(ys[i]), int(xs[i])
                batch[i, :, y:y + 5, x:x + 5] = 0.0
        return batch


class GeoAugment:
    """预训练用轻几何增强: 仅仿射 + 弹性."""
    @staticmethod
    def apply(batch):
        batch_size, _, height, width = batch.shape
        device = batch.device
        angles = torch.empty(batch_size, device=device).uniform_(-5, 5) * 3.14159 / 180
        cos_a, sin_a = angles.cos(), angles.sin()
        dx = torch.empty(batch_size, device=device).uniform_(-4, 4) / width * 2
        dy = torch.empty(batch_size, device=device).uniform_(-4, 4) / height * 2
        scale = torch.empty(batch_size, device=device).uniform_(0.9, 1.1)
        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0] = scale * cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = dx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = scale * cos_a
        theta[:, 1, 2] = dy
        grid = functional.affine_grid(theta, batch.shape, align_corners=False)
        batch = functional.grid_sample(batch, grid, mode='bilinear',
                                        padding_mode='border', align_corners=False)
        if torch.rand(1).item() < 0.5:
            batch = _elastic_deform(batch, alpha=6.0, sigma=4.0)
        return batch


def make_mae_mask(batch_size, height, width, patch=8, ratio=0.6, device='cuda'):
    """FCMAE: 返回 [B, 1, H, W] mask, 1=可见, 0=mask."""
    ph, pw = height // patch, width // patch
    n_patch = ph * pw
    n_keep = int(n_patch * (1 - ratio))
    masks = []
    for _ in range(batch_size):
        idx = torch.randperm(n_patch, device=device)
        flat = torch.zeros(n_patch, device=device)
        flat[idx[:n_keep]] = 1.0
        m = flat.view(1, ph, pw)
        m = m.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)
        masks.append(m)
    return torch.stack(masks)  # [B, 1, H, W]


class CharDataset(Dataset):
    def __init__(self, data):
        self.labels = [lbl for _, lbl in data]
        self.images = [_load_image_tensor(fn) for fn, _ in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i].clone(), self.labels[i]


def collate_strong_aug(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    images = StrongAugment.apply(images, labels)
    return images, labels


def collate_no_aug(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def collate_pretrain(batch):
    """预训练: geo 增强后图既是输入也是重建 target."""
    images, _ = zip(*batch)
    images = torch.stack(images)
    images = GeoAugment.apply(images)
    return images, images.clone()


def make_loader(data, batch=64, train=False, pretrain=False):
    ds = CharDataset(data)
    if pretrain:
        return DataLoader(ds, batch, shuffle=True, collate_fn=collate_pretrain,
                          num_workers=0, drop_last=True)
    if train:
        return DataLoader(ds, batch, shuffle=True, collate_fn=collate_strong_aug,
                          num_workers=0, drop_last=False)
    return DataLoader(ds, batch, shuffle=False, collate_fn=collate_no_aug,
                      num_workers=0, drop_last=False)


def load_random_split_data(ratio=(0.7, 0.2, 0.1), seed=42):
    """随机 7:2:1 划分 (每类 55 张 -> 38/11/6).

    论文若按 7:2:1 跑, val/test 同分布, 没有"51-55 笔迹漂移"问题.
    """
    with open(f'{DATA_DIR}/english.csv') as f:
        label_map = {r['image']: r['label'] for r in csv.DictReader(f)}
    all_labels = sorted(set(label_map.values()))
    l2i = {l: i for i, l in enumerate(all_labels)}
    by_class = {i: [] for i in range(NUM_CLASSES)}
    for img_path, label in label_map.items():
        by_class[l2i[label]].append(os.path.basename(img_path))
    rng = random.Random(seed)
    train_d, val_d, test_d = [], [], []
    for idx, files in by_class.items():
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_tr = int(round(n * ratio[0]))
        n_val = int(round(n * ratio[1]))
        # 余下归 test
        train_d.extend((f, idx) for f in files[:n_tr])
        val_d.extend((f, idx) for f in files[n_tr:n_tr + n_val])
        test_d.extend((f, idx) for f in files[n_tr + n_val:])
    _register_sensitive(l2i)
    return train_d, val_d, test_d, all_labels, l2i


def load_combined_train_data():
    """train+val 合并: 001-050 全部用于训练, 仅留 051-055 为 test."""
    train_d, val_d, test_d, all_labels, l2i = load_split_data()
    return train_d + val_d, test_d, all_labels, l2i
