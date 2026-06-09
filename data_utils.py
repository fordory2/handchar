"""数据加载 + 预处理"""
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from project_constants import NUM_WORKERS, PIN_MEMORY

DATA_DIR = os.environ.get(
    "HANDCHAR_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "English-Handwritten-Characters-Dataset"),
)
RESAMPLE_FILTER = Image.Resampling.LANCZOS

# 模块级 tensor 缓存: file_name -> [1, H, W] float tensor.
# 课设全集只有 3410 张 64×48 灰度图 (~40MB), 一次解码后永久驻留, 后续 fold/模型
# 切换时 CharDataset() 构造直接命中, 不再重跑 LANCZOS resize.
_IMAGE_CACHE = {}

# 敏感字符集合: 对特定增强易"变成另一类"的字符, 跳过对应增强
SENSITIVE_CHARS = {
    'rotate':    'bdpq69MWNZun',  # 旋转易混
    'erode':     'il1j.',         # 细笔画, 腐蚀后近空白
    'dilate':    'Oo0',           # 内孔被填实
    'close':     'cu',            # 开口被封 -> o
    'open':      'eB',            # 闭合环被打开
    'erase_big': 'il1.',          # 单笔画, 大遮罩直接消失
}
_SENSITIVE_IDX = {k: set() for k in SENSITIVE_CHARS}


def _register_sensitive(l2i):
    for group, chars in SENSITIVE_CHARS.items():
        _SENSITIVE_IDX[group] = {l2i[c] for c in chars if c in l2i}


def _load_image_tensor(file_name, size):
    key = (file_name, size)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached
    img = Image.open(f'{DATA_DIR}/Img/{file_name}').convert('L')
    img = img.resize(size, RESAMPLE_FILTER)
    arr = 1.0 - np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    _IMAGE_CACHE[key] = tensor
    return tensor


def load_split_data():
    """按课程设计划分: 001-045 train, 046-050 val, 051-055 test"""
    with open(f'{DATA_DIR}/english.csv') as f:
        label_map = {r['image']: r['label'] for r in csv.DictReader(f)}
    all_labels = sorted(set(label_map.values()))
    l2i = {l: i for i, l in enumerate(all_labels)}
    train_d, val_d, test_d = [], [], []
    for img_path, label in label_map.items():
        file_name = os.path.basename(img_path)
        num = int(file_name.split('-')[1].split('.')[0])
        idx = l2i[label]
        if num <= 50:
            train_d.append((file_name, idx))  # 001-050 CV pool
        else:
            test_d.append((file_name, idx))   # 051-055 holdout
    _register_sensitive(l2i)
    return train_d, val_d, test_d, all_labels, l2i


class Augment:
    """批量 tensor 增强: B 张图一次过. 按类掩码避免敏感类被增强变成另一类."""
    @staticmethod
    def apply_batch(batch, labels):
        """batch: [B, 1, H, W], labels: [B] long -> [B, 1, H, W]"""
        batch_size, channels, height, width = batch.shape
        device = batch.device

        def sensitive_mask(group):
            idx_set = _SENSITIVE_IDX[group]
            if not idx_set:
                return torch.zeros(batch_size, dtype=torch.bool, device=device)
            return torch.tensor(
                [int(l.item()) in idx_set for l in labels],
                dtype=torch.bool, device=device,
            )

        # 仿射: 角度 ±7° (旋转敏感类清零角度, 位移/缩放保留)
        angles = torch.empty(batch_size, device=device).uniform_(-7, 7) * 3.14159 / 180
        angles = angles.masked_fill(sensitive_mask('rotate'), 0.0)
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

        # 高斯噪声 (50%, 全类别)
        noise_mask = torch.rand(batch_size, device=device) > 0.5
        if noise_mask.any():
            noise = torch.randn_like(batch[noise_mask]) * 0.03
            batch[noise_mask] = torch.clamp(batch[noise_mask] + noise, 0.0, 1.0)

        # 矩形擦除 (30%): 敏感单笔画类上限 10%, 其它上限 20%
        erase_mask = torch.rand(batch_size, device=device) < 0.3
        if erase_mask.any():
            big_sens = sensitive_mask('erase_big')
            indices = erase_mask.nonzero(as_tuple=True)[0]
            for idx in indices:
                cap = 0.10 if bool(big_sens[idx]) else 0.20
                h_size = int(height * random.uniform(0.05, cap))
                w_size = int(width * random.uniform(0.05, cap))
                y0 = random.randint(0, height - h_size)
                x0 = random.randint(0, width - w_size)
                batch[idx, :, y0:y0+h_size, x0:x0+w_size] = 0.0

        # 形态学 (30%): dilate/erode/open/close, kernel=3 (半径1), 敏感类按 op 跳过
        morph_mask = torch.rand(batch_size, device=device) < 0.3
        if morph_mask.any():
            op = random.choice(['dilate', 'erode', 'open', 'close'])
            applicable = morph_mask & ~sensitive_mask(op)
            if applicable.any():
                sub = batch[applicable]
                if op == 'dilate':
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                elif op == 'erode':
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                elif op == 'close':
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                else:  # open
                    sub = -functional.max_pool2d(-sub, 3, 1, 1)
                    sub = functional.max_pool2d(sub, 3, 1, 1)
                batch[applicable] = sub
        return batch


class CharDataset(Dataset):
    def __init__(self, data, size=(48, 64), train=True):
        self.size = size
        self.train = train
        self.labels = [label for _, label in data]
        # 全量预加载: 命中模块级 _IMAGE_CACHE 时零开销, 否则解码一次后留存
        self.images = [_load_image_tensor(file_name, self.size)
                       for file_name, _ in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i].clone(), self.labels[i]


def collate_with_augment(batch):
    """collate_fn: 拼 batch 后批量增强, GPU 一次处理 64 张图"""
    images, labels = zip(*batch)
    images = torch.stack(images)  # [B, 1, H, W]
    labels = torch.tensor(labels)
    images = Augment.apply_batch(images, labels)
    return images, labels


def make_loaders(train_d, val_d=None, test_d=None, size=(48, 64), batch=32):
    use_workers = NUM_WORKERS > 0
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                         persistent_workers=use_workers)
    loaders = {}
    ds_train = CharDataset(train_d, size, train=True)
    loaders['train'] = DataLoader(ds_train, batch, shuffle=True,
                                  collate_fn=collate_with_augment, **loader_kwargs)
    if val_d:
        ds_val = CharDataset(val_d, size, train=False)
        loaders['val'] = DataLoader(ds_val, batch, shuffle=False, **loader_kwargs)
    if test_d:
        ds_test = CharDataset(test_d, size, train=False)
        loaders['test'] = DataLoader(ds_test, batch, shuffle=False, **loader_kwargs)
    return loaders
