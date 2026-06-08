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

DATA_DIR = 'D:/课程设计1—62类手写数字识别/English-Handwritten-Characters-Dataset'
RESAMPLE_FILTER = Image.Resampling.LANCZOS


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
    return train_d, val_d, test_d, all_labels, l2i


class Augment:
    """批量 tensor 增强: B 张图一次过"""
    @staticmethod
    def apply_batch(batch):
        """batch: [B, 1, H, W] -> [B, 1, H, W]"""
        batch_size, channels, height, width = batch.shape
        device = batch.device
        # 每张图独立的变换参数
        angles = torch.empty(batch_size, device=device).uniform_(-10, 10) * 3.14159 / 180
        cos_a, sin_a = angles.cos(), angles.sin()
        dx = torch.empty(batch_size, device=device).uniform_(-4, 4) / width * 2
        dy = torch.empty(batch_size, device=device).uniform_(-4, 4) / height * 2
        scale = torch.empty(batch_size, device=device).uniform_(0.9, 1.1)
        # 构建 [B, 2, 3] 仿射矩阵
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
        # 高斯噪声 (一半图片)
        mask = torch.rand(batch_size, device=device) > 0.5
        if mask.any():
            noise = torch.randn_like(batch[mask]) * 0.03
            batch[mask] = torch.clamp(batch[mask] + noise, 0.0, 1.0)
        # 随机矩形遮罩 (30%图片, 模拟笔画缺失)
        erase_mask = torch.rand(batch_size, device=device) < 0.3
        if erase_mask.any():
            indices = erase_mask.nonzero(as_tuple=True)[0]
            for idx in indices:
                h_size = int(height * random.uniform(0.1, 0.3))
                w_size = int(width * random.uniform(0.1, 0.3))
                y0 = random.randint(0, height - h_size)
                x0 = random.randint(0, width - w_size)
                batch[idx, :, y0:y0+h_size, x0:x0+w_size] = 0.0
        return batch


class CharDataset(Dataset):
    def __init__(self, data, size=(48, 64), train=True):
        self.size = size
        self.train = train
        self.labels = [label for _, label in data]
        # 全量预加载: 数据集仅 3410 张 64×48 灰度图 (~40MB), 一次读完零等待
        self.images = []
        for idx, (file_name, _) in enumerate(data):
            img = Image.open(f'{DATA_DIR}/Img/{file_name}').convert('L')
            img = img.resize(self.size, RESAMPLE_FILTER)
            arr = 1.0 - np.array(img, dtype=np.float32) / 255.0
            self.images.append(torch.from_numpy(arr).unsqueeze(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i].clone(), self.labels[i]


def collate_with_augment(batch):
    """collate_fn: 拼 batch 后批量增强, GPU 一次处理 64 张图"""
    images, labels = zip(*batch)
    images = torch.stack(images)  # [B, 1, H, W]
    labels = torch.tensor(labels)
    images = Augment.apply_batch(images)
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
