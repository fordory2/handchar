"""数据加载 + 预处理"""
import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from project_constants import NUM_WORKERS, PIN_MEMORY

def _resolve_data_dir():
    """优先级: 环境变量 > AutoDL 标准路径 > 项目本地路径."""
    env = os.environ.get("HANDCHAR_DATA_DIR")
    if env and os.path.exists(os.path.join(env, "english.csv")):
        return env
    candidates = [
        "/root/autodl-tmp/English-Handwritten-Characters-Dataset",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "English-Handwritten-Characters-Dataset"),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "english.csv")):
            return path
    return candidates[-1]  # 兜底: 报错时路径信息最有用


DATA_DIR = _resolve_data_dir()
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
        if num <= 45:
            train_d.append((file_name, idx))  # 001-045 train
        elif num <= 50:
            val_d.append((file_name, idx))    # 046-050 val (修 val 缺失导致拿 test 当 val 的泄漏)
        else:
            test_d.append((file_name, idx))   # 051-055 holdout
    _register_sensitive(l2i)
    return train_d, val_d, test_d, all_labels, l2i


def load_all_data():
    """加载全部 3410 张 (62 类 × 55 张), 不分 train/val/test. 用于 K-Fold CV."""
    with open(f'{DATA_DIR}/english.csv') as f:
        label_map = {r['image']: r['label'] for r in csv.DictReader(f)}
    all_labels = sorted(set(label_map.values()))
    l2i = {l: i for i, l in enumerate(all_labels)}
    all_d = []
    for img_path, label in label_map.items():
        file_name = os.path.basename(img_path)
        all_d.append((file_name, l2i[label]))
    _register_sensitive(l2i)
    return all_d, all_labels, l2i


def load_holdout_data(writer_min=51):
    """加载干净留出集 (默认写手 051-055). 与 load_kfold_splits(exclude_writer_min=51) 互补:
    fold 模型只在写手 <51 上训练, 这里返回 >=writer_min 的样本, 二者无交集.

    返回 (holdout_list, all_labels, l2i), holdout_list: [(file_name, label_idx), ...]
    """
    all_d, all_labels, l2i = load_all_data()
    holdout = [(fn, lbl) for (fn, lbl) in all_d
               if int(fn.split('-')[1].split('.')[0]) >= writer_min]
    return holdout, all_labels, l2i


def load_kfold_splits(n_splits=5, random_state=42, exclude_writer_min=None):
    """StratifiedKFold 切分 3410 张全集, 每类 ~44 train / ~11 val.

    自实现 (避免 sklearn / numpy 版本冲突): 对每类 55 张索引 RNG 打乱, 均分 n_splits 段,
    fold k 取第 k 段为 val, 其余为 train. random_state=42 固定 → 5 fold 结果可复现.

    返回 (folds, all_labels, l2i):
      folds: List[(train_list, val_list)] 长度 n_splits
      train_list / val_list: [(file_name, label_idx), ...]
    """
    all_d, all_labels, l2i = load_all_data()
    # 干净留出: exclude_writer_min=51 时, 把写手编号 >=51 (即 051-055) 整块剔除出 CV 池,
    # 使所有 fold 模型都没见过留出集 -> bagging 集成可在 051-055 上做无泄漏评估.
    if exclude_writer_min is not None:
        all_d = [(fn, lbl) for (fn, lbl) in all_d
                 if int(fn.split('-')[1].split('.')[0]) < exclude_writer_min]
    # 按类聚集索引
    cls_to_idx = {}
    for i, (_, lbl) in enumerate(all_d):
        cls_to_idx.setdefault(lbl, []).append(i)

    rng = np.random.RandomState(random_state)
    # 每类内打乱, 然后按 numpy.array_split 均分 n_splits 段
    cls_fold_idx = {}  # cls -> [val_idx_of_fold_0, val_idx_of_fold_1, ...]
    for cls, idxs in cls_to_idx.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        cls_fold_idx[cls] = np.array_split(shuffled, n_splits)

    folds = []
    for k in range(n_splits):
        val_idx = []
        train_idx = []
        for cls, segments in cls_fold_idx.items():
            for j, seg in enumerate(segments):
                if j == k:
                    val_idx.extend(seg.tolist())
                else:
                    train_idx.extend(seg.tolist())
        tr = [all_d[i] for i in train_idx]
        val = [all_d[i] for i in val_idx]
        folds.append((tr, val))
    return folds, all_labels, l2i


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

        # 仿射: 角度 ±5° (旋转敏感类清零角度, 位移/缩放保留)
        angles = torch.empty(batch_size, device=device).uniform_(-5, 5) * 3.14159 / 180
        angles = angles.masked_fill(sensitive_mask('rotate'), 0.0)
        cos_a, sin_a = angles.cos(), angles.sin()
        dx = torch.empty(batch_size, device=device).uniform_(-6, 6) / width * 2
        dy = torch.empty(batch_size, device=device).uniform_(-6, 6) / height * 2
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

        # 形态学 (30%): dilate/erode 拟真笔粗抖动为主, open/close 罕见笔触缺陷
        morph_mask = torch.rand(batch_size, device=device) < 0.3
        if morph_mask.any():
            op = random.choices(
                ['dilate', 'erode', 'open', 'close'],
                weights=[0.4, 0.4, 0.1, 0.1],
            )[0]
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

        # Random Erasing (p=0.5): 矩形遮罩面积 2%-15%, 填 0 (纸色). 强迫不依赖具体像素块.
        # 跳过 erase_big 敏感类 (il1.) — 单笔画/小点, 大遮罩易直接消失
        erase_prob = float(os.environ.get("HANDCHAR_ERASE_PROB", 0.5))
        if erase_prob <= 0:
            return batch
        erase_mask = torch.rand(batch_size, device=device) < erase_prob
        erase_applicable = erase_mask & ~sensitive_mask('erase_big')
        if erase_applicable.any():
            img_area = height * width
            for i in torch.nonzero(erase_applicable, as_tuple=False).flatten().tolist():
                # 随机面积 2-15%, 长宽比 0.3-3.3 (torchvision RandomErasing 默认范围)
                area = img_area * random.uniform(0.02, 0.15)
                ratio = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
                eh = int(round(math.sqrt(area * ratio)))
                ew = int(round(math.sqrt(area / ratio)))
                if eh < 1 or ew < 1 or eh >= height or ew >= width:
                    continue
                y0 = random.randint(0, height - eh)
                x0 = random.randint(0, width - ew)
                batch[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0
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


MIXUP_ALPHA = 0.2
MIXUP_PROB = float(os.environ.get("HANDCHAR_MIXUP_PROB", 0.5))


def collate_with_augment(batch):
    """collate_fn: 拼 batch → 批量增强 → 50% 概率走 MixUp.

    输出 4 元组 (images, y_a, y_b, lam):
      - 未触发 MixUp: y_b == y_a, lam = 1.0 → 训练循环统一用 lam*focal(y_a) + (1-lam)*focal(y_b)
      - 触发: lam ~ Beta(α, α), images = lam*x + (1-lam)*x[perm], y_b = labels[perm]
    """
    images, labels = zip(*batch)
    images = torch.stack(images)  # [B, 1, H, W]
    labels = torch.tensor(labels)
    images = Augment.apply_batch(images, labels)

    if random.random() < MIXUP_PROB:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        # Beta(0.2,0.2) 极端化, 强制 lam 偏向 0/1 — 多数 batch 主导一边, 少数真混合
        perm = torch.randperm(images.size(0))
        images = lam * images + (1.0 - lam) * images[perm]
        y_b = labels[perm]
    else:
        lam = 1.0
        y_b = labels
    return images, labels, y_b, lam


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
