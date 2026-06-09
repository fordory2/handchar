"""EMNIST byclass 无标签预训练数据池.

设计目标:
1. 加载 EMNIST byclass (~70 万张手写英文字母+数字, 28×28)
2. 修正 EMNIST 已知方向错位 (transpose + flip)
3. resize 到课设输入尺寸 (64×48), 同款归一化 (ink=1, paper=0)
4. 感知哈希去重: 剔除可能与课设 test (051-055) 近似的样本
5. 与课设 train (001-050) 拼成无标签预训练池

输出 Dataset 与 CharDataset 等同接口: `[1, 64, 48]` float tensor, 假 label=0.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import DATA_DIR, RESAMPLE_FILTER, load_split_data

# torchvision 默认存放路径; 4090 上建议 export HANDCHAR_TORCH_DATA=/root/autodl-tmp/torch_data
TORCH_DATA_ROOT = os.environ.get("HANDCHAR_TORCH_DATA",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_data"))


def fix_emnist_orientation(pil_img):
    """EMNIST 原图是转置+镜像的, 还原成正常方向."""
    return pil_img.transpose(Image.TRANSPOSE)


def emnist_to_tensor(pil_img, size=(48, 64)):
    """EMNIST PIL (28×28, 黑底白字) → [1, H, W] float tensor (与课设同款归一化).

    课设: `1.0 - arr/255` (白底黑字 → ink=1)
    EMNIST: `arr/255` (黑底白字, ink 本就是高值 → 不需要 invert)
    最终两者都是 "ink=1, paper=0".
    """
    img = fix_emnist_orientation(pil_img)
    img = img.resize(size, RESAMPLE_FILTER)  # PIL resize 接 (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0  # 已是 ink=1
    return torch.from_numpy(arr).unsqueeze(0)


def phash_simple(arr, hash_size=8):
    """简单感知哈希: DCT 低频区域阈值化.

    arr: numpy float32 [H, W], 任意尺寸. 返回 hash_size**2 维 bool 数组.
    """
    from scipy.fftpack import dct
    img_size = hash_size * 4
    # 等比 resize: PIL 已被外部处理过, 这里直接 numpy 插值
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.LANCZOS)
    a = np.array(img, dtype=np.float64)
    d = dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def build_test_hashes(test_data, size=(48, 64)):
    """对课设 test (051-055) 计算感知哈希用于查重."""
    hashes = []
    for file_name, _ in test_data:
        img = Image.open(f'{DATA_DIR}/Img/{file_name}').convert('L')
        img = img.resize(size, RESAMPLE_FILTER)
        arr = 1.0 - np.array(img, dtype=np.float32) / 255.0
        hashes.append(phash_simple(arr))
    return hashes


def collect_emnist_tensors(max_samples=None, dedup_against=None, dedup_threshold=5, verbose=True):
    """下载/加载 EMNIST byclass train split, 返回 [N, 1, 64, 48] float tensor.

    Args:
        max_samples: 最多取多少张 (None=全量 ~70 万)
        dedup_against: 课设 test 的 phash 列表; 若提供, 剔除汉明距离 < dedup_threshold 的 EMNIST 样本
        dedup_threshold: 汉明距离阈值 (8×8 hash 共 64 bit, < 5 视为近似重复)
    """
    os.makedirs(TORCH_DATA_ROOT, exist_ok=True)
    if verbose:
        print(f"[EMNIST] root={TORCH_DATA_ROOT}, 下载 byclass split (若已存在则跳过)...")
    ds = EMNIST(root=TORCH_DATA_ROOT, split='byclass', train=True, download=True)
    n_total = len(ds)
    if max_samples is None:
        max_samples = n_total
    else:
        max_samples = min(max_samples, n_total)
    if verbose:
        print(f"[EMNIST] 全集 {n_total}, 取 {max_samples}")

    tensors = []
    n_dropped = 0
    if verbose:
        print(f"[EMNIST] 开始遍历 + 转 tensor"
              f"{' + phash 去重' if dedup_against is not None else ''}...")
    log_step = max(1000, max_samples // 50)  # 至少打 ~50 次进度
    for i in range(max_samples):
        pil, _ = ds[i]  # ds[i] 返回 (PIL Image L mode, label)
        t = emnist_to_tensor(pil)
        if dedup_against is not None:
            h = phash_simple(t.squeeze(0).numpy())
            if any(hamming(h, ref) < dedup_threshold for ref in dedup_against):
                n_dropped += 1
                continue
        tensors.append(t)
        if verbose and (i + 1) % log_step == 0:
            print(f"\r[EMNIST] 进度 {i + 1}/{max_samples}, 保留 {len(tensors)}, "
                  f"剔除 {n_dropped}", end="", flush=True)
    if verbose:
        print()
    if verbose:
        print(f"[EMNIST] 完成. 保留 {len(tensors)}, 剔除 {n_dropped} (phash<{dedup_threshold})")
    return torch.stack(tensors)


class UnlabeledImageDataset(Dataset):
    """无标签预训练数据集. 输出 (image, dummy_label=0)."""
    def __init__(self, images):
        self.images = images  # torch tensor [N, 1, H, W] or list of [1, H, W]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if not torch.is_tensor(img):
            img = img.clone()
        return img.clone() if torch.is_tensor(img) else img, 0


def build_pretrain_pool(use_emnist=True, max_emnist=None, dedup=True, verbose=True):
    """组装无标签预训练池: 课设 train (001-050) + EMNIST byclass (可选, 已去重)."""
    train_data, _, test_data, _, _ = load_split_data()

    # 1. 课设 train 部分: 走 CharDataset 的 __getitem__ 取已归一化 tensor
    from data_utils import CharDataset
    course_ds = CharDataset(train_data, train=False)
    course_tensors = [course_ds[i][0] for i in range(len(course_ds))]
    course_tensor = torch.stack(course_tensors)
    if verbose:
        print(f"[Pool] 课设 train (001-050): {len(course_tensor)}")

    # 2. EMNIST 部分
    if use_emnist:
        ref_hashes = build_test_hashes(test_data) if dedup else None
        if dedup and verbose:
            print(f"[Pool] 计算课设 test phash 用于去重 ({len(ref_hashes)} 张)")
        emnist_tensor = collect_emnist_tensors(max_samples=max_emnist,
                                                dedup_against=ref_hashes,
                                                verbose=verbose)
        if verbose:
            print(f"[Pool] EMNIST byclass (去重后): {len(emnist_tensor)}")
        pool = torch.cat([course_tensor, emnist_tensor], dim=0)
    else:
        pool = course_tensor

    if verbose:
        print(f"[Pool] 合并预训练池: {len(pool)}")
    return UnlabeledImageDataset(pool)


if __name__ == "__main__":
    # smoke test: 取 2000 张 EMNIST, 验证 pipeline + 去重
    ds = build_pretrain_pool(use_emnist=True, max_emnist=2000, dedup=True, verbose=True)
    print(f"\nFinal pool size: {len(ds)}")
    img, lbl = ds[0]
    print(f"sample[0]: img.shape={tuple(img.shape)}, dtype={img.dtype}, "
          f"min={img.min():.3f}, max={img.max():.3f}, label={lbl}")
