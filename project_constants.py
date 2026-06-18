"""Shared project constants.

支持环境变量覆盖, 用于本地 (默认) vs AutoDL/Linux 4090 (env 覆盖):
    export HANDCHAR_BATCH_SIZE=256
    export HANDCHAR_NUM_WORKERS=8
    export HANDCHAR_LR=0.004
"""

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("HANDCHAR_BATCH_SIZE", 64))
LEARNING_RATE = float(os.environ.get("HANDCHAR_LR", 0.001))
NUM_CLASSES = 62
AUX_CLASSES = 3   # digit / upper / lower
DROPOUT = 0.146
NUM_WORKERS = int(os.environ.get("HANDCHAR_NUM_WORKERS", 0))  # Windows 默认 0; Linux 可设 8
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # 锁定确定性: 同 seed 两次结果一致

# 训练 epoch 集中管理 (大一统方案统一 20 epoch)
TRAIN_EPOCHS = 20
PRETRAIN_EPOCHS = 20
META_EPOCHS = 10

CONFUSABLE_PAIRS = (
    ("0", "O"),
    ("0", "o"),
    ("O", "o"),
    ("1", "I"),
    ("1", "l"),
    ("I", "l"),
    ("5", "S"),
    ("C", "c"),
)

# Pair aux head: 主头治不动的混淆字符 + "other", 11-way 细分类任务.
# 强迫 unified 特征学到 0/O/o, 1/I/l 这种细微区分.
CONFUSABLE_UNION = ("0", "O", "o", "1", "I", "l", "5", "S", "C", "c")
PAIR_NUM_CLASSES = len(CONFUSABLE_UNION) + 1  # 11 = 10 + "other"
PAIR_OTHER_IDX = len(CONFUSABLE_UNION)        # 10
