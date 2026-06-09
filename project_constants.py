"""Shared project constants."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_CLASSES = 62
AUX_CLASSES = 3   # digit / upper / lower
DROPOUT = 0.146
NUM_WORKERS = 0  # Windows: 单进程避免 _cache 多副本、速度波动
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = False

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
