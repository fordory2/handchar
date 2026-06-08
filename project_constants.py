"""Shared project constants."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_CLASSES = 62
DROPOUT = 0.146
NUM_WORKERS = 0  # Windows: 单进程避免 _cache 多副本、速度波动
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = False

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
