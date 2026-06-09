"""Command line helpers for experiment scripts."""

import argparse
import random

import numpy as np
import torch

from project_constants import TRAIN_EPOCHS


def make_parser():
    return argparse.ArgumentParser()


def add_training_arguments(parser):
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="MixUp Beta 分布参数 (推荐 0.2)")
    parser.add_argument("--cutmix_alpha", type=float, default=0.0,
                        help="CutMix Beta 分布参数 (推荐 1.0)")
    return parser


def parse_training_args(configure_parser):
    parser = make_parser()
    configure_parser(parser)
    add_training_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    return args
