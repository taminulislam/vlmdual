"""
Small utilities for the baseline training script:
- set_seed
- load_yaml
- build_optimizer + warmup_cosine_lr
- save/load checkpoint
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input size


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(params, cfg_optim: Dict[str, Any]) -> torch.optim.Optimizer:
    t = cfg_optim.get("type", "adamw").lower()
    if t != "adamw":
        raise ValueError(f"only adamw supported; got {t}")
    return torch.optim.AdamW(
        params,
        lr=float(cfg_optim["lr"]),
        weight_decay=float(cfg_optim["weight_decay"]),
        betas=tuple(cfg_optim.get("betas", (0.9, 0.999))),
    )


def warmup_cosine_lr(
    current_epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr_factor: float = 0.01,
) -> float:
    """Linear warmup then cosine decay to ``base_lr * min_lr_factor``."""
    if current_epoch < warmup_epochs:
        return base_lr * (current_epoch + 1) / max(1, warmup_epochs)
    progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_factor
    return min_lr + (base_lr - min_lr) * cos


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
