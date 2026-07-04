"""Классификатор шлифов на бэкбонах torchvision + функции потерь.

Вместо U-Net/segmentation_models_pytorch (масок нет) — image-level классификация.
Поддержаны семейства efficientnet, resnet, mobilenet_v3 из torchvision.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Создаёт классификатор с заменённой головой под num_classes."""
    import torchvision.models as tvm

    name = backbone.lower()
    weights = "DEFAULT" if pretrained else None

    if not hasattr(tvm, name):
        raise ValueError(f"Неизвестный бэкбон '{backbone}'. См. torchvision.models.")
    model = getattr(tvm, name)(weights=weights)

    # Замена классификационной головы в зависимости от семейства
    if name.startswith("efficientnet") or name.startswith("mobilenet"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name.startswith("resnet") or name.startswith("resnext"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif name.startswith("densenet"):
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Не знаю, как заменить голову для '{backbone}'.")
    return model


class FocalLoss(nn.Module):
    """Мультиклассовый focal loss с опциональными весами классов."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    weight = class_weights if cfg["train"].get("class_weighted", False) else None
    kind = cfg["train"].get("loss", "ce").lower()
    if kind == "focal":
        return FocalLoss(gamma=cfg["train"].get("focal_gamma", 2.0), weight=weight)
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Неизвестный loss '{kind}' (ожидалось ce|focal)")


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """Обратно-частотные веса классов, нормированные к среднему 1."""
    import numpy as np

    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    inv = counts.sum() / (num_classes * counts)
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)
