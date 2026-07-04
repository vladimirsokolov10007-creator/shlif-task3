"""torch Dataset для фото шлифов + аугментации на torchvision.transforms.v2.

Аугментации из ТЗ (flips, rotate90, elastic, яркость/контраст/оттенок/насыщенность)
реализованы средствами torchvision, без albumentations.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRotate90:
    """Случайный поворот на k*90° (torchvision v2 не имеет прямого аналога)."""

    def __call__(self, img):
        import torchvision.transforms.v2.functional as F

        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            img = F.rotate(img, 90 * k)
        return img


def build_transforms(cfg: dict, train: bool) -> Callable:
    from torchvision.transforms import v2

    size = cfg["train"]["img_size"]
    ops: list = [v2.ToImage()]

    if train:
        aug = cfg["augment"]
        ops += [v2.Resize((size, size), antialias=True)]
        if aug.get("hflip", 0):
            ops.append(v2.RandomHorizontalFlip(p=aug["hflip"]))
        if aug.get("vflip", 0):
            ops.append(v2.RandomVerticalFlip(p=aug["vflip"]))
        if aug.get("rotate90", False):
            ops.append(RandomRotate90())
        if aug.get("elastic_p", 0):
            ops.append(v2.RandomApply([v2.ElasticTransform(alpha=40.0, sigma=5.0)], p=aug["elastic_p"]))
        cj = aug.get("color_jitter") or {}
        if cj:
            ops.append(v2.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))
    else:
        ops += [v2.Resize((size, size), antialias=True)]

    ops += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(ops)


class ShlifPhotoDataset(Dataset):
    """Датасет фото шлифов. Ожидает DataFrame с колонками path, class_id."""

    def __init__(self, df: pd.DataFrame, transforms: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            img = im.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        label = int(row["class_id"])
        return img, label


def tiles_to_tensor(tiles_rgb: np.ndarray, size: int) -> torch.Tensor:
    """Батч тайлов (N,H,W,3 uint8) -> нормированный тензор (N,3,size,size).

    Используется на инференсе панорам (без аугментаций).
    """
    from torchvision.transforms import v2

    tf = v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return torch.stack([tf(np.ascontiguousarray(t)) for t in tiles_rgb])
