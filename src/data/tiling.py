"""Нарезка больших панорам на тайлы с перекрытием и обратная сшивка.

Масок для панорам нет, поэтому «сегментация» слабая: панораму режем на тайлы,
классификатор даёт вероятности классов на тайл, а карта классов собирается
блендингом вероятностей по зонам перекрытия.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def compute_tiles(height: int, width: int, tile: int, overlap: float) -> List[Tile]:
    """Список тайлов размера `tile` с перекрытием `overlap` (доля 0..1).

    Последний тайл в ряду/столбце прижимается к краю, чтобы покрыть всё изображение.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap должен быть в [0, 1)")
    step = max(1, int(round(tile * (1 - overlap))))

    def starts(size: int) -> List[int]:
        if size <= tile:
            return [0]
        pts = list(range(0, size - tile + 1, step))
        if pts[-1] != size - tile:
            pts.append(size - tile)
        return pts

    ys = starts(height)
    xs = starts(width)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append(Tile(y, x, min(y + tile, height), min(x + tile, width)))
    return tiles


def iter_tiles(image: np.ndarray, tile: int, overlap: float) -> Iterator[Tuple[Tile, np.ndarray]]:
    """Итерирует (координаты, вырезанный тайл HxWxC)."""
    h, w = image.shape[:2]
    for t in compute_tiles(h, w, tile, overlap):
        yield t, image[t.y0:t.y1, t.x0:t.x1]


def blend_weights(h: int, w: int) -> np.ndarray:
    """Веса для мягкого блендинга (сильнее в центре тайла, слабее по краям).

    Треугольная (Bartlett) оконная функция по обеим осям -> плавные швы.
    """
    wy = np.bartlett(max(h, 2))[:h] if h > 2 else np.ones(h)
    wx = np.bartlett(max(w, 2))[:w] if w > 2 else np.ones(w)
    wy = np.clip(wy, 1e-3, None)
    wx = np.clip(wx, 1e-3, None)
    return np.outer(wy, wx).astype(np.float32)


class ProbCanvas:
    """Аккумулятор вероятностей классов на масштабированном холсте.

    Тайлы больших панорам проецируются на холст с ограниченной стороной
    (map_max_side), чтобы держать память под контролем. Перекрытия усредняются
    с весами blend_weights.
    """

    def __init__(self, full_h: int, full_w: int, num_classes: int, max_side: int):
        scale = min(1.0, max_side / max(full_h, full_w))
        self.scale = scale
        self.H = max(1, int(round(full_h * scale)))
        self.W = max(1, int(round(full_w * scale)))
        self.num_classes = num_classes
        self.acc = np.zeros((self.H, self.W, num_classes), dtype=np.float32)
        self.wsum = np.zeros((self.H, self.W, 1), dtype=np.float32)

    def add(self, tile: Tile, probs: np.ndarray) -> None:
        """probs: вектор длины num_classes для данного тайла."""
        y0 = int(round(tile.y0 * self.scale))
        x0 = int(round(tile.x0 * self.scale))
        y1 = max(y0 + 1, int(round(tile.y1 * self.scale)))
        x1 = max(x0 + 1, int(round(tile.x1 * self.scale)))
        y1, x1 = min(y1, self.H), min(x1, self.W)
        wsm = blend_weights(y1 - y0, x1 - x0)[..., None]
        self.acc[y0:y1, x0:x1] += wsm * probs[None, None, :]
        self.wsum[y0:y1, x0:x1] += wsm

    def prob_map(self) -> np.ndarray:
        """Нормированная карта вероятностей HxWxC."""
        return self.acc / np.clip(self.wsum, 1e-6, None)

    def label_map(self) -> np.ndarray:
        """Карта классов HxW (argmax)."""
        return self.prob_map().argmax(axis=-1).astype(np.int32)

    def class_fractions(self) -> np.ndarray:
        """Доли классов по площади (по покрытым пикселям холста)."""
        labels = self.label_map()
        covered = self.wsum[..., 0] > 1e-6
        vals = labels[covered]
        if vals.size == 0:
            return np.zeros(self.num_classes, dtype=np.float32)
        counts = np.bincount(vals, minlength=self.num_classes).astype(np.float32)
        return counts / counts.sum()
