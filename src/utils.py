"""Общие утилиты: конфиг, сид, логгер, метрики, починка CP866-имён."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


# --------------------------------------------------------------------------- #
#  Конфиг и пути
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Корень проекта = на два уровня выше этого файла (src/utils.py)."""
    return Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = "configs/config.yaml") -> Dict[str, Any]:
    """Загружает YAML-конфиг. Относительные пути внутри резолвятся от корня."""
    import yaml

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(cfg_path: str | os.PathLike) -> Path:
    """Превращает относительный путь из конфига в абсолютный (от корня проекта)."""
    p = Path(cfg_path)
    return p if p.is_absolute() else project_root() / p


# --------------------------------------------------------------------------- #
#  Воспроизводимость
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
#  Логгер
# --------------------------------------------------------------------------- #
def get_logger(name: str = "shlif", logfile: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # уже настроен
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    # На Windows консоль часто в cp1251: не даём падать на не-кодируемых символах.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
#  Имена из архива: CP437 -> CP866 (кириллица в старых zip)
# --------------------------------------------------------------------------- #
def fix_zip_name(name: str) -> str:
    """Чинит искажённое имя записи из zip (legacy-кодировка)."""
    for enc in ("cp866", "cp1251"):
        try:
            return name.encode("cp437").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


# --------------------------------------------------------------------------- #
#  Метрики классификации
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, num_classes: int) -> Dict[str, Any]:
    """per-class F1, macro-F1, accuracy, confusion matrix."""
    from sklearn.metrics import confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "per_class_f1": [float(x) for x in per_class],
        "macro_f1": float(macro),
        "accuracy": acc,
        "confusion": cm.tolist(),
    }
