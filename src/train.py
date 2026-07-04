"""Обучение классификатора шлифов: 5-fold StratifiedCV, per-class F1 + macro-F1,
чекпоинты, early stopping, логирование.

Использование:
    python -m src.train --config configs/config.yaml            # все фолды
    python -m src.train --fold 0                                # только фолд 0
    python -m src.train --fold 0 --epochs 5 --backbone resnet18 # быстрый прогон
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.dataset import ShlifPhotoDataset, build_transforms
from src.model import build_loss, build_model, compute_class_weights
from src.utils import classification_metrics, get_logger, load_config, resolve, set_seed


def load_photo_df(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg["paths"]["index_csv"]))
    df = df[(df["part"] == "photo") & (df["class_id"] >= 0)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("В index.csv нет размеченных фото. Запусти src.data.index.")
    return df


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, criterion) -> dict:
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
    m = classification_metrics(y_true, y_pred, num_classes)
    m["loss"] = loss_sum / max(n, 1)
    return m


def train_one_fold(cfg: dict, df: pd.DataFrame, train_idx, val_idx, fold: int,
                   run_dir: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["classes"]
    num_classes = len(classes)
    tr = cfg["train"]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = ShlifPhotoDataset(train_df, build_transforms(cfg, train=True))
    val_ds = ShlifPhotoDataset(val_df, build_transforms(cfg, train=False))

    train_ld = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                          num_workers=tr["num_workers"], drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False,
                        num_workers=tr["num_workers"])

    model = build_model(tr["backbone"], num_classes, tr["pretrained"]).to(device)
    weights = compute_class_weights(train_df["class_id"].values, num_classes).to(device)
    criterion = build_loss(cfg, weights).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = None
    if tr.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tr["epochs"])

    log.info("Fold %d | train=%d val=%d | классы-веса=%s | device=%s",
             fold, len(train_df), len(val_df),
             np.round(weights.cpu().numpy(), 2).tolist(), device)

    ckpt_path = resolve(cfg["paths"]["checkpoints_dir"]) / f"{tr['backbone']}_fold{fold}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_csv = run_dir / f"fold{fold}_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "macro_f1", "accuracy"])

    best_metric, best_epoch, patience = -1.0, -1, 0
    best_state = None
    metric_key = tr.get("val_metric", "macro_f1")

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        if scheduler:
            scheduler.step()
        train_loss = run_loss / max(seen, 1)

        val = evaluate(model, val_ld, device, num_classes, criterion)
        cur = val[metric_key]
        dt = time.time() - t0
        log.info("  e%02d | train_loss=%.4f val_loss=%.4f macro_f1=%.4f acc=%.4f | %.1fs",
                 epoch, train_loss, val["loss"], val["macro_f1"], val["accuracy"], dt)
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val['loss']:.4f}",
                                    f"{val['macro_f1']:.4f}", f"{val['accuracy']:.4f}"])

        if cur > best_metric:
            best_metric, best_epoch, patience = cur, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state": best_state,
                "backbone": tr["backbone"],
                "classes": classes,
                "img_size": tr["img_size"],
                "fold": fold,
                "metric": best_metric,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= tr["early_stopping_patience"]:
                log.info("  early stopping на эпохе %d (best e%d, %s=%.4f)",
                         epoch, best_epoch, metric_key, best_metric)
                break

    # финальная оценка лучшей модели
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, val_ld, device, num_classes, criterion)
    log.info("Fold %d ИТОГ | macro_f1=%.4f acc=%.4f", fold, final["macro_f1"], final["accuracy"])
    for cls, f1 in zip(classes, final["per_class_f1"]):
        log.info("    F1[%-11s] = %.4f", cls, f1)
    final["best_epoch"] = best_epoch
    final["checkpoint"] = str(ckpt_path)
    final["val_true"] = [int(v) for v in val_df["class_id"].values]
    return final


def summarize(cfg: dict, fold_results: list[dict], run_dir: Path, log) -> None:
    classes = cfg["data"]["classes"]
    per_class = np.array([r["per_class_f1"] for r in fold_results])
    macro = np.array([r["macro_f1"] for r in fold_results])
    log.info("=" * 64)
    log.info("СВОДКА ПО %d ФОЛДАМ", len(fold_results))
    log.info("  macro_f1: %.4f ± %.4f", macro.mean(), macro.std())
    for i, cls in enumerate(classes):
        log.info("  F1[%-11s]: %.4f ± %.4f", cls, per_class[:, i].mean(), per_class[:, i].std())
    log.info("=" * 64)
    summary = {
        "classes": classes,
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(macro.std()),
        "per_class_f1_mean": per_class.mean(0).tolist(),
        "folds": [{k: v for k, v in r.items() if k != "val_true"} for r in fold_results],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Сводка сохранена: %s", run_dir / "summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение классификатора шлифов (5-fold CV).")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--fold", type=int, default=None, help="Обучить только этот фолд.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(cfg["project"]["seed"])
    run_dir = resolve(cfg["paths"]["runs_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train", run_dir / "train.log")
    log.info("Конфиг: backbone=%s img=%d epochs=%d bs=%d loss=%s",
             cfg["train"]["backbone"], cfg["train"]["img_size"], cfg["train"]["epochs"],
             cfg["train"]["batch_size"], cfg["train"]["loss"])

    df = load_photo_df(cfg)
    y = df["class_id"].values
    skf = StratifiedKFold(n_splits=cfg["train"]["num_folds"], shuffle=True,
                          random_state=cfg["project"]["seed"])
    splits = list(skf.split(np.zeros(len(y)), y))

    fold_results = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        if args.fold is not None and fold != args.fold:
            continue
        res = train_one_fold(cfg, df, tr_idx, val_idx, fold, run_dir, log)
        fold_results.append(res)

    if len(fold_results) > 1:
        summarize(cfg, fold_results, run_dir, log)


if __name__ == "__main__":
    main()
