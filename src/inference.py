"""Инференс на полной панораме: тайлинг -> классификация тайлов ->
блендинг вероятностей по overlap -> карта классов + доли классов + оверлей.

Использование:
    python -m src.inference --config configs/config.yaml \
        --image "data/raw/.../Панорамы/10.jpg" \
        --checkpoint checkpoints/efficientnet_b0_fold0.pt \
        --out outputs/
    python -m src.inference --all-panoramas --checkpoint checkpoints/efficientnet_b0_fold0.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import tiles_to_tensor
from src.data.tiling import ProbCanvas, compute_tiles
from src.model import build_model
from src.utils import get_logger, load_config, resolve

Image.MAX_IMAGE_PIXELS = None
log = get_logger("inference")

# Палитра для оверлея (RGB), по индексам классов
PALETTE = np.array([
    [200, 60, 60],    # 0
    [60, 160, 60],    # 1
    [60, 90, 200],    # 2
    [210, 180, 40],   # 3
    [160, 60, 200],   # 4
    [40, 190, 190],   # 5
], dtype=np.uint8)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt["img_size"]


def read_panorama(path: Path, downsample: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if downsample > 1:
            im = im.resize((im.width // downsample, im.height // downsample), Image.BILINEAR)
        return np.asarray(im)


@torch.no_grad()
def predict_panorama(model, image: np.ndarray, classes, img_size, cfg, device) -> dict:
    inf = cfg["inference"]
    tile, overlap = inf["tile_size"], inf["overlap"]
    bs = inf["batch_size"]
    h, w = image.shape[:2]
    num_classes = len(classes)

    tiles = compute_tiles(h, w, tile, overlap)
    canvas = ProbCanvas(h, w, num_classes, inf["map_max_side"])
    log.info("  панорама %dx%d -> %d тайлов (%d px, overlap %.0f%%)",
             w, h, len(tiles), tile, overlap * 100)

    for i in range(0, len(tiles), bs):
        batch_tiles = tiles[i:i + bs]
        crops = [image[t.y0:t.y1, t.x0:t.x1] for t in batch_tiles]
        x = tiles_to_tensor(crops, img_size).to(device)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        for t, p in zip(batch_tiles, probs):
            canvas.add(t, p)
        if (i // bs) % 20 == 0 and i > 0:
            log.info("    ...%d/%d тайлов", i, len(tiles))

    fractions = canvas.class_fractions()
    return {
        "classes": classes,
        "fractions": {c: float(f) for c, f in zip(classes, fractions)},
        "label_map": canvas.label_map(),
        "prob_map": canvas.prob_map(),
    }


def save_overlay(image: np.ndarray, label_map: np.ndarray, out_png: Path,
                 alpha: float = 0.45, max_side: int = 0) -> None:
    """Сохраняет оверлей карты классов поверх панорамы.

    max_side > 0 -> компактное превью с ограничением большей стороны (по умолчанию
    полноразмерный PNG может весить десятки МБ). Пропорции сохраняются.
    """
    from PIL import Image as PImage

    h, w = image.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        base = np.asarray(PImage.fromarray(image).resize((w, h), PImage.BILINEAR))
    else:
        base = image
    color = PALETTE[label_map % len(PALETTE)]
    color_full = np.asarray(PImage.fromarray(color).resize((w, h), PImage.NEAREST))
    blended = (base.astype(np.float32) * (1 - alpha) + color_full.astype(np.float32) * alpha)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    PImage.fromarray(blended.astype(np.uint8)).save(out_png)


def run_one(model, classes, img_size, cfg, device, image_path: Path, out_dir: Path) -> dict:
    log.info("Панорама: %s", image_path.name)
    image = read_panorama(image_path, cfg["inference"].get("read_downsample", 1))
    result = predict_panorama(model, image, classes, img_size, cfg, device)

    stem = image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(image, result["label_map"], out_dir / f"{stem}_overlay.png",
                 max_side=cfg["inference"].get("overlay_max_side", 0))
    np.save(out_dir / f"{stem}_labelmap.npy", result["label_map"].astype(np.uint8))
    frac = result["fractions"]
    with open(out_dir / f"{stem}_fractions.json", "w", encoding="utf-8") as f:
        json.dump({"image": image_path.name, "fractions": frac}, f, ensure_ascii=False, indent=2)
    log.info("  доли классов: %s", {k: round(v, 4) for k, v in frac.items()})
    return {"image": image_path.name, **frac}


def find_panoramas(cfg: dict) -> list[Path]:
    df = pd.read_csv(resolve(cfg["paths"]["index_csv"]))
    return [Path(p) for p in df[df["part"] == "panorama"]["path"].tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Инференс на панорамах.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="Путь к одной панораме.")
    ap.add_argument("--all-panoramas", action="store_true", help="Все панорамы из index.csv.")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, img_size = load_checkpoint(resolve(args.checkpoint), device)
    out_dir = resolve(args.out)

    if args.image:
        images = [Path(args.image)]
    elif args.all_panoramas:
        images = find_panoramas(cfg)
    else:
        ap.error("Укажи --image ИЛИ --all-panoramas")

    summary = []
    for img in images:
        summary.append(run_one(model, classes, img_size, cfg, device, img, out_dir))

    pd.DataFrame(summary).to_csv(out_dir / "panorama_fractions.csv", index=False)
    log.info("Сводка долей: %s", out_dir / "panorama_fractions.csv")


if __name__ == "__main__":
    main()
