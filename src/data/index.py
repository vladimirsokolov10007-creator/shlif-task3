"""Обход data/raw -> index.csv (path, class, class_id, part, width, height)
и печать EDA-отчёта (распределение классов, размеры, панорамы).

Использование:
    python -m src.data.index --config configs/config.yaml
    python -m src.data.index --report-only        # только отчёт, без перезаписи csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

from PIL import Image

from src.utils import get_logger, load_config, resolve

Image.MAX_IMAGE_PIXELS = None  # панорамы > лимита PIL — доверяем своим данным
log = get_logger("index")

UNKNOWN = "__unknown__"


def norm(s: str) -> str:
    return s.strip().lower()


def classify_path(rel_parts: list[str], class_map: dict[str, str]) -> Optional[str]:
    """Возвращает класс по самому глубокому совпавшему сегменту пути."""
    cmap = {norm(k): v for k, v in class_map.items()}
    match = None
    for seg in rel_parts:  # от корня к листу -> последнее совпадение = самое глубокое
        if norm(seg) in cmap:
            match = cmap[norm(seg)]
    return match


def read_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось прочитать размер %s: %s", path.name, e)
        return -1, -1


def build_index(cfg: dict, with_sizes: bool = True) -> list[dict]:
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    exts = {e.lower() for e in cfg["data"]["image_exts"]}
    pano_dir = norm(cfg["data"]["panorama_dirname"])
    class_map = cfg["data"]["class_map"]
    classes = cfg["data"]["classes"]
    class_to_id = {c: i for i, c in enumerate(classes)}

    rows: list[dict] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(raw_dir)
        parts = list(rel.parts)

        is_pano = any(norm(p) == pano_dir for p in parts)
        if is_pano:
            cls, cid, part = "", -1, "panorama"
        else:
            cls = classify_path(parts, class_map) or UNKNOWN
            cid = class_to_id.get(cls, -1)
            part = "photo"

        w, h = read_size(path) if with_sizes else (-1, -1)
        rows.append(
            {
                "path": str(path),
                "rel": str(rel),
                "class": cls,
                "class_id": cid,
                "part": part,
                "width": w,
                "height": h,
            }
        )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "rel", "class", "class_id", "part", "width", "height"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("index.csv записан: %s (%d строк)", out_csv, len(rows))


def report(rows: list[dict], cfg: dict) -> None:
    photos = [r for r in rows if r["part"] == "photo"]
    panos = [r for r in rows if r["part"] == "panorama"]
    titles = cfg["data"].get("class_titles", {})

    log.info("=" * 64)
    log.info("EDA-ОТЧЁТ")
    log.info("=" * 64)
    log.info("Всего файлов: %d  |  фото: %d  |  панорам: %d", len(rows), len(photos), len(panos))

    # распределение классов
    from collections import Counter

    cnt = Counter(r["class"] for r in photos)
    log.info("-" * 64)
    log.info("Распределение фото по классам:")
    total = max(len(photos), 1)
    for cls in cfg["data"]["classes"] + [UNKNOWN]:
        n = cnt.get(cls, 0)
        if n == 0 and cls == UNKNOWN:
            continue
        bar = "#" * int(40 * n / total)
        log.info("  %-12s %5d (%5.1f%%) %s", cls, n, 100 * n / total, bar)
        if titles.get(cls):
            log.info("               -> %s", titles[cls])
    if cnt.get(UNKNOWN):
        log.warning("  ВНИМАНИЕ: %d фото не отнесены к классу (нет в class_map).", cnt[UNKNOWN])

    # размеры фото
    sized = [r for r in photos if r["width"] > 0]
    if sized:
        ws = sorted(r["width"] for r in sized)
        hs = sorted(r["height"] for r in sized)
        log.info("-" * 64)
        log.info("Размеры фото: width %d..%d, height %d..%d (медиана %dx%d)",
                 ws[0], ws[-1], hs[0], hs[-1], ws[len(ws) // 2], hs[len(hs) // 2])

    # панорамы
    if panos:
        log.info("-" * 64)
        log.info("Панорамы (%d):", len(panos))
        for r in sorted(panos, key=lambda x: x["rel"]):
            mp = (r["width"] * r["height"] / 1e6) if r["width"] > 0 else 0
            log.info("  %-40s %6dx%-6d (%.0f Мп)", Path(r["rel"]).name, r["width"], r["height"], mp)
    log.info("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Индексация data/raw и EDA-отчёт.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--report-only", action="store_true", help="Не перезаписывать csv.")
    ap.add_argument("--no-sizes", action="store_true", help="Не читать размеры (быстрее).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = build_index(cfg, with_sizes=not args.no_sizes)
    if not args.report_only:
        write_csv(rows, resolve(cfg["paths"]["index_csv"]))
    report(rows, cfg)


if __name__ == "__main__":
    main()
