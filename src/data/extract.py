"""Распаковка zip-архива задачи в data/raw с починкой CP866-имён.

Использование:
    python -m src.data.extract --config configs/config.yaml
    python -m src.data.extract --zip "path/to/archive.zip" --out data/raw
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.utils import fix_zip_name, get_logger, load_config, resolve

log = get_logger("extract")


def find_zip(cfg: dict) -> Path:
    """Находит архив: сначала по имени из конфига, затем первый *.zip в корне."""
    from src.utils import project_root

    named = resolve(cfg["paths"]["zip"])
    if named.exists():
        return named
    candidates = sorted(project_root().glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Архив не найден: ни '{named}', ни *.zip в корне проекта."
        )
    log.warning("Архив '%s' не найден, беру самый большой *.zip: %s", named.name, candidates[0].name)
    return candidates[0]


def safe_target(out_dir: Path, decoded_name: str) -> Path:
    """Безопасный путь назначения (защита от path traversal)."""
    rel = Path(decoded_name)
    target = (out_dir / rel).resolve()
    if not str(target).startswith(str(out_dir.resolve())):
        raise ValueError(f"Небезопасный путь в архиве: {decoded_name}")
    return target


def extract(zip_path: Path, out_dir: Path, overwrite: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        log.info("В архиве %d записей. Распаковка в %s", len(infos), out_dir)
        for info in infos:
            decoded = fix_zip_name(info.filename)
            if info.is_dir() or decoded.endswith("/"):
                (out_dir / decoded).mkdir(parents=True, exist_ok=True)
                continue
            target = safe_target(out_dir, decoded)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                count += 1
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
            if count % 200 == 0:
                log.info("  ...распаковано %d файлов", count)
    log.info("Готово: %d файлов в %s", count, out_dir)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Распаковка архива задачи с починкой имён.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--zip", default=None, help="Переопределить путь к архиву.")
    ap.add_argument("--out", default=None, help="Переопределить выходную папку.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    zip_path = Path(args.zip).resolve() if args.zip else find_zip(cfg)
    out_dir = resolve(args.out) if args.out else resolve(cfg["paths"]["raw_dir"])
    log.info("Архив: %s (%.1f МБ)", zip_path.name, zip_path.stat().st_size / 1e6)
    extract(zip_path, out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
