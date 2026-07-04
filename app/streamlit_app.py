"""Streamlit-демо: загрузка панорамы/фото -> карта классов + доли.

Запуск:
    pip install streamlit
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference import PALETTE, load_checkpoint, predict_panorama, save_overlay  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Шлиф — классификация по сортам", layout="wide")
st.title("Шлиф · сегментация панорамы по сортам руды")

cfg = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_dir = resolve(cfg["paths"]["checkpoints_dir"])
ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
if not ckpts:
    st.warning("Нет чекпоинтов в %s. Сначала обучите модель (python -m src.train)." % ckpt_dir)
    st.stop()

ckpt_name = st.sidebar.selectbox("Чекпоинт", [c.name for c in ckpts])
model, classes, img_size = load_checkpoint(ckpt_dir / ckpt_name, device)
titles = cfg["data"].get("class_titles", {})

uploaded = st.file_uploader("Загрузите панораму или фото шлифа", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    image = np.asarray(Image.open(uploaded).convert("RGB"))
    st.write(f"Размер: {image.shape[1]}×{image.shape[0]}")
    with st.spinner("Инференс по тайлам..."):
        result = predict_panorama(model, image, classes, img_size, cfg, device)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оверлей карты классов")
        tmp = Path("outputs/_streamlit_overlay.png")
        save_overlay(image, result["label_map"], tmp)
        st.image(str(tmp), use_container_width=True)
    with col2:
        st.subheader("Доли классов")
        for cls, frac in result["fractions"].items():
            st.write(f"**{titles.get(cls, cls)}** — {frac*100:.1f}%")
            st.progress(min(1.0, frac))

    st.caption("Цвета классов: " + ", ".join(
        f"{titles.get(c, c)}" for c in classes))
