"""Shared rendering helpers for label/prediction overlays.

Used by `scripts/labeling/15_label_disagreement.py` (whole-tile bucketing) and
`scripts/labeling/18_bucket_overlays.py` (per-detection RCA bucket renders).
Keeping these in one module guarantees both tools draw identical visuals so
the per-detection bucket inspection and the whole-tile disagreement view stay
visually consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_COLOR = (255, 90, 90)   # red — human label
PRED_COLOR = (90, 200, 255)   # cyan — model prediction
AGREE_COLOR = (120, 220, 120) # green — overlap (reserved; not currently drawn)


def parse_label_polys(label_path: Path, img_w: int, img_h: int) -> List[np.ndarray]:
    """Parse a YOLO polygon-segmentation label file → list of (N,2) pixel-coord arrays."""
    if not label_path.exists():
        return []
    polys: List[np.ndarray] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = np.array([float(p) for p in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        xs = coords[0::2] * img_w
        ys = coords[1::2] * img_h
        polys.append(np.column_stack([xs, ys]))
    return polys


def polys_to_mask(polys: List[np.ndarray], h: int, w: int) -> np.ndarray:
    """Rasterize a list of polygons (pixel coords) into a single binary mask."""
    if not polys:
        return np.zeros((h, w), dtype=bool)
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for p in polys:
        draw.polygon([(float(x), float(y)) for x, y in p], fill=1)
    return np.array(mask_img, dtype=bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def render_overlay(
    img_path: Path,
    label_polys: List[np.ndarray],
    pred_masks: List[np.ndarray],
    pred_confs: List[float],
    stats: dict,
    bucket: str,
    esri_inset_path: Path | None = None,
) -> Image.Image:
    """Compose tile + red label polys + cyan prediction masks + header strip.

    `stats` keys consumed: n_label_polys, n_pred, overall_iou, confident_fps,
    label_polys_uncovered, max_pred_conf. Missing keys render as ``?``.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    for p in label_polys:
        if len(p) >= 3:
            od.polygon([(float(x), float(y)) for x, y in p],
                       fill=LABEL_COLOR + (160,), outline=LABEL_COLOR + (255,), width=2)

    for pm, _conf in zip(pred_masks, pred_confs):
        if not pm.any():
            continue
        if pm.shape != (h, w):
            pm_img = Image.fromarray((pm.astype(np.uint8) * 255), mode="L").resize((w, h), Image.NEAREST)
            pm = np.array(pm_img) > 127
        rgba_layer = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_layer[pm] = (*PRED_COLOR, 160)
        layer_img = Image.fromarray(rgba_layer, mode="RGBA")
        overlay = Image.alpha_composite(overlay, layer_img)

    img_rgba = img.convert("RGBA")
    composed = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    bar_h = 56
    canvas_w = w + (h if esri_inset_path else 0)
    canvas = Image.new("RGB", (canvas_w, h + bar_h), (24, 24, 24))
    canvas.paste(composed, (0, bar_h))

    if esri_inset_path and Path(esri_inset_path).exists():
        esri = Image.open(esri_inset_path).convert("RGB").resize((h, h), Image.LANCZOS)
        canvas.paste(esri, (w, bar_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    title = f"{img_path.name} — bucket: {bucket}"
    sub = (f"label_polys={stats.get('n_label_polys', '?')} pred={stats.get('n_pred', '?')} "
           f"IoU={stats.get('overall_iou', '?')} confident_FP={stats.get('confident_fps', '?')} "
           f"uncovered_FN={stats.get('label_polys_uncovered', '?')} "
           f"max_conf={stats.get('max_pred_conf', '?')}")
    legend = "red=label  cyan=prediction" + ("  | right: Esri reference (high-res)" if esri_inset_path else "")
    draw.text((12, 6), title, font=font, fill=(240, 240, 240))
    draw.text((12, 26), sub, font=small, fill=(180, 220, 180))
    draw.text((12, 41), legend, font=small, fill=(170, 170, 170))

    return canvas
