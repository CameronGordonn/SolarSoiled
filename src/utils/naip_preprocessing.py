"""NAIP image preprocessing: tone mapping and band-order detection."""

import numpy as np
from skimage import exposure
from PIL import Image
from pathlib import Path
from typing import Tuple, Union


def detect_rgb_band_order(src) -> Tuple[int, int, int]:
    """Return (r_band, g_band, b_band) 1-indexed from a rasterio dataset."""
    if hasattr(src, "descriptions") and src.descriptions:
        descriptions = [desc.lower() if desc else "" for desc in src.descriptions]
        band_map = {}
        for idx, desc in enumerate(descriptions, 1):
            if "red" in desc:
                band_map["r"] = idx
            elif "green" in desc:
                band_map["g"] = idx
            elif "blue" in desc:
                band_map["b"] = idx
        if len(band_map) == 3:
            return (band_map["r"], band_map["g"], band_map["b"])
    return (1, 2, 3)


def tone_map_naip_clahe(rgb_data: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """Apply per-channel CLAHE to a (3, H, W) uint8/uint16 array; return uint8."""
    if rgb_data.dtype == np.uint16:
        rgb_data = (rgb_data / 256).astype(np.uint8)
    elif rgb_data.dtype != np.uint8:
        rgb_data = rgb_data.astype(np.uint8)

    # equalize_adapthist requires 2D input per channel — passing (3,H,W) or (H,W,3) distorts output
    result = np.zeros_like(rgb_data, dtype=np.float64)
    for c in range(rgb_data.shape[0]):
        result[c] = exposure.equalize_adapthist(rgb_data[c] / 255.0, kernel_size=None, clip_limit=clip_limit, nbins=256)
    return (result * 255).astype(np.uint8)


def tone_map_naip_gamma(rgb_data: np.ndarray, gamma: float = 0.85, saturation_boost: float = 1.1) -> np.ndarray:
    """Apply gamma correction and saturation boost to a (3, H, W) array; return uint8."""
    if rgb_data.dtype == np.uint16:
        rgb_data = (rgb_data / 256).astype(np.uint8)
    elif rgb_data.dtype != np.uint8:
        rgb_data = rgb_data.astype(np.uint8)

    from skimage.color import rgb2hsv, hsv2rgb
    rgb_gamma = np.power(rgb_data / 255.0, gamma)
    hsv = rgb2hsv(rgb_gamma)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 1)
    return (hsv2rgb(hsv) * 255).astype(np.uint8)


def save_tile_as_png(rgb_data: np.ndarray, output_path: Union[str, Path], tone_mapping_method: str = "clahe") -> None:
    """Save a (3, H, W) RGB array to PNG after tone mapping."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tone_mapping_method == "clahe":
        rgb_processed = tone_map_naip_clahe(rgb_data)
    elif tone_mapping_method == "gamma":
        rgb_processed = tone_map_naip_gamma(rgb_data)
    else:
        raise ValueError(f"Unknown tone_mapping_method: {tone_mapping_method}")
    Image.fromarray(np.transpose(rgb_processed, (1, 2, 0)), mode="RGB").save(output_path)
