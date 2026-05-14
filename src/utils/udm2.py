from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


TILE_SIZE = 640

def load_udm2(path: Path):
    return rasterio.open(path)

def tile_cloud_fraction(udm2_ds, tx: int, ty: int, tile_size: int = TILE_SIZE) -> float:
    window = Window(
        col_off=tx * tile_size,
        row_off=ty * tile_size,
        width=tile_size,
        height=tile_size,
    )

    cloud = udm2_ds.read(2, window=window)
    shadow = udm2_ds.read(3, window=window)

    mask = (cloud > 0) | (shadow > 0)
    total = mask.size
    cloudy = mask.sum()

    if total == 0:
        return 0.0

    return cloudy / total
