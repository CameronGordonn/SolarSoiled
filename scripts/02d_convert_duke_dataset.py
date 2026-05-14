"""Convert Duke/Bradbury solar array dataset to YOLOv11 polygon segmentation format."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.naip_preprocessing import detect_rgb_band_order, tone_map_naip_clahe
from utils.tile_metadata import TileIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TILE_SIZE = 640
CLASS_ID = 0
SPLITS = ("train", "val", "test")

_transformer_cache: dict = {}


def get_transformer(target_crs: str) -> Transformer:
    if target_crs not in _transformer_cache:
        _transformer_cache[target_crs] = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return _transformer_cache[target_crs]


def load_annotations(meta_csv: Path, ll_csv: Path) -> tuple:
    meta = pd.read_csv(meta_csv)
    ll = pd.read_csv(ll_csv)
    logger.info(f"Loaded {len(meta)} polygon metadata rows, {len(ll)} vertex rows")
    ll = ll.merge(meta[["polygon_id", "image_name"]], on="polygon_id", how="left")
    missing = ll["image_name"].isna().sum()
    if missing:
        logger.warning(f"{missing} vertex rows have no metadata match — dropping")
        ll = ll.dropna(subset=["image_name"])
    return meta, ll


def build_polygon_map(ll: pd.DataFrame) -> dict:
    index: dict = {}
    skipped = 0
    for _, row in ll.iterrows():
        n = int(row["number_vertices"])
        coords = []
        for i in range(1, n + 1):
            lon = row.get(f"lon{i}")
            lat = row.get(f"lat{i}")
            if pd.notna(lon) and pd.notna(lat):
                coords.append((float(lon), float(lat)))
        if len(coords) < 3:
            skipped += 1
            continue
        stem = Path(str(row["image_name"])).stem
        if stem not in index:
            index[stem] = []
        index[stem].append((int(row["polygon_id"]), coords))
    if skipped:
        logger.warning(f"Skipped {skipped} polygons with < 3 valid vertices")
    logger.info(f"Built polygon map: {len(index)} unique source image stems")
    return index


def lonlat_to_pixel(lon: float, lat: float, transformer: Transformer, inv_transform) -> tuple:
    proj_x, proj_y = transformer.transform(lon, lat)
    return inv_transform * (proj_x, proj_y)


def polygon_to_pixels(wgs84_coords: list, transformer: Transformer, inv_full_transform) -> list:
    return [lonlat_to_pixel(lon, lat, transformer, inv_full_transform) for lon, lat in wgs84_coords]


def clip_and_normalize(pixel_coords: list, tile_x0: int, tile_y0: int, tile_w: int, tile_h: int,
                        out_w: int, out_h: int) -> list | None:
    """Clip polygon to tile bounds and normalize to the output image size (out_w × out_h).

    tile_w/tile_h are the content dimensions (may be < out_w/out_h for padded edge tiles).
    out_w/out_h are the actual saved PNG dimensions — always tile_size for consistency.
    """
    xs = [c[0] for c in pixel_coords]
    ys = [c[1] for c in pixel_coords]
    if max(xs) < tile_x0 or min(xs) > tile_x0 + tile_w:
        return None
    if max(ys) < tile_y0 or min(ys) > tile_y0 + tile_h:
        return None
    norm = []
    for px, py in pixel_coords:
        cx = max(tile_x0, min(tile_x0 + tile_w, px))
        cy = max(tile_y0, min(tile_y0 + tile_h, py))
        norm.append(((cx - tile_x0) / out_w, (cy - tile_y0) / out_h))
    deduped = [norm[0]]
    for pt in norm[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped if len(deduped) >= 3 else None


def make_yolo_line(norm_pairs: list) -> str:
    flat = [v for pair in norm_pairs for v in pair]
    return f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in flat)


def process_image(ortho_path: Path, polygons: list, tile_size: int) -> list:
    records = []
    with rasterio.open(ortho_path) as src:
        img_w, img_h = src.width, src.height
        crs = src.crs.to_string()
        r_band, g_band, b_band = detect_rgb_band_order(src)
        inv_full = ~src.transform
        transformer = get_transformer(crs)
        n_cols = (img_w + tile_size - 1) // tile_size
        n_rows = (img_h + tile_size - 1) // tile_size
        logger.info(f"    {img_h}×{img_w}px → {n_rows}r×{n_cols}c = {n_rows*n_cols} tiles, {len(polygons)} polygons")

        pixel_polygons = []
        for pid, wgs84_coords in polygons:
            try:
                px_coords = polygon_to_pixels(wgs84_coords, transformer, inv_full)
                pixel_polygons.append((pid, px_coords))
            except Exception as e:
                logger.debug(f"    Polygon {pid} projection failed: {e} — skipping")

        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * tile_size
                y0 = row * tile_size
                win_w = min(tile_size, img_w - x0)
                win_h = min(tile_size, img_h - y0)
                window = Window(x0, y0, win_w, win_h)
                tile_transform = src.window_transform(window)

                label_lines = []
                for _pid, px_coords in pixel_polygons:
                    norm = clip_and_normalize(px_coords, x0, y0, win_w, win_h, tile_size, tile_size)
                    if norm is not None:
                        label_lines.append(make_yolo_line(norm))

                try:
                    rgb = src.read(indexes=[r_band, g_band, b_band], window=window)
                except Exception as e:
                    logger.warning(f"    Read failed ({row},{col}): {e} — skipping")
                    continue

                if rgb.size == 0:
                    continue
                if rgb.dtype == np.uint16:
                    rgb = (rgb / 256).astype(np.uint8)
                elif rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)

                records.append({
                    "rgb_chw": rgb,
                    "label_lines": label_lines,
                    "transform": tile_transform,
                    "win_w": win_w,
                    "win_h": win_h,
                    "crs": crs,
                    "source": ortho_path.name,
                })
    return records


def subsample_empty_tiles(records: list, max_empty_ratio: float, rng: random.Random) -> list:
    annotated = [r for r in records if r["label_lines"]]
    background = [r for r in records if not r["label_lines"]]
    if not annotated:
        logger.warning("    No annotated tiles in this batch — keeping all")
        return records
    max_bg = int(len(annotated) * max_empty_ratio / max(1e-9, 1 - max_empty_ratio))
    if len(background) <= max_bg:
        return records
    kept = rng.sample(background, max_bg)
    logger.info(f"    Subsampled background: {len(background)} → {len(kept)} (ratio={max_empty_ratio})")
    return annotated + kept


def write_tiles(records: list, split: str, output_dir: Path, tile_index: TileIndex, counter: list, tile_size: int) -> None:
    from PIL import Image as PILImage
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    for rec in records:
        tile_name = f"duke_tile_{counter[0]:06d}.png"
        counter[0] += 1
        rgb_mapped = tone_map_naip_clahe(rec["rgb_chw"], clip_limit=0.03)
        _, win_h, win_w = rgb_mapped.shape
        if win_w < tile_size or win_h < tile_size:
            padded = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            padded[:, :win_h, :win_w] = rgb_mapped
            rgb_mapped = padded
        PILImage.fromarray(np.transpose(rgb_mapped, (1, 2, 0)), mode="RGB").save(images_dir / tile_name)
        (labels_dir / tile_name.replace(".png", ".txt")).write_text("\n".join(rec["label_lines"]))
        t = rec["transform"]
        tile_index.add_tile(
            tile_name=tile_name, transform=t, crs=rec["crs"],
            width=rec["win_w"], height=rec["win_h"], source_file=rec["source"],
            bounds={"minx": t.c, "miny": t.f + rec["win_h"] * t.e, "maxx": t.c + rec["win_w"] * t.a, "maxy": t.f},
        )


def write_data_yaml(output_dir: Path) -> None:
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump({"path": str(output_dir.resolve()), "train": "images/train", "val": "images/val",
                   "test": "images/test", "nc": 1, "names": {0: "solar_array"}},
                  f, default_flow_style=False, sort_keys=False)


def split_images(stems: list, ratios: tuple, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n = len(shuffled)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return {"train": shuffled[:i1], "val": shuffled[i1:i2], "test": shuffled[i2:]}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--meta-csv",   type=Path, default=repo/"data/raw/duke/polygonDataExceptVertices.csv")
    p.add_argument("--ll-csv",     type=Path, default=repo/"data/raw/duke/polygonVertices_LatitudeLongitude.csv")
    p.add_argument("--ortho-dir",  type=Path, default=repo/"data/raw/duke/ortho")
    p.add_argument("--output-dir", type=Path, default=repo/"data/yolo/duke")
    p.add_argument("--tile-index", type=Path, default=repo/"data/interim/duke_tile_index.json")
    p.add_argument("--tile-size",  type=int,  default=640)
    p.add_argument("--split",      type=float, nargs=3, metavar=("TR", "VA", "TE"), default=[0.70, 0.15, 0.15])
    p.add_argument("--max-empty-ratio", type=float, default=0.30)
    p.add_argument("--min-polygons", type=int, default=0)
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()
    global TILE_SIZE
    TILE_SIZE = args.tile_size

    for path, name in [(args.meta_csv, "--meta-csv"), (args.ll_csv, "--ll-csv"), (args.ortho_dir, "--ortho-dir")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    total = sum(args.split)
    ratios = tuple(r / total for r in args.split)

    for split in SPLITS:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    args.tile_index.parent.mkdir(parents=True, exist_ok=True)

    _meta, ll = load_annotations(args.meta_csv, args.ll_csv)
    polygon_map = build_polygon_map(ll)

    ortho_files = {}
    for ext in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        for f in sorted(args.ortho_dir.glob(ext)):
            ortho_files[f.stem] = f

    matched = [s for s in ortho_files if s in polygon_map]
    no_image = [s for s in polygon_map if s not in ortho_files]
    logger.info(f"Ortho files found: {len(ortho_files)} | Matched: {len(matched)} | No ortho: {len(no_image)}")

    if args.min_polygons > 0:
        matched = [s for s in matched if len(polygon_map[s]) >= args.min_polygons]
        logger.info(f"After --min-polygons={args.min_polygons}: {len(matched)} images")

    if not matched:
        raise RuntimeError("No matched images. Download imagery first:\n  python scripts/02e_download_duke_dataset.py --imagery-only")

    assignment = split_images(matched, ratios, args.seed)
    for sn, stems in assignment.items():
        logger.info(f"  {sn}: {len(stems)} source images")

    tile_index = TileIndex()
    counter = [0]
    rng = random.Random(args.seed)
    split_stats: dict = {}

    for split_name in SPLITS:
        stems = assignment[split_name]
        logger.info(f"\nProcessing '{split_name}' ({len(stems)} images)...")
        all_records = []
        for stem in stems:
            ortho_path = ortho_files[stem]
            polygons = polygon_map[stem]
            logger.info(f"  {stem}: {len(polygons)} polygons")
            try:
                recs = process_image(ortho_path, polygons, args.tile_size)
                all_records.extend(recs)
            except Exception as e:
                logger.error(f"  Failed {stem}: {e}")

        all_records = subsample_empty_tiles(all_records, args.max_empty_ratio, rng)
        annotated = sum(1 for r in all_records if r["label_lines"])
        background = len(all_records) - annotated
        logger.info(f"  Writing {len(all_records)} tiles ({annotated} annotated, {background} background)")
        write_tiles(all_records, split_name, args.output_dir, tile_index, counter, args.tile_size)
        split_stats[split_name] = {"tiles": len(all_records), "annotated": annotated, "background": background}

    tile_index.save(args.tile_index)
    write_data_yaml(args.output_dir)

    for sn, s in split_stats.items():
        er = s["background"] / s["tiles"] if s["tiles"] else 0
        logger.info(f"  {sn:6s}: {s['tiles']:5d} tiles ({s['annotated']} annotated, empty_ratio={er:.2f})")
    logger.info(f"  Total: {counter[0]} tiles → {args.output_dir}")


if __name__ == "__main__":
    main()
