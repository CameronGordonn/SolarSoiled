"""Merge Duke per-panel YOLO polygons into installation-level polygons.

Duke labels each individual solar panel (~1.7m²). For the task of locating
solar installations (not counting panels), we want one polygon per rooftop
array — matching NAIP's whole-array label convention.

Strategy:
  1. Cluster panel centroids within each tile using DBSCAN
  2. Replace each cluster with the convex hull of all its panels
  3. Write one YOLO polygon line per installation

Output: data/yolo/duke_160_merged/ (images symlinked, labels rewritten)

Usage:
    PYTHONPATH=. python scripts/02h_merge_duke_labels.py
    PYTHONPATH=. python scripts/02h_merge_duke_labels.py --eps 25 --inspect
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

try:
    from shapely.geometry import MultiPoint
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("WARNING: shapely not found — using bounding-box fallback. "
          "Install with: pip install shapely")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    raise SystemExit("scikit-learn required: pip install scikit-learn")

TILE_PX = 160   # Duke chip size


def parse_polygons(label_file: Path) -> list[np.ndarray]:
    """Return list of (N,2) pixel-coord arrays, one per polygon."""
    polys = []
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        coords[:, 0] *= TILE_PX
        coords[:, 1] *= TILE_PX
        polys.append(coords)
    return polys


def convex_hull_yolo(polygons: list[np.ndarray]) -> str:
    """Merge polygons → convex hull → YOLO polygon string (class 0)."""
    all_pts = np.vstack(polygons)
    if HAS_SHAPELY:
        geom = MultiPoint(all_pts).convex_hull
        if geom.geom_type == "Point":
            pts = [(geom.x, geom.y)] * 3
        elif geom.geom_type == "LineString":
            pts = list(zip(*geom.xy))
        else:
            xs, ys = geom.exterior.xy
            pts = list(zip(xs[:-1], ys[:-1]))  # drop repeated closing vertex
    else:
        x0, y0 = all_pts.min(axis=0)
        x1, y1 = all_pts.max(axis=0)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    norm = [(x / TILE_PX, y / TILE_PX) for x, y in pts]
    flat = " ".join(f"{v:.6f}" for xy in norm for v in xy)
    return f"0 {flat}"


def merge_label_file(src: Path, dst: Path, eps_px: float) -> tuple[int, int]:
    """Merge one label file. Returns (n_panels_in, n_installations_out)."""
    polys = parse_polygons(src)
    if not polys:
        dst.write_text("")
        return 0, 0

    centroids = np.array([p.mean(axis=0) for p in polys])
    cluster_ids = DBSCAN(eps=eps_px, min_samples=1).fit_predict(centroids)

    merged = []
    for cid in np.unique(cluster_ids):
        cluster = [polys[i] for i in range(len(polys)) if cluster_ids[i] == cid]
        merged.append(convex_hull_yolo(cluster))

    dst.write_text("\n".join(merged) + "\n")
    return len(polys), len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=20.0,
                        help="DBSCAN eps in pixels. Panels within this distance "
                             "are merged into one installation. Default 20px ≈ 6–12m "
                             "depending on Duke resolution. Increase if adjacent "
                             "rooftops are being merged; decrease if one array is "
                             "split into multiple clusters.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print per-file merge stats for the first 20 files "
                             "in train/ so you can sanity-check the eps value.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    src_root = repo / "data" / "yolo" / "duke_160"
    dst_root = repo / "data" / "yolo" / "duke_160_merged"

    if not src_root.exists():
        raise FileNotFoundError(f"Duke dataset not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_img = src_root / "images" / split
        src_lbl = src_root / "labels" / split
        if not src_img.exists():
            continue

        dst_img = dst_root / "images" / split
        dst_lbl = dst_root / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # Symlink images (no need to copy — saves disk space)
        for img in src_img.glob("*"):
            link = dst_img / img.name
            if not link.exists():
                link.symlink_to(img.resolve())

        total_in = total_out = n_files = 0
        inspect_count = 0

        for lbl_file in sorted(src_lbl.glob("*.txt")):
            n_in, n_out = merge_label_file(lbl_file, dst_lbl / lbl_file.name, args.eps)
            total_in += n_in
            total_out += n_out
            n_files += 1

            if args.inspect and inspect_count < 20 and n_in > 1:
                print(f"  {lbl_file.name}: {n_in} panels → {n_out} installations")
                inspect_count += 1

        ratio = total_in / total_out if total_out else 0
        print(f"{split:6s}: {n_files} tiles, {total_in} panels → "
              f"{total_out} installations  (avg {ratio:.1f} panels/array)")

    # Write data.yaml
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": str(dst_root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc": 1,
            "names": {0: "solar_array"},
        }, f)
    print(f"\nCreated: {yaml_path}")
    print(f"Inspect results: open data/yolo/duke_160_merged/labels/train/ "
          f"and compare a few tiles against data/yolo/duke_160/labels/train/")
    print(f"If adjacent rooftops are being merged, re-run with --eps 12")
    print(f"If one array is split into multiple polygons, re-run with --eps 30")


if __name__ == "__main__":
    main()
