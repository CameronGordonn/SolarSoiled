#!/usr/bin/env python3
"""Audit YOLO dataset tiles by (region, vintage) parsed from source NAIP filenames.

Outputs:
  outputs/eval/vintage_stratified_metrics.csv  (per-split, per-stratum tile + object counts)
  console summary table

Bridges Roboflow-mangled training filenames (`tile_NNNNNN_png.rf.HASH.png`) back to
tile_index.json keys via strip_roboflow_suffix().
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.tile_metadata import TileIndex, parse_source_filename, strip_roboflow_suffix


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip",
                   help="YOLO dataset root containing images/{train,val,test} and labels/{...}")
    p.add_argument("--tile-index", type=Path, default=REPO_ROOT / "data" / "interim" / "tile_index.json")
    p.add_argument("--output-csv", type=Path, default=REPO_ROOT / "outputs" / "eval" / "vintage_stratified_metrics.csv")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def count_label_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def stratify_split(images_dir: Path, labels_dir: Path, tile_index: TileIndex) -> dict:
    """Walk images_dir, return dict of {(region, vintage): {tile_count, object_count, unmatched, sample_sources}}."""
    rows: dict[tuple, dict] = defaultdict(lambda: {"tiles": 0, "objects": 0, "unmatched": 0, "sources": Counter()})

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        canonical, entry = tile_index.lookup_by_stripped_name(img_path.name)
        if entry is None:
            rows[("unknown", None)]["tiles"] += 1
            rows[("unknown", None)]["unmatched"] += 1
            continue
        region, vintage = parse_source_filename(entry.get("source"))
        key = (region, vintage)
        rows[key]["tiles"] += 1
        rows[key]["objects"] += count_label_objects(labels_dir / (img_path.stem + ".txt"))
        if entry.get("source"):
            rows[key]["sources"][entry["source"]] += 1
    return rows


def main() -> int:
    args = parse_args()

    if not args.tile_index.exists():
        print(f"ERROR: tile index not found: {args.tile_index}", file=sys.stderr)
        return 1
    if not args.data_root.exists():
        print(f"ERROR: data root not found: {args.data_root}", file=sys.stderr)
        return 1

    tile_index = TileIndex(tile_index_path=args.tile_index)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_for_csv: list[dict] = []

    print(f"Auditing {args.data_root} against {args.tile_index} ({len(tile_index.data)} tiles indexed)\n")
    for split in args.splits:
        images_dir = args.data_root / "images" / split
        labels_dir = args.data_root / "labels" / split
        if not images_dir.exists():
            print(f"  [skip] {split}: {images_dir} does not exist")
            continue
        strata = stratify_split(images_dir, labels_dir, tile_index)
        total_tiles = sum(s["tiles"] for s in strata.values())
        total_objs = sum(s["objects"] for s in strata.values())
        print(f"  {split}: {total_tiles} tiles, {total_objs} objects across {len(strata)} stratum(s)")

        for (region, vintage), stats in sorted(strata.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
            label_region = region or "unknown"
            label_vintage = vintage if vintage is not None else ""
            row = {
                "split": split,
                "region": label_region,
                "vintage": label_vintage,
                "tiles": stats["tiles"],
                "objects": stats["objects"],
                "unmatched_to_tile_index": stats["unmatched"],
                "tile_pct_of_split": round(100 * stats["tiles"] / total_tiles, 2) if total_tiles else 0.0,
                "objs_per_tile": round(stats["objects"] / stats["tiles"], 2) if stats["tiles"] else 0.0,
            }
            rows_for_csv.append(row)
            print(
                f"    {label_region!r:<10} {str(label_vintage):<6} "
                f"tiles={stats['tiles']:<5} objects={stats['objects']:<5} "
                f"objs/tile={row['objs_per_tile']:.2f}"
                + (f" UNMATCHED={stats['unmatched']}" if stats["unmatched"] else "")
            )
        print()

    if not rows_for_csv:
        print("No tiles found in any split — nothing to write.", file=sys.stderr)
        return 1

    fieldnames = list(rows_for_csv[0].keys())
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"Wrote {len(rows_for_csv)} rows to {args.output_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
