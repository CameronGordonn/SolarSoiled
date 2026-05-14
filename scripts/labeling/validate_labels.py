"""Validate YOLO label files in a dataset: coordinate range, size distribution, empty ratio."""

import argparse
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LARGE_OBJ_THRESHOLD = 0.85  # flag objects wider/taller than this fraction of image


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def load_data_yaml(data_path: Path) -> tuple[Path, list]:
    with open(data_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root, cfg


def scan_split(dataset_root: Path, split: str) -> dict:
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split
    if not labels_dir.exists():
        return {"split": split, "skipped": True}

    label_files = sorted(labels_dir.glob("*.txt"))
    n_images = len([p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0

    violations = []   # (file, line_num, coord_value)
    large_objs = []   # (file, width_or_height)
    sizes_w = []
    sizes_h = []
    empty = 0
    total_objects = 0

    for lf in label_files:
        text = lf.read_text().strip()
        if not text:
            empty += 1
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            empty += 1
            continue

        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                continue

            for v in coords:
                if v < 0.0 or v > 1.0:
                    violations.append((lf.name, i, v))

            xs = coords[0::2]
            ys = coords[1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            sizes_w.append(w)
            sizes_h.append(h)
            total_objects += 1

            if w > LARGE_OBJ_THRESHOLD or h > LARGE_OBJ_THRESHOLD:
                large_objs.append((lf.name, max(w, h)))

    return {
        "split": split,
        "skipped": False,
        "n_label_files": len(label_files),
        "n_images": n_images,
        "total_objects": total_objects,
        "empty_labels": empty,
        "violations": violations,
        "large_objects": large_objs,
        "sizes_w": sizes_w,
        "sizes_h": sizes_h,
    }


def print_stats(result: dict) -> bool:
    """Print per-split stats. Returns True if clean, False if violations found."""
    split = result["split"]
    if result.get("skipped"):
        print(f"  {split}: not found — skipping")
        return True

    n = result["n_label_files"]
    imgs = result["n_images"]
    objs = result["total_objects"]
    empty = result["empty_labels"]
    violations = result["violations"]
    large = result["large_objects"]

    import statistics
    sw = result["sizes_w"]
    sh = result["sizes_h"]
    med_w = statistics.median(sw) if sw else 0
    med_h = statistics.median(sh) if sh else 0
    p95_w = sorted(sw)[int(len(sw) * 0.95)] if sw else 0
    p95_h = sorted(sh)[int(len(sh) * 0.95)] if sh else 0

    ok = len(violations) == 0
    status = "OK" if ok else "FAIL"
    warn = " [LARGE OBJ WARNING]" if large else ""
    empty_ratio = empty / n if n else 0

    print(f"  {split:6s}: {imgs:4d} images | {n:4d} labels | {objs:5d} objects | "
          f"empty={empty}/{n} ({empty_ratio:.1%}) | status={status}{warn}")
    print(f"          size median W×H: {med_w:.3f}×{med_h:.3f}  p95: {p95_w:.3f}×{p95_h:.3f}")

    if violations:
        print(f"  !! {len(violations)} coordinate violation(s):")
        for fname, lineno, val in violations[:10]:
            print(f"       {fname} line {lineno}: {val:.6f}")
        if len(violations) > 10:
            print(f"       ... and {len(violations)-10} more")

    if large:
        print(f"  ?? {len(large)} suspiciously large object(s) (>{LARGE_OBJ_THRESHOLD:.0%} of image dim):")
        for fname, size in large[:5]:
            print(f"       {fname}: {size:.3f} — may be edge-tile normalization artifact")
        if len(large) > 5:
            print(f"       ... and {len(large)-5} more")

    return ok


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"ERROR: data.yaml not found: {data_path}")
        sys.exit(1)

    dataset_root, _cfg = load_data_yaml(data_path)
    print(f"Dataset: {dataset_root}")

    all_ok = True
    for split in args.splits:
        result = scan_split(dataset_root, split)
        ok = print_stats(result)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("PASS — all label coordinates valid")
    else:
        print("FAIL — fix coordinate violations before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
