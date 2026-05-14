#!/usr/bin/env python3
"""Audit YOLO segmentation dataset quality before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO segmentation dataset quality")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--rules", type=str, default="configs/yolo/dataset_audit.yaml")
    parser.add_argument("--out", type=str, default="outputs/eval/dataset_audit_report.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_dataset_root(data_yaml: Path, data_cfg: dict) -> Path:
    root = Path(data_cfg.get("path", data_yaml.parent))
    return (root if root.is_absolute() else (data_yaml.parent / root).resolve()).expanduser().resolve()


def list_images(d: Path) -> List[Path]:
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS] if d.exists() else []


def list_labels(d: Path) -> List[Path]:
    return [p for p in d.rglob("*.txt") if p.is_file()] if d.exists() else []


def parse_label_file(label_path: Path) -> Tuple[int, Dict[int, int], List[str]]:
    line_count = 0
    class_counts: Dict[int, int] = {}
    issues: List[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return line_count, class_counts, issues
    for idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 7:
            issues.append(f"{label_path}: line {idx} too few tokens ({len(parts)})")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"{label_path}: line {idx} non-numeric values")
            continue
        if len(coords) % 2 != 0:
            issues.append(f"{label_path}: line {idx} odd coordinate count")
            continue
        if len(coords) < 6:
            issues.append(f"{label_path}: line {idx} fewer than 3 polygon points")
            continue
        for value in coords:
            if value < 0.0 or value > 1.0:
                issues.append(f"{label_path}: line {idx} coord {value} outside [0,1]")
                break
        class_counts[cls] = class_counts.get(cls, 0) + 1
        line_count += 1
    return line_count, class_counts, issues


def split_stats(dataset_root: Path, split: str) -> dict:
    images = list_images(dataset_root / "images" / split)
    labels = list_labels(dataset_root / "labels" / split)
    label_stems: Set[str] = {p.stem for p in labels}
    image_stems: Set[str] = {p.stem for p in images}

    empty_labels = 0
    total_objects = 0
    class_counts: Dict[int, int] = {}
    format_issues: List[str] = []

    for lp in labels:
        count, fc, issues = parse_label_file(lp)
        if count == 0:
            empty_labels += 1
        total_objects += count
        for cls, val in fc.items():
            class_counts[cls] = class_counts.get(cls, 0) + val
        format_issues.extend(issues)

    return {
        "split": split, "images": len(images), "labels": len(labels),
        "objects": total_objects, "empty_labels": empty_labels,
        "empty_label_ratio": (empty_labels / len(labels)) if labels else 0.0,
        "missing_labels": sorted(image_stems - label_stems),
        "orphan_labels": sorted(label_stems - image_stems),
        "class_counts": dict(sorted(class_counts.items())),
        "format_issues": format_issues, "image_stems": image_stems,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_yaml = Path(args.data).expanduser().resolve() if args.data else repo_root / "data" / "yolo" / "naip" / "data.yaml"
    rules_yaml = Path(args.rules).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    if not rules_yaml.exists():
        raise FileNotFoundError(f"Audit rules not found: {rules_yaml}")

    data_cfg = load_yaml(data_yaml)
    rules_cfg = load_yaml(rules_yaml).get("audit_rules", {})
    dataset_root = get_dataset_root(data_yaml, data_cfg)
    splits = ["train", "val", "test"]

    per_split = {s: split_stats(dataset_root, s) for s in splits}
    warnings: List[str] = []

    min_images = rules_cfg.get("min_images_per_split", {})
    max_empty_ratio = float(rules_cfg.get("max_empty_label_ratio", 1.0))

    for split in splits:
        stats = per_split[split]
        if stats["images"] < int(min_images.get(split, 0)):
            warnings.append(f"{split}: images {stats['images']} below minimum {min_images.get(split, 0)}")
        if stats["empty_label_ratio"] > max_empty_ratio:
            warnings.append(f"{split}: empty label ratio {stats['empty_label_ratio']:.3f} > {max_empty_ratio:.3f}")
        if stats["missing_labels"]:
            warnings.append(f"{split}: {len(stats['missing_labels'])} images missing labels")
        if stats["orphan_labels"]:
            warnings.append(f"{split}: {len(stats['orphan_labels'])} labels missing images")
        if stats["format_issues"]:
            warnings.append(f"{split}: {len(stats['format_issues'])} label format issues")

    if rules_cfg.get("check_split_leakage", True):
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = sorted(per_split[a]["image_stems"] & per_split[b]["image_stems"])
            if overlap:
                warnings.append(f"split leakage: {a}/{b} overlap count={len(overlap)}")

    if rules_cfg.get("check_class_balance", True):
        if sum(1 for s in splits if per_split[s]["objects"] > 0) < 2:
            warnings.append("class balance: fewer than two splits contain labeled objects")

    for split in splits:
        per_split[split].pop("image_stems", None)

    report = {
        "data_yaml": str(data_yaml), "dataset_root": str(dataset_root), "rules_yaml": str(rules_yaml),
        "per_split": per_split,
        "class_distribution": {s: per_split[s]["class_counts"] for s in splits},
        "warnings": warnings, "status": "ok" if not warnings else "warning",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit report: {out_path}")
    for w in warnings:
        print(f"  WARNING: {w}")
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
