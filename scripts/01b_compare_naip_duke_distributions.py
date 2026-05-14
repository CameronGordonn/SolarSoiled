#!/usr/bin/env python3
"""NAIP/Duke distributional equivalence baseline.

Tyler's first asked-for measurement: are NAIP and Duke really the same kind of
imagery, or is the joint training trying to bridge a gap that no amount of
oversampling will close? This script computes per-tile image stats (brightness,
sharpness, RGB mean/std) and per-polygon stats (area_px, area_m2 after GSD
correction, aspect, polys-per-tile) for both domains, then writes:

  outputs/eval/domain_equivalence/distributions.csv
  outputs/eval/domain_equivalence/report.md
  outputs/eval/domain_equivalence/hist_<feature>.png
  outputs/eval/domain_equivalence/manifest.json

Verdict in report.md is one paragraph: do polygons in m² overlap after the
~4× GSD correction? If yes, the existing `02d`/`02f` upscale-at-train logic is
right and the Duke ramp is well-founded. If no, joint training has a domain
gap problem the ramp can't fix.

Usage: python scripts/01b_compare_naip_duke_distributions.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest


# Effective GSD per domain — used to convert pixel area → m².
# NAIP Santa Cruz tiles are 0.6 m/px; Duke Bradbury 160px chips are ~0.3 m/px
# in source, but training upscales them 4× so they enter the model at the
# equivalent of ~0.075 m/px. For domain-equivalence purposes (image-space
# comparison vs ground-truth-space comparison), report both pixel-area
# distributions and m²-area distributions using *source* GSD.
DOMAIN_GSD_M = {"naip": 0.6, "duke": 0.3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--naip-root", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip")
    p.add_argument("--duke-root", type=Path, default=REPO_ROOT / "data" / "yolo" / "duke_160")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "eval" / "domain_equivalence")
    p.add_argument("--limit-per-split", type=int, default=None,
                   help="Sample only N tiles per split per domain (smoke test).")
    return p.parse_args()


def collect_tile_paths(domain_root: Path, splits: Iterable[str]) -> list[tuple[str, Path, Path]]:
    """Return list of (split, image_path, label_path)."""
    out = []
    for split in splits:
        img_dir = domain_root / "images" / split
        lbl_dir = domain_root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            out.append((split, img_path, label_path))
    return out


def image_stats(img_path: Path) -> dict:
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    gray = img.mean(axis=2)
    # Laplacian-variance proxy via 3x3 kernel without scipy:
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {
        "img_w": w, "img_h": h,
        "brightness": float(gray.mean()),
        "rgb_r_mean": float(img[..., 0].mean()),
        "rgb_g_mean": float(img[..., 1].mean()),
        "rgb_b_mean": float(img[..., 2].mean()),
        "rgb_r_std": float(img[..., 0].std()),
        "rgb_g_std": float(img[..., 1].std()),
        "rgb_b_std": float(img[..., 2].std()),
        "sharpness": float(lap.var()),
    }


def polygon_stats(label_path: Path, img_w: int, img_h: int, gsd_m: float) -> list[dict]:
    if not label_path.exists():
        return []
    out: list[dict] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        xs = np.array(coords[0::2], dtype=np.float64) * img_w
        ys = np.array(coords[1::2], dtype=np.float64) * img_h
        if len(xs) < 3:
            continue
        # Shoelace area in pixels
        area_px = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        bbox_w = xs.max() - xs.min()
        bbox_h = ys.max() - ys.min()
        aspect = (max(bbox_w, bbox_h) / min(bbox_w, bbox_h)) if min(bbox_w, bbox_h) > 0 else float("nan")
        out.append({
            "area_px": area_px,
            "area_m2": area_px * (gsd_m ** 2),
            "bbox_w_px": bbox_w,
            "bbox_h_px": bbox_h,
            "aspect": aspect,
            "n_vertices": len(xs),
        })
    return out


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic + asymptotic p-value (no scipy)."""
    a = np.sort(a[~np.isnan(a)])
    b = np.sort(b[~np.isnan(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    all_vals = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = len(a) * len(b) / (len(a) + len(b))
    # Kolmogorov asymptotic p-value
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1) ** (j - 1) * np.exp(-2.0 * (lam ** 2) * (j ** 2))))
    p = max(0.0, min(1.0, p))
    return d, p


def percentiles(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan"),
                "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def render_histograms(out_dir: Path, naip_df: dict, duke_df: dict, features: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping histograms", file=sys.stderr)
        return
    for feat in features:
        a = np.asarray(naip_df.get(feat, []), dtype=np.float64)
        b = np.asarray(duke_df.get(feat, []), dtype=np.float64)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) == 0 and len(b) == 0:
            continue
        # Shared bins. Log-x for area features (heavy right tail).
        log_x = feat.startswith("area_")
        if log_x:
            a = a[a > 0]; b = b[b > 0]
            if len(a) == 0 or len(b) == 0:
                continue
            lo = float(min(a.min(), b.min()))
            hi = float(max(a.max(), b.max()))
            bins = np.geomspace(max(lo, 1e-3), hi, 50)
        else:
            lo = float(min(a.min() if len(a) else 0, b.min() if len(b) else 0))
            hi = float(max(a.max() if len(a) else 1, b.max() if len(b) else 1))
            bins = np.linspace(lo, hi, 50)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(a, bins=bins, alpha=0.55, label=f"NAIP (n={len(a)})", color="#1f77b4", density=True)
        ax.hist(b, bins=bins, alpha=0.55, label=f"Duke (n={len(b)})", color="#d62728", density=True)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(feat)
        ax.set_ylabel("density")
        ax.set_title(f"NAIP vs Duke — {feat}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{feat}.png", dpi=120)
        plt.close(fig)


def write_report(out_dir: Path, naip_summary: dict, duke_summary: dict, ks: dict) -> None:
    lines = ["# NAIP vs Duke equivalence report", ""]
    lines.append("Generated by `scripts/01b_compare_naip_duke_distributions.py`. The point of "
                 "this report is one question: are NAIP and Duke similar enough that a joint model "
                 "can transfer between them, or are we trying to bridge an unbridgeable gap?")
    lines.append("")
    # Verdict
    area_m2_ks = ks.get("area_m2", (float("nan"), float("nan")))
    naip_med = naip_summary.get("area_m2", {}).get("median", float("nan"))
    duke_med = duke_summary.get("area_m2", {}).get("median", float("nan"))
    if not np.isnan(naip_med) and not np.isnan(duke_med):
        ratio = duke_med / naip_med if naip_med > 0 else float("nan")
        verdict = ("**Verdict (preliminary):** Duke median panel area in m² is "
                   f"{duke_med:.1f} m² vs NAIP {naip_med:.1f} m² (ratio {ratio:.2f}×). "
                   f"KS statistic on area_m2 = {area_m2_ks[0]:.3f} (p≈{area_m2_ks[1]:.3g}). ")
        if abs(np.log(ratio)) < np.log(2.0):
            verdict += ("The two domains overlap within 2× in real-world panel size, "
                        "so the existing 4× upscale-at-train logic is well-grounded "
                        "and the Duke ramp is the right tool. Domain-mismatch is not the "
                        "primary failure mode.")
        else:
            verdict += ("Real-world panel sizes differ by more than 2×, suggesting a "
                        "labeling-convention or scene-type gap that no upscale or oversample "
                        "ratio can paper over. Investigate before continuing the ramp.")
        lines.append(verdict)
        lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append("| feature | NAIP n | NAIP median (p25–p75) | Duke n | Duke median (p25–p75) | KS D | KS p |")
    lines.append("|---|---:|---|---:|---|---:|---:|")
    for feat in sorted(set(naip_summary.keys()) | set(duke_summary.keys())):
        ns = naip_summary.get(feat, {})
        ds = duke_summary.get(feat, {})
        d, p = ks.get(feat, (float("nan"), float("nan")))
        lines.append(
            f"| `{feat}` | {ns.get('n', 0)} "
            f"| {ns.get('median', float('nan')):.3g} "
            f"({ns.get('p25', float('nan')):.3g}–{ns.get('p75', float('nan')):.3g}) "
            f"| {ds.get('n', 0)} "
            f"| {ds.get('median', float('nan')):.3g} "
            f"({ds.get('p25', float('nan')):.3g}–{ds.get('p75', float('nan')):.3g}) "
            f"| {d:.3f} | {p:.3g} |"
        )
    lines.append("")
    lines.append("Histograms: `hist_<feature>.png` in this directory. "
                 "Polygon area features use a shared log-binning; image features use linear bins. "
                 f"NAIP source GSD assumed {DOMAIN_GSD_M['naip']} m/px; Duke {DOMAIN_GSD_M['duke']} m/px.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    args = parse_args() if argv is None else parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict] = []
    poly_features = {"naip": {}, "duke": {}}
    img_features = {"naip": {}, "duke": {}}

    domain_roots = {"naip": args.naip_root, "duke": args.duke_root}
    for domain, root in domain_roots.items():
        gsd = DOMAIN_GSD_M[domain]
        tiles = collect_tile_paths(root, args.splits)
        if args.limit_per_split:
            # cap per split
            from collections import defaultdict
            by_split: dict[str, list] = defaultdict(list)
            for t in tiles:
                by_split[t[0]].append(t)
            tiles = []
            for s, ts in by_split.items():
                tiles.extend(ts[: args.limit_per_split])
        print(f"{domain}: scanning {len(tiles)} tiles")
        for split, img_path, label_path in tiles:
            try:
                istats = image_stats(img_path)
            except Exception as e:
                print(f"  skip {img_path.name}: {e}", file=sys.stderr)
                continue
            polys = polygon_stats(label_path, istats["img_w"], istats["img_h"], gsd)
            row = {"domain": domain, "split": split, "tile_id": img_path.name,
                   "n_polys": len(polys), **istats}
            csv_rows.append(row)
            for k in ("brightness", "sharpness", "rgb_r_mean", "rgb_g_mean", "rgb_b_mean"):
                img_features[domain].setdefault(k, []).append(istats[k])
            img_features[domain].setdefault("n_polys", []).append(len(polys))
            for p in polys:
                for k, v in p.items():
                    poly_features[domain].setdefault(k, []).append(v)

    # Distributions CSV
    csv_path = args.out_dir / "distributions.csv"
    with csv_path.open("w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # Aggregate stats
    naip_summary: dict[str, dict] = {}
    duke_summary: dict[str, dict] = {}
    ks: dict[str, tuple[float, float]] = {}
    feature_set = sorted(set(poly_features["naip"]) | set(poly_features["duke"])
                         | set(img_features["naip"]) | set(img_features["duke"]))
    for feat in feature_set:
        nv = np.asarray(poly_features["naip"].get(feat, img_features["naip"].get(feat, [])),
                        dtype=np.float64)
        dv = np.asarray(poly_features["duke"].get(feat, img_features["duke"].get(feat, [])),
                        dtype=np.float64)
        naip_summary[feat] = percentiles(nv)
        duke_summary[feat] = percentiles(dv)
        ks[feat] = ks_2samp(nv, dv)

    # Histograms
    render_histograms(
        args.out_dir,
        {**poly_features["naip"], **img_features["naip"]},
        {**poly_features["duke"], **img_features["duke"]},
        ["brightness", "sharpness", "n_polys", "area_px", "area_m2", "aspect"],
    )

    # Report
    write_report(args.out_dir, naip_summary, duke_summary, ks)
    print(f"Wrote {args.out_dir / 'report.md'}")

    # Manifest
    write_manifest(
        args.out_dir,
        stage="eval",
        model_version="domain_equivalence_v1",
        inputs=[str(args.naip_root), str(args.duke_root)],
        metrics={
            "naip_n_tiles": int(sum(1 for r in csv_rows if r["domain"] == "naip")),
            "duke_n_tiles": int(sum(1 for r in csv_rows if r["domain"] == "duke")),
            "naip_n_polys": int(sum(r["n_polys"] for r in csv_rows if r["domain"] == "naip")),
            "duke_n_polys": int(sum(r["n_polys"] for r in csv_rows if r["domain"] == "duke")),
            "ks_area_m2": float(ks.get("area_m2", (float("nan"),))[0]),
        },
        known_limitations=[
            "GSD values hardcoded (NAIP=0.6m, Duke=0.3m); per-tile GSD not yet read from tile_index",
            "Sharpness uses a 3x3 Laplacian-variance proxy, not a true 5x5 Laplacian",
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
