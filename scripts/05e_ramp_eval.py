#!/usr/bin/env python3
"""Per-step Duke-ramp evaluation: NAIP test mAP50, Duke test mAP50, RCA snapshot.

Called once after each `03_train_yolov8_seg.py` ramp run. Evaluates both
data.yamls (NAIP test, Duke test), runs the per-detection RCA on NAIP test
for a failure-mode snapshot, and appends one row to `outputs/eval/ramp_curve.csv`
so the ramp's regression curve is visible at a glance.

The hard stop-rule from the ramp config: if `naip_regression_delta > 0.07`,
print a HALT line so the runbook script (or the human running it) knows the
previous step's weights are the production candidate. NAIP-test precision is
also reported separately because mAP50 alone can miss over-prediction failures
(high recall, very low precision).

Usage:
  python scripts/05e_ramp_eval.py --run R1 \\
      --weights runs/segment/<train-run>/weights/best.pt \\
      --ramp-config configs/yolo/experiments_joint_v2_ramp.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.solarsoiled.manifest import write_manifest
from src.utils.rca import run_rca_pass, summarize_rca
from src.utils.train_utils import compute_f1, resolve_weights, select_device


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Ramp step name (R0, R1, ...)")
    p.add_argument("--weights", required=True, help="Path to weights from this ramp step")
    p.add_argument("--ramp-config", type=Path,
                   default=REPO_ROOT / "configs" / "yolo" / "experiments_joint_v2_ramp.yaml")
    p.add_argument("--naip-data", type=Path,
                   default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml")
    p.add_argument("--duke-data", type=Path,
                   default=REPO_ROOT / "data" / "yolo" / "duke_160" / "data.yaml")
    p.add_argument("--curve-csv", type=Path,
                   default=REPO_ROOT / "outputs" / "eval" / "ramp_curve.csv")
    p.add_argument("--curve-png", type=Path,
                   default=REPO_ROOT / "outputs" / "eval" / "ramp_curve.png")
    p.add_argument("--skip-rca", action="store_true",
                   help="Skip the 05c RCA pass (saves time during smoke tests)")
    p.add_argument("--limit", type=int, default=None,
                   help="Pass through to 05c (smoke testing)")
    return p.parse_args(argv)


def run_ultralytics_val(weights: Path, data_yaml: Path) -> dict:
    """Run model.val() on the test split and return a metrics dict."""
    from ultralytics import YOLO
    device = select_device()
    model = YOLO(str(weights), task="segment")
    metrics = model.val(data=str(data_yaml), split="test", device=device, verbose=False)
    sp, sr = float(metrics.seg.mp), float(metrics.seg.mr)
    return {
        "seg": {
            "map50": float(metrics.seg.map50),
            "map50_95": float(metrics.seg.map),
            "precision": sp,
            "recall": sr,
            "f1": compute_f1(sp, sr),
        },
        "box": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }


def run_rca(weights: Path, data_yaml: Path, run_name: str, limit: int | None) -> dict:
    """Run SAHI RCA on NAIP test split and return failure-mode summary."""
    rca_dir = REPO_ROOT / "outputs" / "eval" / run_name / "rca_naip_test"
    rca_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_rca_pass(
        weights_path=weights,
        data_yaml=data_yaml,
        splits=["test"],
        out_dir=rca_dir,
        sahi=True,
        conf=0.05,
        iou=0.5,
        limit=limit,
    )
    out_json = rca_dir / "failure_modes.json"
    return summarize_rca(csv_path, out_json)


def append_curve_row(curve_csv: Path, row: dict) -> None:
    fieldnames = list(row.keys())
    write_header = not curve_csv.exists()
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    with curve_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def render_curve_png(curve_csv: Path, png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(curve_csv.open()))
    if not rows:
        return
    rows.sort(key=lambda r: r.get("step", ""))
    steps = [r["step"] for r in rows]
    naip = [float(r.get("naip_test_map50", "nan") or "nan") for r in rows]
    duke = [float(r.get("duke_test_map50", "nan") or "nan") for r in rows]
    naip_p = [float(r.get("naip_test_precision", "nan") or "nan") for r in rows]
    sahi_f1s = [float(r.get("sahi_f1_val", "nan") or "nan") for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, naip, "o-", label="NAIP test mAP50 (model.val)", color="#1f77b4")
    ax.plot(steps, duke, "s-", label="Duke test mAP50", color="#d62728")
    ax.plot(steps, naip_p, "^--", label="NAIP test precision", color="#2ca02c")
    ax.plot(steps, sahi_f1s, "D-", label="SAHI F1 val (production metric)", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Ramp step")
    ax.set_ylabel("metric")
    ax.set_title("Duke ramp curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = yaml.safe_load(args.ramp_config.read_text())
    ramp_meta = cfg.get("ramp", {})
    baseline = float(ramp_meta.get("baseline_naip_test_map50", 0.563))
    halt_delta = float(ramp_meta.get("naip_regression_halt_delta", 0.07))
    step_meta = next((s for s in ramp_meta.get("steps", []) if s.get("name") == args.run), {})
    naip_repeat = step_meta.get("naip_repeat")

    weights_path = resolve_weights(args.weights, REPO_ROOT)

    print(f"[{args.run}] eval on NAIP test")
    naip_metrics = run_ultralytics_val(weights_path, args.naip_data)
    print(f"[{args.run}] eval on Duke test")
    duke_metrics = run_ultralytics_val(weights_path, args.duke_data)

    rca_summary: dict = {}
    if not args.skip_rca:
        print(f"[{args.run}] RCA on NAIP test")
        rca_summary = run_rca(weights_path, args.naip_data, args.run, args.limit)

    naip_map = float(naip_metrics.get("seg", {}).get("map50", float("nan")))
    duke_map = float(duke_metrics.get("seg", {}).get("map50", float("nan")))
    naip_p = float(naip_metrics.get("seg", {}).get("precision", float("nan")))
    naip_r = float(naip_metrics.get("seg", {}).get("recall", float("nan")))
    delta = baseline - naip_map if naip_map == naip_map else float("nan")

    fn_rate_small = float("nan")
    fp_rate_overall = float("nan")
    sizes = (rca_summary.get("by_panel_size", {}) if rca_summary else {})
    if "small_lt800" in sizes:
        fn_rate_small = float(sizes["small_lt800"].get("fn_rate", float("nan")))
    overall = (rca_summary.get("overall", {}) if rca_summary else {})
    if overall.get("tp", 0) + overall.get("fp", 0) > 0:
        fp_rate_overall = overall["fp"] / (overall["tp"] + overall["fp"])

    # SAHI F1 from 05d sweep (the production metric). Read from the sweep JSON
    # if it already exists for this run — 05e doesn't re-run 05d to keep runtime
    # reasonable, but callers can run 05d separately and this will pick it up.
    sahi_f1 = float("nan")
    sahi_conf = float("nan")
    sweep_best = REPO_ROOT / "outputs" / "eval" / args.run / "sahi_threshold_sweep_best.json"
    if sweep_best.exists():
        try:
            sb = json.loads(sweep_best.read_text())
            sahi_f1 = float(sb.get("f1", float("nan")))
            sahi_conf = float(sb.get("conf", float("nan")))
        except Exception:
            pass

    row = {
        "step": args.run,
        "naip_repeat": naip_repeat if naip_repeat is not None else "",
        "weights": weights_path.name,
        "naip_test_map50": naip_map,
        "duke_test_map50": duke_map,
        "naip_test_precision": naip_p,
        "naip_test_recall": naip_r,
        "naip_regression_delta": delta,
        "sahi_f1_val": sahi_f1,
        "sahi_conf_val": sahi_conf,
        "fn_rate_small_arrays": fn_rate_small,
        "fp_rate": fp_rate_overall,
    }
    append_curve_row(args.curve_csv, row)
    render_curve_png(args.curve_csv, args.curve_png)

    sahi_str = f"{sahi_f1:.3f} @ conf={sahi_conf:.2f}" if sahi_f1 == sahi_f1 else "n/a (run 05d to populate)"
    print(f"\n[{args.run}] NAIP={naip_map:.3f} Duke={duke_map:.3f} "
          f"P={naip_p:.3f} R={naip_r:.3f} Δ_NAIP={delta:.3f} "
          f"SAHI_F1={sahi_str} "
          f"FN_small={fn_rate_small:.2%} FP_rate={fp_rate_overall:.2%}")
    print(f"        appended → {args.curve_csv}")

    if delta == delta and delta > halt_delta:
        print(f"\n*** HALT: NAIP regression Δ={delta:.3f} > {halt_delta:.3f} threshold. "
              f"Previous ramp step's weights are the production candidate. ***")

    write_manifest(
        REPO_ROOT / "outputs" / "eval" / args.run,
        stage="eval",
        model_version=f"ramp-{args.run}",
        model_weights=weights_path,
        inputs=[str(args.naip_data), str(args.duke_data), str(args.ramp_config)],
        metrics={
            "naip_test_map50": naip_map,
            "duke_test_map50": duke_map,
            "naip_test_precision": naip_p,
            "naip_test_recall": naip_r,
            "naip_regression_delta": delta,
            "sahi_f1_val": sahi_f1,
            "sahi_conf_val": sahi_conf,
            "fn_rate_small_arrays": fn_rate_small,
            "fp_rate": fp_rate_overall,
        },
        known_limitations=[
            f"Halt threshold Δ>{halt_delta} drawn from configs/yolo/experiments_joint_v2_ramp.yaml",
            "RCA uses SAHI inference at conf=0.05; threshold-calibrated metrics differ (use 05d for that)",
        ],
        extra={"step": args.run, "naip_repeat": naip_repeat},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
