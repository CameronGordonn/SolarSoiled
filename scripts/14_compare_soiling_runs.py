"""Quick comparison table for runs/soiling/<run>/metrics.json files.

Usage:
    python scripts/14_compare_soiling_runs.py                          # all runs
    python scripts/14_compare_soiling_runs.py run_d run_e run_f        # specific runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="Run names (default: all under runs/soiling/)")
    args = parser.parse_args()

    runs_dir = repo_root / "runs" / "soiling"
    if args.runs:
        names = args.runs
    else:
        names = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())

    rows = []
    for name in names:
        m_path = runs_dir / name / "metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        rows.append({
            "run": name,
            "n_folds": m.get("n_folds"),
            "target": m.get("target_mode", "binary"),
            "auc": round(m.get("mean_auc", float("nan")), 3),
            "ap": round(m.get("mean_ap", float("nan")), 3),
            "spearman": round(m["mean_spearman"], 3) if "mean_spearman" in m else None,
            "fold_aucs": [round(a, 3) for a in m.get("fold_aucs", [])],
            "holdout_year": m.get("holdout_year"),
            "holdout_auc": round(m["holdout_auc"], 3) if "holdout_auc" in m else None,
            "holdout_n": m.get("holdout_n"),
        })
    df = pd.DataFrame(rows).sort_values("run")
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
