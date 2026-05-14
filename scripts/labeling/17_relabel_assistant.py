#!/usr/bin/env python3
"""Solo CLI relabel walkthrough for tiles flagged by the disagreement analyzer.

Reads outputs/eval/label_disagreement*.csv (sorted by disagreement_score desc),
walks the user through each tile, and applies one of these actions:

  k   keep all polygons as-is
  r   remove specific polygons (prompted for indices, 1-based)
  R   remove ALL polygons (use when current label is fully wrong)
  a   flag this tile for re-annotation (no in-place edits — user redraws in
        Roboflow / makesense.ai later; tile is logged to outputs/label_viz/relabel_queue.csv)
  s   skip (no change)
  q   quit early (unprocessed tiles stay queued)

Side effects:
  - In-place edits to label files under data/yolo/naip/labels/<split>/
  - outputs/label_viz/relabel_log.csv  (audit trail of every decision)
  - outputs/label_viz/relabel_queue.csv  (tiles flagged for re-annotation)

Operation is idempotent: a label already edited won't be re-shown unless
--force-rerun is passed.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disagreement-csv", type=Path,
                   default=REPO_ROOT / "outputs" / "eval" / "label_disagreement.csv")
    p.add_argument("--overlay-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "label_viz" / "disagreement")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "yolo" / "naip" / "data.yaml")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--top-n", type=int, default=30, help="Walk through this many tiles, sorted by disagreement_score desc")
    p.add_argument("--min-score", type=int, default=1,
                   help="Skip tiles with disagreement_score below this threshold")
    p.add_argument("--log", type=Path, default=REPO_ROOT / "outputs" / "label_viz" / "relabel_log.csv")
    p.add_argument("--queue", type=Path, default=REPO_ROOT / "outputs" / "label_viz" / "relabel_queue.csv")
    p.add_argument("--backup-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "label_viz" / "relabel_backups")
    p.add_argument("--open-cmd", default=None,
                   help="Optional shell command to open the overlay PNG (e.g. 'xdg-open' or 'open'). Auto-detected if omitted.")
    p.add_argument("--force-rerun", action="store_true",
                   help="Re-process tiles already present in the relabel log")
    return p.parse_args()


def load_data_root(data_yaml: Path) -> Path:
    with data_yaml.open() as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def detect_open_cmd() -> str | None:
    for cmd in ("xdg-open", "open", "wslview"):
        if shutil.which(cmd):
            return cmd
    return None


def already_processed(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    seen: set[str] = set()
    with log_path.open() as f:
        for row in csv.DictReader(f):
            seen.add(row["tile_id"])
    return seen


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def backup_label(label_path: Path, backup_dir: Path) -> None:
    if not label_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{label_path.stem}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"
    shutil.copy2(label_path, dest)


def read_polys(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    return [line for line in label_path.read_text().splitlines() if line.strip()]


def write_polys(label_path: Path, polys: List[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(polys) + ("\n" if polys else ""))


def prompt_remove_indices(n: int) -> List[int]:
    while True:
        raw = input(f"  Indices to remove (comma-separated, 1..{n}, or empty to cancel): ").strip()
        if not raw:
            return []
        try:
            idxs = sorted({int(x.strip()) - 1 for x in raw.split(",") if x.strip()})
        except ValueError:
            print("    Invalid — enter integers like '1,3'")
            continue
        if any(i < 0 or i >= n for i in idxs):
            print(f"    Out of range (allowed 1..{n})")
            continue
        return idxs


def main() -> int:
    args = parse_args()

    if not args.disagreement_csv.exists():
        print(f"ERROR: disagreement CSV not found: {args.disagreement_csv}", file=sys.stderr)
        print("  Run scripts/labeling/15_label_disagreement.py first.", file=sys.stderr)
        return 1

    data_root = load_data_root(args.data)
    labels_dir = data_root / "labels" / args.split
    if not labels_dir.exists():
        print(f"ERROR: labels dir not found: {labels_dir}", file=sys.stderr)
        return 1

    open_cmd = args.open_cmd or detect_open_cmd()
    seen = set() if args.force_rerun else already_processed(args.log)

    with args.disagreement_csv.open() as f:
        all_rows = [r for r in csv.DictReader(f)]
    all_rows.sort(key=lambda r: int(r.get("disagreement_score", 0)), reverse=True)

    work = [r for r in all_rows
            if int(r.get("disagreement_score", 0)) >= args.min_score
            and r["tile_id"] not in seen][: args.top_n]

    if not work:
        print("Nothing to review (all tiles already processed or below --min-score).")
        return 0

    print(f"Reviewing {len(work)} tile(s). Press Enter after opening each overlay PNG.")
    print("  Commands: k=keep  r=remove some  R=remove all  a=flag for re-annotation  s=skip  q=quit\n")

    log_fields = ["timestamp", "tile_id", "split", "bucket", "disagreement_score",
                  "action", "removed_indices", "polys_before", "polys_after", "note"]
    queue_fields = ["timestamp", "tile_id", "split", "bucket", "reason", "source_ortho"]

    for i, row in enumerate(work, 1):
        tile_id = row["tile_id"]
        bucket = row.get("bucket", "")
        score = row.get("disagreement_score", "")
        overlay = args.overlay_dir / tile_id

        label_path = labels_dir / (Path(tile_id).stem + ".txt")
        polys = read_polys(label_path)
        n_before = len(polys)

        print(f"\n[{i}/{len(work)}] {tile_id}")
        print(f"  bucket={bucket}  disagreement_score={score}  current_polygons={n_before}  source={row.get('source_ortho','?')}")
        print(f"  overlay: {overlay}")
        if overlay.exists() and open_cmd:
            try:
                subprocess.Popen([open_cmd, str(overlay)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                print(f"  (couldn't auto-open: {exc}; open it manually)")
        elif not overlay.exists():
            print("  (overlay PNG not found — render it via 15_label_disagreement.py first)")

        action = input("  Action [k/r/R/a/s/q]: ").strip()
        timestamp = datetime.now().isoformat(timespec="seconds")
        log_row = {
            "timestamp": timestamp, "tile_id": tile_id, "split": args.split,
            "bucket": bucket, "disagreement_score": score,
            "action": "", "removed_indices": "", "polys_before": n_before,
            "polys_after": n_before, "note": "",
        }

        if action == "q":
            print("Quitting; remaining tiles stay queued.")
            break
        elif action == "" or action.lower() == "s":
            log_row["action"] = "skip"
        elif action == "k":
            log_row["action"] = "keep"
        elif action == "r":
            if n_before == 0:
                print("  No polygons to remove.")
                log_row["action"] = "noop_no_polys"
            else:
                idxs = prompt_remove_indices(n_before)
                if not idxs:
                    log_row["action"] = "cancel_remove"
                else:
                    backup_label(label_path, args.backup_dir)
                    keep = [p for j, p in enumerate(polys) if j not in idxs]
                    write_polys(label_path, keep)
                    log_row["action"] = "remove"
                    log_row["removed_indices"] = ";".join(str(j + 1) for j in idxs)
                    log_row["polys_after"] = len(keep)
        elif action == "R":
            if n_before == 0:
                print("  No polygons to remove.")
                log_row["action"] = "noop_no_polys"
            else:
                backup_label(label_path, args.backup_dir)
                write_polys(label_path, [])
                log_row["action"] = "remove_all"
                log_row["polys_after"] = 0
        elif action == "a":
            note = input("  Note (what's missing/wrong, optional): ").strip()
            log_row["action"] = "flag_for_reannotation"
            log_row["note"] = note
            append_row(args.queue, {
                "timestamp": timestamp, "tile_id": tile_id, "split": args.split,
                "bucket": bucket, "reason": note or "flagged from disagreement review",
                "source_ortho": row.get("source_ortho", ""),
            }, queue_fields)
        else:
            print(f"  Unknown action {action!r} — skipping.")
            log_row["action"] = "unknown_input"

        append_row(args.log, log_row, log_fields)

    print(f"\nDone. Log: {args.log}")
    if args.queue.exists():
        print(f"Re-annotation queue: {args.queue}")
    if args.backup_dir.exists() and any(args.backup_dir.iterdir()):
        print(f"Backups for any modified labels: {args.backup_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
