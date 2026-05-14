"""HTML quality report builder for ``solarsoiled eval --report``.

Consumes the artifacts produced by scripts/05c, 05d, and 18_bucket_overlays.py
under ``outputs/eval/<run-name>/`` and emits a single self-contained HTML file
(images base64-embedded) suitable to email or open directly in a browser.

API:
    build_report(report_dir, out_path, *, weights_resolved) -> Path
"""
from __future__ import annotations

import base64
import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after backend set)

from solarsoiled.manifest import write_manifest

# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Artifacts:
    manifest: dict | None
    per_detection: list[dict] | None
    sweep: list[dict] | None
    sweep_best: dict | None
    failure_modes: dict | None
    label_viz_root: Path | None
    missing_reasons: dict = field(default_factory=dict)


def _load_json(p: Path, reasons: dict) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        reasons[p.name] = "FileNotFoundError"
    except json.JSONDecodeError as exc:
        reasons[p.name] = f"JSONDecodeError: {exc}"
    return None


def _load_csv(p: Path, reasons: dict) -> list[dict] | None:
    try:
        with p.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except FileNotFoundError:
        reasons[p.name] = "FileNotFoundError"
    except Exception as exc:
        reasons[p.name] = str(exc)
    return None


def _load_artifacts(d: Path) -> Artifacts:
    reasons: dict[str, str] = {}
    manifest = _load_json(d / "manifest.json", reasons)
    per_detection = _load_csv(d / "per_detection.csv", reasons)
    sweep = _load_csv(d / "sahi_threshold_sweep.csv", reasons)
    sweep_best = _load_json(d / "sahi_threshold_sweep_best.json", reasons)
    failure_modes = _load_json(d / "failure_modes.json", reasons)

    # label_viz: look one level up in outputs/label_viz/<run-name>/
    run_name = d.name
    label_viz_root: Path | None = None
    for candidate in [
        d.parents[1] / "label_viz" / run_name,
        d.parent.parent / "label_viz" / run_name,
    ]:
        if candidate.is_dir():
            label_viz_root = candidate
            break

    return Artifacts(
        manifest=manifest,
        per_detection=per_detection,
        sweep=sweep,
        sweep_best=sweep_best,
        failure_modes=failure_modes,
        label_viz_root=label_viz_root,
        missing_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

_CSS = """
body {
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 16px;
  color: #1a1a1a;
  background: #f9f9f9;
}
h1 { font-size: 1.5rem; margin-bottom: 4px; }
h2 { font-size: 1.15rem; margin: 28px 0 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
.meta { color: #555; font-size: .85rem; margin-bottom: 16px; }
.beta-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  background: #f59e0b; color: #fff; font-size: .75rem; font-weight: 600;
}
.missing { color: #888; font-style: italic; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: 8px 0; }
th { background: #eee; text-align: left; padding: 5px 8px; }
td { padding: 4px 8px; border-bottom: 1px solid #eee; font-family: monospace; }
tr:last-child td { border-bottom: none; }
.callout {
  background: #eff6ff; border-left: 4px solid #3b82f6;
  padding: 10px 14px; margin: 8px 0; font-size: .9rem;
}
.callout code { font-weight: 600; }
.overlay-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 8px;
}
.overlay-grid figure { margin: 0; }
.overlay-grid figcaption { font-size: .7rem; color: #555; text-align: center; margin-top: 2px; word-break: break-all; }
.overlay-grid img { width: 100%; border-radius: 4px; border: 1px solid #ddd; }
.bucket-title { font-weight: 600; margin: 16px 0 4px; font-size: .9rem; }
footer { color: #888; font-size: .75rem; margin-top: 40px; padding-top: 12px; border-top: 1px solid #eee; }
ul.limitations { margin: 4px 0; padding-left: 20px; font-size: .85rem; }
"""


def _section(title: str, body_html: str, *, missing: bool = False) -> str:
    body = body_html if not missing else f'<p class="missing">{body_html}</p>'
    return f"<h2>{title}</h2>\n{body}\n"


def _img_to_b64(p: Path) -> str:
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _html_table(
    rows: list[dict],
    cols: list[str],
    *,
    color_col: str | None = None,
    fmt: dict[str, str] | None = None,
) -> str:
    fmt = fmt or {}
    header = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for row in rows:
        cells = []
        for c in cols:
            val = row.get(c, "")
            if val == "":
                display = ""
            elif c in fmt:
                display = fmt[c].format(val)
            elif _looks_float(val):
                try:
                    display = f"{float(val):.4f}"
                except (ValueError, TypeError):
                    display = str(val)
            else:
                display = str(val)
            style = ""
            if color_col and c == color_col:
                try:
                    f1 = float(val)
                    lightness = int(90 - f1 * 40)
                    style = f' style="background:hsl(120,60%,{lightness}%)"'
                except (ValueError, TypeError):
                    pass
            cells.append(f"<td{style}>{display}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _looks_float(v: Any) -> bool:
    if isinstance(v, float):
        return True
    try:
        float(str(v))
        return "." in str(v)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------

def _pr_curve_png_b64(sweep: list[dict]) -> str:
    recalls, precisions, f1s = [], [], []
    for row in sweep:
        try:
            recalls.append(float(row["recall"]))
            precisions.append(float(row["precision"]))
            f1s.append(float(row["f1"]))
        except (KeyError, ValueError):
            continue
    if not recalls:
        return ""

    fig, ax = plt.subplots(figsize=(4, 3), dpi=110)
    sc = ax.scatter(recalls, precisions, c=f1s, cmap="viridis", s=40, zorder=3)
    plt.colorbar(sc, ax=ax, label="F1")

    best_idx = f1s.index(max(f1s))
    ax.annotate(
        f"F1={f1s[best_idx]:.3f}",
        xy=(recalls[best_idx], precisions[best_idx]),
        xytext=(recalls[best_idx] + 0.03, precisions[best_idx] + 0.03),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )
    ax.plot(recalls[best_idx], precisions[best_idx], "*r", ms=10, zorder=4)

    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.set_title("PR Curve — SAHI threshold sweep", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(manifest: dict | None, weights_resolved: Any) -> str:
    if manifest is None:
        return _section("Model", '<p class="missing">manifest.json not found — run scripts/05d to generate.</p>')

    mv = manifest.get("model_version", "unknown")
    sha = manifest.get("model_weights_sha256") or ""
    sha12 = sha[:12] if sha else "n/a"
    gen = manifest.get("generated_at", "")
    beta = manifest.get("beta", True)
    limitations = manifest.get("known_limitations", [])

    badge = '<span class="beta-badge">BETA</span>' if beta else ""
    lim_html = ""
    if limitations:
        items = "".join(f"<li>{_esc(l)}</li>" for l in limitations)
        lim_html = f"<ul class='limitations'>{items}</ul>"

    body = f"""
<div class="meta">
  {badge}
  <strong>{_esc(mv)}</strong>
  &nbsp;·&nbsp; sha256[:12]: <code>{sha12}</code>
  &nbsp;·&nbsp; generated: <code>{_esc(gen)}</code>
</div>
{lim_html}
"""
    return _section("Model", body)


def _render_calibration(sweep: list[dict] | None, best: dict | None) -> str:
    if sweep is None and best is None:
        return _section(
            "Calibration",
            "sahi_threshold_sweep.csv and sahi_threshold_sweep_best.json not generated yet — run scripts/05d.",
            missing=True,
        )

    parts = []

    if best:
        conf = best.get("conf", "?")
        iou = best.get("iou", "?")
        f1 = best.get("f1", 0)
        p = best.get("precision", 0)
        r = best.get("recall", 0)
        tp = best.get("tp", "?")
        fp = best.get("fp", "?")
        fn = best.get("fn", "?")
        parts.append(f"""
<div class="callout">
  Best operating point: <code>conf={conf}, iou={iou}</code>
  &nbsp;·&nbsp; <strong>F1={f1:.3f}</strong>
  &nbsp; P={p:.3f} R={r:.3f}
  &nbsp; TP={tp} FP={fp} FN={fn}
</div>
""")

    if sweep:
        cols = ["conf", "iou", "tp", "fp", "fn", "precision", "recall", "f1"]
        parts.append(_html_table(sweep, cols, color_col="f1"))

        b64 = _pr_curve_png_b64(sweep)
        if b64:
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width:480px;margin-top:12px;">')

    return _section("Calibration", "\n".join(parts))


def _render_failure_modes(fm: dict | None) -> str:
    if fm is None:
        return _section(
            "Failure-mode breakdown",
            "failure_modes.json not generated yet — run scripts/05c --summarize.",
            missing=True,
        )

    cols = ["bucket", "tp", "fp", "fn", "fn_rate", "precision"]
    parts = []

    overall = fm.get("overall", {})
    if overall:
        tp, fp, fn = overall.get("tp", 0), overall.get("fp", 0), overall.get("fn", 0)
        p = overall.get("precision", 0)
        r = overall.get("recall", 0)
        parts.append(f'<div class="callout">Overall — TP={tp} FP={fp} FN={fn} &nbsp; P={p:.3f} R={r:.3f}</div>')

    for group_key, group_label in [
        ("by_panel_size", "By panel size"),
        ("by_edge_distance", "By edge distance"),
        ("by_density", "By density"),
    ]:
        group = fm.get(group_key, {})
        if not group:
            continue
        rows = [{"bucket": k, **v} for k, v in group.items()]
        parts.append(f'<div class="bucket-title">{group_label}</div>')
        parts.append(_html_table(rows, cols))

    return _section("Failure-mode breakdown", "\n".join(parts))


def _render_overlays(label_viz_root: Path | None, n: int = 6) -> str:
    if label_viz_root is None:
        return _section(
            "Sample overlays",
            "outputs/label_viz/<run-name>/ not found — run scripts/labeling/18_bucket_overlays.py.",
            missing=True,
        )

    bucket_dirs = sorted(p for p in label_viz_root.iterdir() if p.is_dir())
    if not bucket_dirs:
        return _section("Sample overlays", "No bucket directories found under label_viz.", missing=True)

    parts = [f"<p>{len(bucket_dirs)} bucket(s) found: {', '.join(d.name for d in bucket_dirs)}</p>"]

    for bdir in bucket_dirs:
        pngs = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in bdir.glob(ext))[:n]
        if not pngs:
            continue
        parts.append(f'<div class="bucket-title">{_esc(bdir.name)} ({len(pngs)} shown)</div>')
        parts.append('<div class="overlay-grid">')
        for png in pngs:
            try:
                b64 = _img_to_b64(png)
                parts.append(
                    f'<figure><img src="{b64}" loading="lazy">'
                    f'<figcaption>{_esc(png.stem[:40])} · {_esc(bdir.name)}</figcaption></figure>'
                )
            except Exception:
                pass
        parts.append("</div>")

    return _section(f"Sample overlays", "\n".join(parts))


def _render_worst_offenders(per_det: list[dict] | None) -> str:
    if per_det is None:
        return _section(
            "Per-tile worst offenders",
            "per_detection.csv not found — run scripts/05c.",
            missing=True,
        )

    tile_stats: dict[str, dict] = {}
    for row in per_det:
        tid = row.get("tile_id", "")
        cls = row.get("class", "")
        iou_val = row.get("iou", "")
        if tid not in tile_stats:
            tile_stats[tid] = {"tile_id": tid, "fp": 0, "fn": 0, "total": 0, "iou_sum": 0.0, "iou_count": 0}
        s = tile_stats[tid]
        if cls == "fp":
            s["fp"] += 1
        elif cls == "fn":
            s["fn"] += 1
        s["total"] += 1
        try:
            s["iou_sum"] += float(iou_val)
            s["iou_count"] += 1
        except (ValueError, TypeError):
            pass

    rows = sorted(tile_stats.values(), key=lambda r: r["fp"] + r["fn"], reverse=True)[:10]
    for r in rows:
        r["mean_iou"] = r["iou_sum"] / r["iou_count"] if r["iou_count"] else 0.0

    cols = ["tile_id", "fp", "fn", "total", "mean_iou"]
    return _section("Per-tile worst offenders (top 10)", _html_table(rows, cols))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_report(
    report_dir: Path | None,
    out_path: Path | None,
    *,
    weights_resolved: Any = None,
    n_overlays: int = 6,
) -> Path:
    """Build a single-file HTML quality report from eval artifacts.

    Args:
        report_dir: outputs/eval/<run-name>/ to ingest. If None, uses the
            most recent directory under outputs/eval/ relative to repo root.
        out_path: HTML output path. Defaults to report_dir/report.html.
        weights_resolved: ResolvedWeights instance (for manifest provenance).
        n_overlays: max overlay PNGs per bucket.

    Returns:
        Path to the written HTML file.
    """
    if report_dir is None:
        from solarsoiled.paths import REPO_ROOT
        eval_root = REPO_ROOT / "outputs" / "eval"
        candidates = [d for d in eval_root.iterdir() if d.is_dir()] if eval_root.is_dir() else []
        if not candidates:
            raise ValueError("No directories found under outputs/eval/ — pass --report-dir explicitly.")
        report_dir = max(candidates, key=lambda d: d.stat().st_mtime)

    report_dir = Path(report_dir)
    if out_path is None:
        out_path = report_dir / "report.html"
    out_path = Path(out_path)

    art = _load_artifacts(report_dir)

    sections = [
        _render_header(art.manifest, weights_resolved),
        _render_calibration(art.sweep, art.sweep_best),
        _render_failure_modes(art.failure_modes),
        _render_overlays(art.label_viz_root, n=n_overlays),
        _render_worst_offenders(art.per_detection),
    ]

    inputs_hash = art.manifest.get("inputs_hash", "n/a") if art.manifest else "n/a"
    missing_list = ", ".join(art.missing_reasons.keys()) or "none"
    footer = (
        f'<footer>inputs_hash: <code>{inputs_hash}</code>'
        f' &nbsp;·&nbsp; missing artifacts: <code>{missing_list}</code>'
        f' &nbsp;·&nbsp; generated: <code>{datetime.now(timezone.utc).isoformat()}</code>'
        f"</footer>"
    )

    run_name = report_dir.name
    mv = art.manifest.get("model_version", "unknown") if art.manifest else "unknown"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>solarsoiled eval report — {_esc(run_name)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>solarsoiled eval — {_esc(run_name)}</h1>
<div class="meta">{_esc(mv)}</div>
{''.join(sections)}
{footer}
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # Write sibling manifest
    best = art.sweep_best or {}
    metrics_payload: dict | None = None
    if best:
        metrics_payload = {
            "best_f1": best.get("f1"),
            "best_conf": best.get("conf"),
            "best_iou": best.get("iou"),
        }

    artifact_paths = [
        report_dir / "per_detection.csv",
        report_dir / "sahi_threshold_sweep.csv",
        report_dir / "sahi_threshold_sweep_best.json",
        report_dir / "failure_modes.json",
        report_dir / "manifest.json",
    ]

    mv_for_manifest = "unknown"
    weights_path_for_manifest: Path | None = None
    if weights_resolved is not None:
        mv_for_manifest = getattr(weights_resolved, "model_version", "unknown")
        weights_path_for_manifest = getattr(weights_resolved, "path", None)

    write_manifest(
        out_path.parent,
        stage="eval",
        model_version=mv_for_manifest,
        model_weights=weights_path_for_manifest,
        inputs=[str(p) for p in artifact_paths],
        metrics=metrics_payload,
        extra={
            "report_html": str(out_path),
            "missing_artifacts": list(art.missing_reasons.keys()),
        },
    )

    return out_path
