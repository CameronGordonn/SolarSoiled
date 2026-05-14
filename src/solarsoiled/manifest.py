"""Output manifest writer — sibling ``manifest.json`` for every artifact dir.

The manifest schema mirrors the per-response metadata block specified in
``docs/PRODUCT_VISION.md`` for the v0 beta API. Writing it from every
script that produces ``outputs/`` or ``runs/`` artifacts means the same
contract holds whether a result was produced by a CLI invocation, an API
job, or a notebook — and saves a retrofit later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "1"

KNOWN_STAGES = {
    "tile",
    "stage1_detect",
    "stage2_train",
    "stage2_score",
    "recommend",
    "eval",
}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def hash_inputs(inputs: Any) -> str:
    """Stable SHA256 over a manifest's ``inputs`` field.

    For a list/tuple of paths, the digest is over each file's content hash so
    moving the file changes nothing but editing it does. For other shapes the
    digest is over canonical JSON.
    """
    if isinstance(inputs, (list, tuple)) and all(
        isinstance(x, (str, Path)) for x in inputs
    ):
        h = hashlib.sha256()
        for raw in sorted(str(p) for p in inputs):
            p = Path(raw)
            if p.is_file():
                h.update(p.name.encode("utf-8"))
                h.update(b"\0")
                h.update(_sha256_file(p).encode("ascii"))
            else:
                h.update(raw.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()
    return hashlib.sha256(_canonical_json(inputs)).hexdigest()


def build_manifest(
    *,
    stage: str,
    model_version: str,
    model_weights: str | Path | None = None,
    inputs: Any = None,
    beta: bool = True,
    metrics: Mapping[str, float] | None = None,
    known_limitations: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a manifest dict without writing it."""
    if stage not in KNOWN_STAGES:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {sorted(KNOWN_STAGES)}"
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "beta": bool(beta),
    }

    if model_weights is not None:
        weights_path = Path(model_weights)
        if weights_path.is_file():
            manifest["model_weights_path"] = str(weights_path)
            manifest["model_weights_sha256"] = _sha256_file(weights_path)
        else:
            manifest["model_weights_path"] = str(weights_path)
            manifest["model_weights_sha256"] = None

    if inputs is not None:
        manifest["inputs"] = inputs if not isinstance(inputs, tuple) else list(inputs)
        manifest["inputs_hash"] = hash_inputs(inputs)

    if metrics:
        manifest["metrics"] = dict(metrics)

    limitations = list(known_limitations) if known_limitations else []
    manifest["known_limitations"] = limitations

    if extra:
        for key, value in extra.items():
            if key in manifest:
                raise ValueError(f"extra field {key!r} collides with reserved manifest key")
            manifest[key] = value

    return manifest


def write_manifest(
    output_dir: str | Path,
    *,
    stage: str,
    model_version: str,
    model_weights: str | Path | None = None,
    inputs: Any = None,
    beta: bool = True,
    metrics: Mapping[str, float] | None = None,
    known_limitations: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "manifest.json",
) -> Path:
    """Build a manifest and write it to ``output_dir/manifest.json``.

    Returns the path written. The output directory is created if missing so
    callers don't need to special-case first runs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        stage=stage,
        model_version=model_version,
        model_weights=model_weights,
        inputs=inputs,
        beta=beta,
        metrics=metrics,
        known_limitations=known_limitations,
        extra=extra,
    )

    out_path = out_dir / filename
    out_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return out_path
