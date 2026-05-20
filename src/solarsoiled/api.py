"""FastAPI backend for the solarsoiled pipeline.

Exposes async job submission, SSE progress streaming, and artifact serving.
Start with: uvicorn solarsoiled.api:app --reload
or:         solarsoiled-api  (after pip install -e ".[api]")

Auth: set SOLARSOILED_API_KEY env var. If unset, auth is disabled (local dev).
Per-partner auth: set SOLARSOILED_KEYS_FILE to a YAML file mapping keys to partner_ids.
If unset, falls back to single-key mode (all authenticated callers are equal).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from queue import Empty

import re

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from solarsoiled.aoi import parse_aoi, write_aoi_geojson
from solarsoiled.jobs import JobRecord, create_job, get_job, submit
from solarsoiled.paths import AoiPaths, REPO_ROOT
from solarsoiled.recommend import (
    recommend_cleaning,
    recommend_per_array,
    write_array_recommendations,
    write_recommendation,
)
from solarsoiled.registry import RegistryError, resolve as resolve_weights, resolve_soiling
from solarsoiled.viz import build_risk_map


app = FastAPI(
    title="SolarSoiled API",
    description="Detect solar arrays, score soiling risk, recommend cleaning.",
    version="0.5.0",
)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain before production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_API_KEY = os.environ.get("SOLARSOILED_API_KEY", "")


def _load_key_registry() -> dict | None:
    path = os.environ.get("SOLARSOILED_KEYS_FILE", "")
    if not path:
        return None
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("keys", {})


_KEY_REGISTRY: dict | None = _load_key_registry()

_SCRIPT_FILES = {
    "tile": REPO_ROOT / "scripts" / "02_tile_naip_image.py",
    "infer": REPO_ROOT / "scripts" / "04_infer_yolov8_seg.py",
    "export_polygons": REPO_ROOT / "scripts" / "06_export_polygons_geojson.py",
    "extract_features": REPO_ROOT / "scripts" / "07_extract_array_features.py",
    "build_features": REPO_ROOT / "scripts" / "09_build_soiling_features.py",
    "predict_risk": REPO_ROOT / "scripts" / "11_predict_soiling_risk.py",
}


def _load_script(key: str):
    path = _SCRIPT_FILES[key]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"_solarsoiled_api_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------- auth + input validation ----------

async def _resolve_caller(x_api_key: str = Header(default="")) -> str | None:
    """Validate the API key and return the caller's partner_id, or None for unrestricted access.

    Returns None in three cases: single-key mode (SOLARSOILED_KEYS_FILE unset),
    wildcard key (partner_id: "*"), or no key configured at all (local dev).
    """
    if _KEY_REGISTRY is None:
        # single-key mode — backward compatible
        if _API_KEY and x_api_key != _API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
        return None
    entry = _KEY_REGISTRY.get(x_api_key)
    if entry is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    pid = entry.get("partner_id", "")
    return None if pid == "*" else pid


def _assert_partner_access(caller_partner_id: str | None, requested_partner_id: str | None) -> None:
    """Raise 403 if the caller's key is partner-bound and doesn't match the requested partner."""
    if caller_partner_id is None or requested_partner_id is None:
        return  # wildcard key, single-key mode, or ownerless job
    if caller_partner_id != requested_partner_id:
        raise HTTPException(status_code=403, detail="Access denied: key not authorized for this partner")


_PARTNER_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,62}$")


def _validate_partner_id(partner_id: str) -> str:
    if not _PARTNER_ID_RE.fullmatch(partner_id):
        raise HTTPException(status_code=422, detail="Invalid partner_id: must be 1-63 alphanumeric/hyphen/underscore characters")
    return partner_id


# ---------- request model ----------

class FeedbackRequest(BaseModel):
    array_id: int
    partner_id: str
    cleaned_at: str                    # YYYY-MM-DD
    pre_clean_kwh_7d: float
    post_clean_kwh_7d: float
    notes: str | None = None


class RunRequest(BaseModel):
    aoi: str
    weights: str = "production"
    soiling_model: str = "soiling_production"
    last_cleaned: str                  # YYYY-MM-DD
    as_of: str | None = None
    partner_id: str | None = None
    skip_tile: bool = False
    skip_detect: bool = False
    skip_score: bool = False
    skip_recommend: bool = False
    risk_threshold: float = 0.6


# ---------- pipeline runner (runs in thread pool) ----------

def _run_pipeline(record: JobRecord, *, req: RunRequest) -> dict:
    aoi_obj = parse_aoi(req.aoi, partner_id=req.partner_id)
    paths = AoiPaths(aoi_obj.aoi_id)
    paths.ensure_root()
    write_aoi_geojson(aoi_obj, paths.aoi_geojson)

    def emit(stage: str, msg: str = "") -> None:
        record.current_stage = stage
        record._events.put({"event": "stage", "data": {"stage": stage, "message": msg}})

    if not req.skip_tile:
        emit("tile", "tiling NAIP imagery")
        _load_script("tile").main(
            download_aoi=None,
            out_tiles_dir=paths.tiles_dir,
            out_tile_index=paths.tile_index,
        )

    if not req.skip_detect:
        if not paths.tile_index.is_file():
            raise RuntimeError(f"tile_index missing — run with skip_tile=false")
        emit("detect", "running YOLOv11 detection")
        try:
            resolved = resolve_weights(req.weights)
        except RegistryError as exc:
            raise RuntimeError(str(exc)) from exc
        _load_script("infer").main([
            "--weights", str(resolved.path),
            "--source", str(paths.tiles_dir),
            "--project", str(paths.root),
            "--name", "detect",
            "--conf", "0.40",
            "--iou", "0.50",
            "--sahi",
        ])
        _load_script("export_polygons").export_polygons(
            labels_dir=paths.detect_labels_dir,
            tile_index_path=paths.tile_index,
            output_geojson=paths.arrays_geojson,
        )

    if not req.skip_score:
        if not paths.arrays_geojson.is_file():
            raise RuntimeError(f"arrays.geojson missing — run with skip_detect=false")
        emit("score", "scoring soiling risk")
        paths.features_dir.mkdir(parents=True, exist_ok=True)
        try:
            resolved_soiling = resolve_soiling(req.soiling_model)
        except RegistryError as exc:
            raise RuntimeError(str(exc)) from exc
        _load_script("extract_features").extract_features(
            input_geojson=paths.arrays_geojson,
            out_table=paths.array_features_parquet,
            out_geo=paths.array_features_geo_parquet,
        )
        build_argv = [
            "--config", str(REPO_ROOT / "configs" / "soiling" / "california.yaml"),
            "--features-config", str(REPO_ROOT / "configs" / "soiling" / "features.yaml"),
            "--arrays", str(paths.array_features_geo_parquet),
            "--out", str(paths.inference_matrix),
        ]
        if req.as_of:
            build_argv += ["--as-of", req.as_of]
        _load_script("build_features").main(build_argv)
        _load_script("predict_risk").main([
            "--model", str(resolved_soiling.path),
            "--features", str(paths.inference_matrix),
            "--arrays", str(paths.array_features_geo_parquet),
            "--out", str(paths.risk_geojson),
        ])

    if not req.skip_recommend:
        if not paths.risk_geojson.is_file():
            raise RuntimeError(f"risk.geojson missing — run with skip_score=false")
        emit("recommend", "computing cleaning recommendations")
        centroid = aoi_obj.polygon.centroid
        payload = recommend_cleaning(
            risk_geojson=paths.risk_geojson,
            last_cleaned=date.fromisoformat(req.last_cleaned),
            aoi_centroid=(float(centroid.y), float(centroid.x)),
            risk_threshold=req.risk_threshold,
        )
        write_recommendation(paths.recommendations_json, payload)
        array_rows = recommend_per_array(
            paths.risk_geojson,
            payload,
            risk_threshold=req.risk_threshold,
        )
        write_array_recommendations(paths.array_recommendations_json, array_rows)

    emit("viz", "rendering map")
    build_risk_map(
        paths.risk_geojson,
        paths.root / "risk_map.html",
        recommendations_json=paths.recommendations_json if paths.recommendations_json.exists() else None,
        array_recommendations_json=paths.array_recommendations_json if paths.array_recommendations_json.exists() else None,
    )

    return {
        "partner_id": aoi_obj.aoi_id,
        "risk_map_url": f"/results/{aoi_obj.aoi_id}/map",
        "arrays_url": f"/results/{aoi_obj.aoi_id}/arrays",
        "recommendations_url": f"/results/{aoi_obj.aoi_id}/recommendations",
    }


# ---------- endpoints ----------

@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    notes = []
    if not _API_KEY:
        notes.append("SOLARSOILED_API_KEY unset — auth disabled")
    try:
        resolve_soiling("soiling_production")
    except Exception as exc:
        notes.append(f"soiling registry: {exc}")
    return {"status": "ok" if not any("registry" in n for n in notes) else "degraded", "notes": notes}


@app.post("/jobs")
async def create_run_job(req: RunRequest, caller: str | None = Depends(_resolve_caller)):
    if req.partner_id is not None:
        _validate_partner_id(req.partner_id)
        _assert_partner_access(caller, req.partner_id)
    record = create_job(partner_id=req.partner_id)
    submit(record, _run_pipeline, req=req)
    return {"job_id": record.job_id, "status": record.status, "partner_id": record.partner_id}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, caller: str | None = Depends(_resolve_caller)):
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _assert_partner_access(caller, record.partner_id)
    return record.to_dict()


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, caller: str | None = Depends(_resolve_caller)):
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _assert_partner_access(caller, record.partner_id)

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(
                    None, lambda: record._events.get(timeout=30)
                )
                if event is None:  # sentinel — stream closed, done/error already emitted
                    break
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
            except Empty:
                if record.status in ("done", "failed"):
                    break
                yield ": keepalive\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, caller: str | None = Depends(_resolve_caller)):
    _validate_partner_id(req.partner_id)
    _assert_partner_access(caller, req.partner_id)
    paths = AoiPaths(req.partner_id)
    paths.ensure_root()

    recovery_pct = (req.post_clean_kwh_7d - req.pre_clean_kwh_7d) / req.pre_clean_kwh_7d * 100

    record = {
        "array_id": req.array_id,
        "partner_id": req.partner_id,
        "cleaned_at": req.cleaned_at,
        "pre_clean_kwh_7d": req.pre_clean_kwh_7d,
        "post_clean_kwh_7d": req.post_clean_kwh_7d,
        "actual_recovery_pct": round(recovery_pct, 2),
        "notes": req.notes,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    p = paths.feedback_json
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    return {
        "status": "recorded",
        "array_id": req.array_id,
        "actual_recovery_pct": record["actual_recovery_pct"],
        "feedback_count": len(existing),
    }


@app.get("/results/{partner_id}/map")
async def get_risk_map(partner_id: str, caller: str | None = Depends(_resolve_caller)):
    _validate_partner_id(partner_id)
    _assert_partner_access(caller, partner_id)
    p = AoiPaths(partner_id).root / "risk_map.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Map not yet generated — run a job first")
    return FileResponse(str(p), media_type="text/html")


@app.get("/results/{partner_id}/arrays")
async def get_arrays(partner_id: str, caller: str | None = Depends(_resolve_caller)):
    _validate_partner_id(partner_id)
    _assert_partner_access(caller, partner_id)
    p = AoiPaths(partner_id).risk_geojson
    if not p.exists():
        raise HTTPException(status_code=404, detail="Arrays not yet scored")
    return FileResponse(str(p), media_type="application/geo+json")


@app.get("/results/{partner_id}/recommendations")
async def get_recommendations(partner_id: str, caller: str | None = Depends(_resolve_caller)):
    _validate_partner_id(partner_id)
    _assert_partner_access(caller, partner_id)
    p = AoiPaths(partner_id).array_recommendations_json
    if not p.exists():
        raise HTTPException(status_code=404, detail="Recommendations not yet generated")
    return FileResponse(str(p), media_type="application/json")


def main() -> None:
    import uvicorn
    uvicorn.run("solarsoiled.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
