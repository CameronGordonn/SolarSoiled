"""Per-AOI output namespace.

Every CLI subcommand and the API layer must read/write artifacts at the same
paths. Centralized here so a layout change is a one-file edit.

Layout matches the v0 beta `/detect/jobs/{job_id}/result` contract sketched
in `docs/PRODUCT_VISION.md`.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AOI_ROOT = REPO_ROOT / "outputs" / "aoi"


class AoiPaths:
    """All artifact paths for a single AOI run."""

    def __init__(self, aoi_id: str, root: Path | None = None) -> None:
        self.aoi_id = aoi_id
        self.root = (root or AOI_ROOT) / aoi_id

    # Stage outputs
    @property
    def aoi_geojson(self) -> Path:
        return self.root / "aoi.geojson"

    @property
    def tiles_dir(self) -> Path:
        return self.root / "tiles"

    @property
    def tile_index(self) -> Path:
        return self.tiles_dir / "tile_index.json"

    @property
    def detect_dir(self) -> Path:
        return self.root / "detect"

    @property
    def detect_labels_dir(self) -> Path:
        return self.detect_dir / "labels"

    @property
    def arrays_geojson(self) -> Path:
        return self.root / "arrays.geojson"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def array_features_parquet(self) -> Path:
        return self.features_dir / "array_features.parquet"

    @property
    def array_features_geo_parquet(self) -> Path:
        return self.features_dir / "array_features.geo.parquet"

    @property
    def inference_matrix(self) -> Path:
        return self.features_dir / "inference_matrix.parquet"

    @property
    def risk_geojson(self) -> Path:
        return self.root / "risk.geojson"

    @property
    def recommendations_json(self) -> Path:
        return self.root / "recommendations.json"

    # Manifests (each subdir already gets one via write_manifest;
    # this is the AOI-level rollup written by `run`)
    @property
    def rollup_manifest(self) -> Path:
        return self.root / "manifest.json"

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root
