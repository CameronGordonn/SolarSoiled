# ── SolarSoiled container ─────────────────────────────────────────────────────
# Build:
#   docker build -t solarsoiled .                          # CPU (default)
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#     -t solarsoiled:gpu .                                 # CUDA 12.1
#
# ── CLI pipeline ──────────────────────────────────────────────────────────────
#   docker run --rm \
#     -e NASA_EARTHDATA_TOKEN=<token> \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/runs:/app/runs \
#     -v $(pwd)/outputs:/app/outputs \
#     -v $(pwd)/.cache:/app/.cache \
#     -v $(pwd)/data/external:/app/data/external \
#     solarsoiled run \
#       --aoi "-122.05,36.90,-121.85,37.05" \
#       --weights production \
#       --soiling-model soiling_production \
#       --last-cleaned 2025-12-01 \
#       --partner-id smoketest
#
# ── API server ────────────────────────────────────────────────────────────────
#   docker run --rm -p 8000:8000 \
#     -e SOLARSOILED_API_KEY=<key> \
#     -e NASA_EARTHDATA_TOKEN=<token> \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/runs:/app/runs \
#     -v $(pwd)/outputs:/app/outputs \
#     -v $(pwd)/.cache:/app/.cache \
#     -v $(pwd)/data/external:/app/data/external \
#     --entrypoint solarsoiled-api \
#     solarsoiled
#
# Or use docker-compose up api  (see docker-compose.yml)
#
# ── Volume mounts ─────────────────────────────────────────────────────────────
#   models/          .pt weight files (production → r2-cameron-20260509.pt, etc.)
#   runs/            soiling model artifacts (runs/soiling/<run>/{model.ubj,calibrator.joblib})
#   outputs/         pipeline writes arrays.geojson, risk.geojson, etc. here
#   .cache/          weather + AQ API response cache (~440 MB after warm-up)
#   data/external/   static features CSV (elevation, WorldCover, OSM distances)
#
# NASA_EARTHDATA_TOKEN (optional): enables MERRA-2 PM2.5/PM10 backfill (1980-present).
# Without it, AQ features are populated from CAMS (2022-present only).
# Register at https://urs.earthdata.nasa.gov
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# OpenCV runtime libs (libgl1 + libglib2.0-0).
# rasterio/GDAL/PROJ/GEOS are bundled in their pip wheels — no apt packages needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── torch first: big layer, rarely changes ────────────────────────────────────
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url ${TORCH_INDEX}

# ── install package (deps pulled from pyproject.toml) ────────────────────────
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[api]"

# ── scripts + configs (imported by CLI as library calls) ─────────────────────
COPY scripts/ scripts/
COPY configs/ configs/

# ── model registry (resolves --weights names; .pt files are runtime mounts) ──
COPY models/registry.yaml models/registry.yaml

VOLUME ["/app/models", "/app/runs", "/app/outputs", "/app/.cache", "/app/data/external"]

ENTRYPOINT ["solarsoiled"]
CMD ["--help"]
