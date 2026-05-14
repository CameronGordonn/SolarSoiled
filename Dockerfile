# ── SolarSoiled container ─────────────────────────────────────────────────────
# CPU build (default):
#   docker build -t solarsoiled .
#
# GPU build (CUDA 12.1):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#     -t solarsoiled:gpu .
#
# Run:
#   docker run --rm \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/outputs:/app/outputs \
#     -v $(pwd)/.cache:/app/.cache \
#     solarsoiled run \
#       --aoi "-122.05,36.90,-121.85,37.05" \
#       --weights production \
#       --soiling-model soiling_production \
#       --last-cleaned 2025-12-01 \
#       --partner-id smoketest
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
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url ${TORCH_INDEX}

# ── install package (deps pulled from pyproject.toml) ────────────────────────
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# ── scripts + configs (imported by CLI as library calls) ─────────────────────
COPY scripts/ scripts/
COPY configs/ configs/

# ── model registry (resolves --weights names; .pt files are runtime mounts) ──
COPY models/registry.yaml models/registry.yaml

# ── runtime mounts ────────────────────────────────────────────────────────────
# /app/models  — place .pt weight files here (production, soiling_production, …)
# /app/outputs — pipeline writes arrays.geojson, risk.geojson, etc. here
# /app/.cache  — weather API response cache (~440 MB after warm-up)
VOLUME ["/app/models", "/app/outputs", "/app/.cache"]

ENTRYPOINT ["solarsoiled"]
CMD ["--help"]
