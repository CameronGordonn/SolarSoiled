#!/bin/bash

# Setup conda environment for NAIP + Roboflow pipeline
# Usage: source setup_conda.sh

set -e

echo "=== NAIP + Roboflow Pipeline Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Conda/Miniconda first."
    exit 1
fi

# Create or activate environment
ENV_NAME="solar-soiling"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Found existing environment: $ENV_NAME"
    echo "Activating: conda activate $ENV_NAME"
else
    echo "Creating environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.11
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo ""
echo "=== Installing Geospatial Stack (via conda) ==="
# Install geospatial packages via conda (better for GDAL/PROJ dependencies)
conda install -y -c conda-forge \
    geopandas \
    shapely \
    pyproj \
    geojson \
    scikit-image \
    pyyaml

echo ""
echo "=== Installing GeoAI (optional but useful) ==="
# GeoAI provides STAC search/download, SAM, and visualization tools
# Prefer conda-forge; if it fails the user can fall back to pip later
if ! conda install -y -c conda-forge geoai; then
    echo "⚠ Warning: could not install geoai via conda."
    echo "  You can still install it later with: pip install geoai-py"
fi

echo ""
echo "=== Installing ML/Data Science Stack (via conda) ==="
conda install -y \
    numpy \
    pandas \
    scikit-learn \
    pytorch::pytorch \
    pytorch::pytorch-cuda=11.8

echo ""
echo "=== Installing Dataset Management (via pip) ==="
pip install --upgrade \
    ultralytics \
    roboflow \
    xgboost

echo ""
echo "=== Verifying Installation ==="
python -c "
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import shapely
import torch
import sklearn
import xgboost as xgb
from ultralytics import YOLO
import roboflow

print('✓ numpy:', np.__version__)
print('✓ pandas:', pd.__version__)
print('✓ rasterio:', rasterio.__version__)
print('✓ geopandas:', gpd.__version__)
print('✓ shapely:', shapely.__version__)
print('✓ torch:', torch.__version__)
print('✓ sklearn:', sklearn.__version__)
print('✓ xgboost:', xgb.__version__)
print('✓ ultralytics (YOLO): OK')
print('✓ roboflow: OK')
print('')
print('✅ All dependencies installed successfully!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use this environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Then run the pipeline:"
echo "  python scripts/02_tile_naip_image.py"
echo "  python scripts/02b_export_to_roboflow.py"
echo "  python scripts/02c_import_from_roboflow.py"
echo "  python scripts/03_train_yolov8_seg.py"
