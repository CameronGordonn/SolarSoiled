# RTX 3060 Setup & Training

Setup + training reference for the RTX 3060 (12 GB VRAM). For non-GPU workflows or general pipeline docs see `docs/NAIP_ROBOFLOW_WORKFLOW.md` and `docs/PHASE1_HANDOFF.md`.

## Setup

```bash
# Verify driver — need 546.x+
nvidia-smi

# Create env (CUDA 11.8)
conda create -n solar-soiling python=3.11 \
  pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate solar-soiling
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA GeForce RTX 3060
```

If CUDA reports `False`: confirm `nvidia-smi` works, then `pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`.

## Presets

Defined in `configs/yolo/experiments.yaml`:

| Preset | Batch | imgsz | Use when |
|---|---|---|---|
| `small`  | 8 | 640 | Default — yolo11s, ~9 GB VRAM |
| `medium` | 4 | 768 | Higher accuracy run |
| `laptop` | 1 | 512 | OOM fallback or CPU |

Memory rough cuts at batch=4, imgsz=640: yolo11n ~8 GB, yolo11s ~9 GB (safe), yolo11m ~11 GB (drop to batch=2). yolo11l OOMs.

## Standard training run

```bash
python scripts/03_train_yolov8_seg.py \
  --model yolo11s-seg.pt --epochs 50 --batch 4 --imgsz 640 \
  --patience 15 --device 0
```

~2.5–3 hr for 50 epochs on the 174-tile NAIP train set. Then:

```bash
python scripts/05b_eval_threshold_sweep.py --weights runs/segment/<run>/weights/best.pt
python scripts/04_infer_yolov8_seg.py
python scripts/06_export_polygons_geojson.py
```

For experiment matrices use `scripts/03b_train_experiment_matrix.py --config configs/yolo/experiments.yaml`. For the active Stage 1 retrain runbook see `docs/PHASE1_HANDOFF.md`.

## Monitoring

```bash
watch -n 1 nvidia-smi
```

Healthy: GPU util 80–95%, mem 9–11 GB for yolo11s, temps < 80 °C. Thermal throttling roughly halves throughput — if you see it, drop batch.

## Troubleshooting

**CUDA OOM:** drop batch first (`--batch 2`), then imgsz (`--imgsz 512`), then model (`yolo11n-seg.pt`). Restart the Python process after an OOM — cached allocations don't always free cleanly.

**Slow training (< 50 it/s):** GPU likely not initialized. Check `python -c "import torch; print(torch.cuda.is_available())"` returns True from inside the active env.

**Driver/CUDA mismatch:** reinstall PyTorch with the matching cuda spec (see Setup).

**Jupyter can't see GPU:** the kernel is probably the system Python, not `solar-soiling`. Restart Jupyter from inside the activated env.

**Mixed precision:** `amp=False` is the default. RTX 3060 has fp16 cores but the speedup is small here and AMP has occasionally produced unstable losses on this dataset — leave it off unless you specifically need batch headroom.
