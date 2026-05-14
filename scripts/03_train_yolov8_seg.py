#!/usr/bin/env python3
"""Train a YOLOv11 segmentation model on NAIP solar array tiles."""

from pathlib import Path
import argparse
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.train_utils import resolve_project_dir, select_device

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Run: pip install -r requirements.txt") from exc


RTX3060_PRESETS = {
    "laptop": {"batch": 1, "imgsz": 512, "accumulate": 1},
    "small":  {"batch": 8, "imgsz": 640, "accumulate": 1},
    "medium": {"batch": 4, "imgsz": 768, "accumulate": 2},
    "large":  {"batch": 2, "imgsz": 896, "accumulate": 4},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 segmentation model")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--auto-augment", type=str, default=None,
                        help="'none' to disable; 'randaugment'/'autoaugment'/'augmix' to set explicitly. "
                             "Omit to use the ultralytics default (randaugment).")
    parser.add_argument("--erasing", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--cos-lr", action="store_true", default=False)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--rtx3060-preset", choices=sorted(RTX3060_PRESETS), default="small")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
    return parser.parse_args()


def validate_dataset(data_yaml: Path) -> None:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_field = cfg.get("train")
    if isinstance(train_field, list) or (
        isinstance(train_field, str) and train_field.endswith(".txt")
    ):
        print("  multi-source dataset (list/txt) — structural check skipped")
        return
    root = Path(cfg.get("path", data_yaml.parent)).expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg"}
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        n_img = sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in exts) if img_dir.exists() else 0
        n_lbl = sum(1 for _ in lbl_dir.rglob("*.txt")) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")
        if n_img == 0 or n_lbl == 0:
            raise ValueError(f"Empty {split} split — check labels under {lbl_dir}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.data:
        data_yaml = Path(args.data).expanduser().resolve()
    else:
        data_yaml = repo_root / "data" / "yolo" / "naip" / "data.yaml"
        if not data_yaml.exists():
            data_yaml = repo_root / "roboflow_upload" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    print(f"Dataset: {data_yaml}")
    validate_dataset(data_yaml)

    device = select_device()
    preset = RTX3060_PRESETS[args.rtx3060_preset]
    batch = args.batch if args.batch is not None else (preset["batch"] if device == "cuda" else 1)
    imgsz = args.imgsz if args.imgsz else preset["imgsz"]
    project_dir = resolve_project_dir(None, repo_root)
    print(f"Training: preset={args.rtx3060_preset} batch={batch} imgsz={imgsz} amp={args.amp}")

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_yaml), imgsz=imgsz, epochs=args.epochs, batch=batch,
        device=device, workers=args.workers, patience=args.patience, seed=args.seed,
        amp=args.amp, degrees=args.degrees, shear=args.shear, perspective=args.perspective,
        scale=args.scale, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste, cos_lr=args.cos_lr,
        project=str(project_dir), name=args.name, pretrained=True, verbose=True,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.fraction != 1.0:
        train_kwargs["fraction"] = args.fraction
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
    if args.auto_augment is not None:
        aa = args.auto_augment.lower()
        train_kwargs["auto_augment"] = None if aa in ("none", "null", "off") else args.auto_augment
    if args.erasing is not None:
        train_kwargs["erasing"] = args.erasing
    if args.translate is not None:
        train_kwargs["translate"] = args.translate

    model.train(**train_kwargs)
    print(f"Training complete! Results saved to: {project_dir / args.name}")


if __name__ == "__main__":
    main()
