#!/usr/bin/env python3
"""Run inference with a trained YOLOv11 segmentation model. Supports SAHI sliced inference."""

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.train_utils import resolve_project_dir, resolve_weights, select_device
from solarsoiled.manifest import write_manifest

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Run: pip install -r requirements.txt") from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run YOLO segmentation inference")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-visuals", dest="save_visuals", action="store_true")
    parser.add_argument("--no-save-visuals", dest="save_visuals", action="store_false")
    parser.set_defaults(save_visuals=True)
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--name", type=str, default="predict")
    parser.add_argument("--sahi", action="store_true", default=False,
                        help="Use SAHI sliced inference (pip install sahi)")
    parser.add_argument("--sahi-slice-size", type=int, default=640)
    parser.add_argument("--sahi-overlap", type=float, default=0.2)
    return parser.parse_args(argv)


def run_standard_inference(model, input_folder: Path, args, device: str) -> None:
    print(f"Running inference on: {input_folder}")
    model.predict(
        source=str(input_folder), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, device=device, save=args.save_visuals,
        save_txt=True, save_conf=True, project=args.project, name=args.name, verbose=True,
    )


def run_sahi_inference(weights_path: Path, input_folder: Path, args, device: str) -> None:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError("Run: pip install sahi") from exc

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_project_dir(args.project, repo_root) / args.name
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights_path),
        confidence_threshold=args.conf, device=device,
    )

    image_paths = sorted(list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpg")))
    print(f"SAHI inference on {len(image_paths)} images "
          f"(slice={args.sahi_slice_size}px, overlap={args.sahi_overlap})")

    for img_path in image_paths:
        img_w, img_h = PILImage.open(img_path).size
        result = get_sliced_prediction(
            str(img_path), detection_model,
            slice_height=args.sahi_slice_size, slice_width=args.sahi_slice_size,
            overlap_height_ratio=args.sahi_overlap, overlap_width_ratio=args.sahi_overlap,
            postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
            postprocess_match_threshold=args.iou, verbose=0,
        )
        lines = []
        for pred in result.object_prediction_list:
            if pred.mask is not None and pred.mask.segmentation:
                coords = pred.mask.segmentation[0]
                if len(coords) >= 6:
                    pts = []
                    for i in range(0, len(coords), 2):
                        pts.append(f"{coords[i] / img_w:.6f}")
                        pts.append(f"{coords[i+1] / img_h:.6f}")
                    lines.append(f"{pred.category.id} " + " ".join(pts))
        (labels_dir / img_path.with_suffix(".txt").name).write_text("\n".join(lines))
        if args.save_visuals:
            result.export_visuals(export_dir=str(output_dir), file_name=img_path.stem)

    print(f"SAHI complete. Labels: {labels_dir}")


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    weights_path = resolve_weights(args.weights, repo_root)

    input_folder = (
        Path(args.source).expanduser().resolve() if args.source
        else repo_root / "data" / "tiles"
    )
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    device = select_device()
    project_dir = resolve_project_dir(args.project, repo_root)

    if args.sahi:
        run_sahi_inference(weights_path, input_folder, args, device)
    else:
        args.project = str(project_dir)
        run_standard_inference(YOLO(str(weights_path), task="segment"), input_folder, args, device)

    output_dir = project_dir / args.name
    print(f"Results saved to: {output_dir}")

    image_paths = sorted(
        list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpg"))
    )
    run_tag = (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )
    write_manifest(
        output_dir,
        stage="stage1_detect",
        model_version=f"stage1-{run_tag}",
        model_weights=weights_path,
        inputs=[str(p) for p in image_paths],
        metrics={
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "n_images": len(image_paths),
        },
        known_limitations=[
            "NAIP Santa Cruz training distribution; 0.6m GSD",
            "Below 0.70 mAP50 GA bar",
        ],
        extra={"sahi": bool(args.sahi)},
    )


if __name__ == "__main__":
    main()
