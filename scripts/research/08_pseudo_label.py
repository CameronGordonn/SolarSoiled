"""Pseudo-label unlabeled NAIP tiles with the current best YOLO model.

Semi-supervised loop: run model on new tiles → review report → merge staging into
data/yolo/naip/ and retrain. Repeat with improved model. See docs/PHASE1_HANDOFF.md.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.train_utils import resolve_weights, select_device


def polygon_to_yolo_line(class_id: int, polygon_xyn: np.ndarray) -> str:
    pts = polygon_xyn.flatten().tolist()
    return f"{class_id} " + " ".join(f"{p:.6f}" for p in pts)


def predict_standard(model, img_path: Path, conf: float, iou: float, device: str) -> list[dict]:
    results = model.predict(str(img_path), conf=conf, iou=iou, device=device, verbose=False)
    r = results[0]
    preds = []
    if r.masks is not None:
        for polygon, box_conf, cls in zip(r.masks.xyn, r.boxes.conf, r.boxes.cls):
            if len(polygon) >= 3:
                preds.append({"line": polygon_to_yolo_line(int(cls), polygon), "conf": float(box_conf)})
    return preds


def predict_sahi(detection_model, img_path: Path, conf: float, iou: float, slice_size: int, overlap: float) -> list[dict]:
    from sahi.predict import get_sliced_prediction
    from PIL import Image as PILImage

    img_w, img_h = PILImage.open(img_path).size
    result = get_sliced_prediction(
        str(img_path), detection_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
        postprocess_match_threshold=iou, verbose=0,
    )
    preds = []
    for pred in result.object_prediction_list:
        if pred.score.value < conf:
            continue
        if pred.mask is not None and pred.mask.segmentation:
            coords = pred.mask.segmentation[0]
            if len(coords) >= 6:
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append(f"{coords[i] / img_w:.6f}")
                    pts.append(f"{coords[i+1] / img_h:.6f}")
                preds.append({"line": f"{pred.category.id} " + " ".join(pts), "conf": pred.score.value})
    return preds


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Pseudo-label unlabeled NAIP tiles with the current best model.")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--tiles-dir", type=Path, default=repo / "data/tiles")
    p.add_argument("--output-dir", type=Path, default=repo / "data/pseudo_labeled")
    p.add_argument("--min-conf", type=float, default=0.50)
    p.add_argument("--auto-accept-conf", type=float, default=0.75)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--sahi", action="store_true")
    p.add_argument("--sahi-slice-size", type=int, default=640)
    p.add_argument("--sahi-overlap", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    weights_path = resolve_weights(args.weights, repo)

    if not args.tiles_dir.exists():
        print(f"ERROR: tiles-dir not found: {args.tiles_dir}", file=sys.stderr)
        sys.exit(1)

    device = select_device()

    staging_img = args.output_dir / "staging" / "images"
    staging_lbl = args.output_dir / "staging" / "labels"
    review_img  = args.output_dir / "review"  / "images"
    review_lbl  = args.output_dir / "review"  / "labels"
    for d in (staging_img, staging_lbl, review_img, review_lbl):
        d.mkdir(parents=True, exist_ok=True)

    if args.sahi:
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            print("ERROR: pip install sahi", file=sys.stderr)
            sys.exit(1)
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights_path),
            confidence_threshold=args.min_conf, device=device,
        )
        yolo_model = None
    else:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: pip install ultralytics", file=sys.stderr)
            sys.exit(1)
        yolo_model = YOLO(str(weights_path), task="segment")
        sahi_model = None

    image_paths = sorted(list(args.tiles_dir.glob("*.png")) + list(args.tiles_dir.glob("*.jpg")))
    if not image_paths:
        print(f"No images found in {args.tiles_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} tiles...")
    report_rows = []
    n_auto = n_review = n_skip = 0

    for img_path in image_paths:
        if args.sahi:
            preds = predict_sahi(sahi_model, img_path, args.min_conf, args.iou,
                                 args.sahi_slice_size, args.sahi_overlap)
        else:
            preds = predict_standard(yolo_model, img_path, args.min_conf, args.iou, device)

        if not preds:
            action = "skip"
            n_skip += 1
        elif all(p["conf"] >= args.auto_accept_conf for p in preds):
            action = "auto_accept"
            n_auto += 1
        else:
            action = "review"
            n_review += 1

        label_text = "\n".join(p["line"] for p in preds)
        label_name = img_path.with_suffix(".txt").name
        if action == "auto_accept":
            shutil.copy2(img_path, staging_img / img_path.name)
            (staging_lbl / label_name).write_text(label_text)
        elif action == "review":
            shutil.copy2(img_path, review_img / img_path.name)
            (review_lbl / label_name).write_text(label_text)

        confs = [p["conf"] for p in preds] if preds else []
        report_rows.append({
            "tile": img_path.name, "n_predictions": len(preds),
            "mean_conf": f"{np.mean(confs):.3f}" if confs else "",
            "max_conf": f"{max(confs):.3f}" if confs else "",
            "recommended_action": action,
        })

    report_path = args.output_dir / "pseudo_label_report.csv"
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile", "n_predictions", "mean_conf", "max_conf", "recommended_action"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"Done. auto_accept={n_auto}  review={n_review}  skip={n_skip}")
    print(f"Staging: {staging_img}\nReview:  {review_img}\nReport:  {report_path}")


if __name__ == "__main__":
    main()
