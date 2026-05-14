"""Greedy bbox-IoU matcher for detection evaluation.

Single source of truth used by `compute_sahi_confusion_matrix.py` and the
per-detection RCA script (`05c_per_detection_rca.py`). Keeping the matcher
in one place means TP/FP/FN counts stay identical across the two tools, and
the cross-check sanity test in the Phase 1 plan stays meaningful.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


BBox = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


class MatchResult(NamedTuple):
    tp: list[tuple[int, int, float]]
    fp: list[int]
    fn: list[int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_predictions(
    preds: Sequence[BBox],
    gts: Sequence[BBox],
    iou_thresh: float = 0.5,
) -> MatchResult:
    """Greedy match — for each pred (in order), claim the unmatched GT with highest IoU.

    Returns:
      tp: list of (pred_idx, gt_idx, iou) for matched pairs
      fp: pred indices with no GT match above threshold
      fn: gt indices that no pred claimed

    Predictions should be ordered by descending confidence for "best-first"
    matching. Callers that don't care about confidence (e.g. label-only TXT
    files where conf isn't preserved) will get the same behaviour as a stable
    file-order traversal — which matches the existing
    `compute_sahi_confusion_matrix.py` loop.
    """
    matched_gt: set[int] = set()
    tps: list[tuple[int, int, float]] = []
    fps: list[int] = []
    for p_idx, pred_bbox in enumerate(preds):
        best_iou = 0.0
        best_gt: int | None = None
        for g_idx, gt_bbox in enumerate(gts):
            if g_idx in matched_gt:
                continue
            v = iou(pred_bbox, gt_bbox)
            if v > best_iou:
                best_iou = v
                best_gt = g_idx
        if best_gt is not None and best_iou >= iou_thresh:
            tps.append((p_idx, best_gt, best_iou))
            matched_gt.add(best_gt)
        else:
            fps.append(p_idx)
    fns = [i for i in range(len(gts)) if i not in matched_gt]
    return MatchResult(tps, fps, fns)
