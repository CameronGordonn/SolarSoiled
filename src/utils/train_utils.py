"""Shared utilities for training, inference, and evaluation scripts."""
from __future__ import annotations

from pathlib import Path


def select_device() -> str:
    import torch
    if torch.cuda.is_available():
        print("Using GPU (CUDA)")
        return "cuda"
    print("CUDA not available — using CPU")
    return "cpu"


def compute_f1(precision: float, recall: float) -> float:
    d = precision + recall
    return 2.0 * precision * recall / d if d > 0 else 0.0


def resolve_project_dir(arg: str | None, repo_root: Path, default: str = "runs/segment") -> Path:
    p = Path(arg).expanduser() if arg else repo_root / default
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def resolve_weights(arg: str | None, repo_root: Path) -> Path:
    candidates = []
    if arg:
        p = Path(arg).expanduser()
        candidates.append(p.resolve() if p.is_absolute() else (repo_root / p).resolve())
    else:
        candidates.extend([
            repo_root / "runs" / "segment" / "train" / "weights" / "best.pt",
            repo_root / "runs" / "segment" / "runs" / "segment" / "train" / "weights" / "best.pt",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Weights not found. Checked: {', '.join(str(c) for c in candidates)}")


def resolve_data_yaml(arg: str | None, repo_root: Path) -> Path:
    p = Path(arg).expanduser().resolve() if arg else repo_root / "data" / "yolo" / "naip" / "data.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {p}")
    return p
