from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.train_utils import resolve_project_dir, resolve_weights


def test_resolve_project_dir_defaults_under_repo_root(tmp_path: Path):
    project = resolve_project_dir(None, tmp_path)
    assert project == tmp_path / "runs" / "segment"


def test_resolve_weights_falls_back_to_nested_run_dir(tmp_path: Path):
    weights = tmp_path / "runs" / "segment" / "runs" / "segment" / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("stub")
    assert resolve_weights(None, tmp_path) == weights


def test_resolve_weights_errors_with_checked_candidates(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checked:"):
        resolve_weights(None, tmp_path)
