"""Unit tests for solarsoiled.registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from solarsoiled.registry import RegistryError, load_registry, resolve


def _write_registry(tmp_path: Path, models: dict, aliases: dict | None = None) -> Path:
    payload = {"schema_version": 1, "models": models, "aliases": aliases or {}}
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _fake_weights(tmp_path: Path, name: str = "fake.pt") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x00" * 64)  # any non-empty file is fine
    return p


def test_resolve_registered_name(tmp_path: Path):
    weights = _fake_weights(tmp_path)
    registry_path = _write_registry(
        tmp_path,
        models={
            "stage1-test": {
                "path": str(weights),
                "stage": "stage1",
                "beta": True,
                "map50_test": 0.7,
                "known_limitations": ["unit-test only"],
            }
        },
    )
    resolved = resolve("stage1-test", registry_path=registry_path)
    assert resolved.model_version == "stage1-test"
    assert resolved.path == weights.resolve()
    assert resolved.stage == "stage1"
    assert resolved.metrics == {"map50_test": 0.7}
    assert resolved.known_limitations == ("unit-test only",)
    assert resolved.source == "registered"


def test_resolve_alias(tmp_path: Path):
    weights = _fake_weights(tmp_path)
    registry_path = _write_registry(
        tmp_path,
        models={"stage1-test": {"path": str(weights), "stage": "stage1", "beta": True}},
        aliases={"production": "stage1-test"},
    )
    resolved = resolve("production", registry_path=registry_path)
    assert resolved.model_version == "stage1-test"
    assert resolved.source == "alias"


def test_resolve_unknown_name_raises(tmp_path: Path):
    weights = _fake_weights(tmp_path)
    registry_path = _write_registry(
        tmp_path,
        models={"stage1-test": {"path": str(weights), "stage": "stage1", "beta": True}},
    )
    with pytest.raises(RegistryError):
        resolve("does-not-exist", registry_path=registry_path)


def test_resolve_passthrough_path(tmp_path: Path):
    """A real filesystem path resolves to ad-hoc passthrough even with no registry."""
    weights = _fake_weights(tmp_path, "ad_hoc.pt")
    resolved = resolve(weights)
    assert resolved.source == "ad-hoc"
    assert resolved.model_version.startswith("ad-hoc:")
    assert len(resolved.model_version) == len("ad-hoc:") + 12
    assert resolved.path == weights.resolve()


def test_load_registry_rejects_dangling_alias(tmp_path: Path):
    weights = _fake_weights(tmp_path)
    registry_path = _write_registry(
        tmp_path,
        models={"a": {"path": str(weights), "stage": "stage1", "beta": True}},
        aliases={"production": "missing-model"},
    )
    with pytest.raises(RegistryError):
        load_registry(registry_path)


def test_resolve_missing_weights_file_raises(tmp_path: Path):
    registry_path = _write_registry(
        tmp_path,
        models={
            "broken": {"path": str(tmp_path / "does_not_exist.pt"), "stage": "stage1", "beta": True}
        },
    )
    with pytest.raises(RegistryError):
        resolve("broken", registry_path=registry_path)


def test_default_registry_resolves_production():
    """The shipped models/registry.yaml resolves 'production' to a real file."""
    resolved = resolve("production")
    assert resolved.path.is_file()
    assert resolved.path.suffix == ".pt"
    assert resolved.source == "alias"
    assert resolved.model_version == "stage1-v0.5-baseline"
