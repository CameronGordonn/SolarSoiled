"""Tests for per-partner API key authorization (SOLARSOILED_KEYS_FILE).

Covers:
- Single-key mode (SOLARSOILED_KEYS_FILE unset) — any valid key reads any partner
- Multi-key mode — partner-bound key gets 200 for its own partner, 403 for others
- Wildcard key (partner_id: "*") — reads any partner
- Invalid key — 401 in both modes
- /jobs, /jobs/{id}, /feedback, /results/* all enforce partner checks
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

KEYS_YAML = textwrap.dedent("""\
    keys:
      sk_acme:
        partner_id: acme-solar
        name: Acme Solar
        scopes: [run, read, feedback]
      sk_other:
        partner_id: other-corp
        name: Other Corp
        scopes: [run, read, feedback]
      sk_admin:
        partner_id: "*"
        name: Internal admin
        scopes: [run, read, feedback]
""")


@pytest.fixture()
def keys_file(tmp_path: Path) -> Path:
    p = tmp_path / "keys.yaml"
    p.write_text(KEYS_YAML)
    return p


def _make_client(keys_path: str = "") -> TestClient:
    """Import api fresh with patched env so _KEY_REGISTRY is re-evaluated."""
    env = {
        "SOLARSOILED_API_KEY": "sk_single",
        "SOLARSOILED_KEYS_FILE": keys_path,
    }
    with patch.dict(os.environ, env, clear=False):
        import importlib
        import solarsoiled.api as api_mod
        importlib.reload(api_mod)
        return TestClient(api_mod.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# _resolve_caller unit tests (no HTTP)
# ---------------------------------------------------------------------------

class TestResolveCallerUnit:
    """Direct unit tests for the auth helper — no HTTP overhead."""

    def test_single_key_mode_valid_key(self):
        env = {"SOLARSOILED_API_KEY": "sk_single", "SOLARSOILED_KEYS_FILE": ""}
        with patch.dict(os.environ, env):
            import importlib
            import solarsoiled.api as api_mod
            importlib.reload(api_mod)
            # _KEY_REGISTRY should be None in single-key mode
            assert api_mod._KEY_REGISTRY is None

    def test_multi_key_mode_registry_loaded(self, keys_file: Path):
        env = {"SOLARSOILED_API_KEY": "", "SOLARSOILED_KEYS_FILE": str(keys_file)}
        with patch.dict(os.environ, env):
            import importlib
            import solarsoiled.api as api_mod
            importlib.reload(api_mod)
            assert api_mod._KEY_REGISTRY is not None
            assert "sk_acme" in api_mod._KEY_REGISTRY
            assert "sk_admin" in api_mod._KEY_REGISTRY

    def test_assert_partner_access_none_caller_passes(self):
        import importlib
        import solarsoiled.api as api_mod
        importlib.reload(api_mod)
        # Should not raise — None caller means single-key / wildcard mode
        api_mod._assert_partner_access(None, "acme-solar")

    def test_assert_partner_access_none_requested_passes(self):
        import importlib
        import solarsoiled.api as api_mod
        importlib.reload(api_mod)
        # Should not raise — ownerless job
        api_mod._assert_partner_access("acme-solar", None)

    def test_assert_partner_access_match_passes(self):
        import importlib
        import solarsoiled.api as api_mod
        importlib.reload(api_mod)
        api_mod._assert_partner_access("acme-solar", "acme-solar")

    def test_assert_partner_access_mismatch_raises(self):
        from fastapi import HTTPException
        import importlib
        import solarsoiled.api as api_mod
        importlib.reload(api_mod)
        with pytest.raises(HTTPException) as exc_info:
            api_mod._assert_partner_access("acme-solar", "other-corp")
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# HTTP integration tests — /results endpoints
# ---------------------------------------------------------------------------

class TestResultsAuth:

    def test_single_key_mode_valid_key_reads_any_partner(self, tmp_path: Path):
        # Create a dummy risk_map.html so the endpoint doesn't 404
        out_dir = tmp_path / "outputs" / "aoi" / "acme-solar"
        out_dir.mkdir(parents=True)
        (out_dir / "risk_map.html").write_text("<html/>")

        client = _make_client(keys_path="")

        with patch("solarsoiled.api.AoiPaths") as mock_paths:
            mock_paths.return_value.root = out_dir
            resp = client.get(
                "/results/acme-solar/map",
                headers={"X-API-Key": "sk_single"},
            )
        # 200 or 404 (file serve depends on AoiPaths impl) but NOT 401/403
        assert resp.status_code not in (401, 403)

    def test_single_key_mode_invalid_key_rejected(self):
        client = _make_client(keys_path="")
        resp = client.get(
            "/results/acme-solar/map",
            headers={"X-API-Key": "wrong_key"},
        )
        assert resp.status_code == 401

    def test_multi_key_own_partner_not_403(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        # The file won't exist so we expect 404, but NOT 403/401
        resp = client.get(
            "/results/acme-solar/map",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 404

    def test_multi_key_cross_partner_forbidden(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/results/other-corp/map",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 403

    def test_multi_key_wildcard_reads_any_partner(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        # File won't exist — expect 404 not 403/401
        resp = client.get(
            "/results/acme-solar/map",
            headers={"X-API-Key": "sk_admin"},
        )
        assert resp.status_code == 404

        resp2 = client.get(
            "/results/other-corp/arrays",
            headers={"X-API-Key": "sk_admin"},
        )
        assert resp2.status_code == 404

    def test_multi_key_invalid_key_rejected(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/results/acme-solar/map",
            headers={"X-API-Key": "sk_bogus"},
        )
        assert resp.status_code == 401

    def test_arrays_endpoint_partner_check(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/results/other-corp/arrays",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 403

    def test_recommendations_endpoint_partner_check(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/results/other-corp/recommendations",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# HTTP integration tests — /feedback
# ---------------------------------------------------------------------------

class TestFeedbackAuth:

    def _feedback_payload(self, partner_id: str) -> dict:
        return {
            "array_id": 1,
            "partner_id": partner_id,
            "cleaned_at": "2026-01-15",
            "pre_clean_kwh_7d": 100.0,
            "post_clean_kwh_7d": 120.0,
        }

    def test_own_partner_feedback_not_403(self, keys_file: Path, tmp_path: Path):
        client = _make_client(keys_path=str(keys_file))
        with patch("solarsoiled.api.AoiPaths") as mock_paths:
            mock_paths.return_value.ensure_root.return_value = None
            mock_paths.return_value.feedback_json = tmp_path / "feedback.json"
            resp = client.post(
                "/feedback",
                json=self._feedback_payload("acme-solar"),
                headers={"X-API-Key": "sk_acme"},
            )
        assert resp.status_code not in (401, 403)

    def test_cross_partner_feedback_forbidden(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.post(
            "/feedback",
            json=self._feedback_payload("other-corp"),
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 403

    def test_invalid_key_feedback_rejected(self, keys_file: Path):
        client = _make_client(keys_path=str(keys_file))
        resp = client.post(
            "/feedback",
            json=self._feedback_payload("acme-solar"),
            headers={"X-API-Key": "sk_bogus"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# HTTP integration tests — /jobs/{job_id} ownership
# ---------------------------------------------------------------------------

class TestJobOwnership:

    def test_own_job_accessible(self, keys_file: Path):
        from solarsoiled.jobs import JobRecord, _store
        record = JobRecord(job_id="job-acme-1", partner_id="acme-solar")
        _store["job-acme-1"] = record

        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/jobs/job-acme-1",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 200

    def test_cross_partner_job_forbidden(self, keys_file: Path):
        from solarsoiled.jobs import JobRecord, _store
        record = JobRecord(job_id="job-other-1", partner_id="other-corp")
        _store["job-other-1"] = record

        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/jobs/job-other-1",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 403

    def test_wildcard_key_reads_any_job(self, keys_file: Path):
        from solarsoiled.jobs import JobRecord, _store
        record = JobRecord(job_id="job-acme-2", partner_id="acme-solar")
        _store["job-acme-2"] = record

        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/jobs/job-acme-2",
            headers={"X-API-Key": "sk_admin"},
        )
        assert resp.status_code == 200

    def test_ownerless_job_accessible_by_any_key(self, keys_file: Path):
        from solarsoiled.jobs import JobRecord, _store
        record = JobRecord(job_id="job-noowner", partner_id=None)
        _store["job-noowner"] = record

        client = _make_client(keys_path=str(keys_file))
        resp = client.get(
            "/jobs/job-noowner",
            headers={"X-API-Key": "sk_acme"},
        )
        assert resp.status_code == 200
