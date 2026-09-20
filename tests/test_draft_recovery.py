"""Startup must preserve valid controls and recover visibly from invalid drafts."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config.defaults import DEFAULT_PARAMS
from state import run_storage


@pytest.fixture
def draft_app(tmp_path, monkeypatch):
    monkeypatch.setenv("HALOFORGE_DEPLOYMENT", "local")
    monkeypatch.setenv("HALOFORGE_DATA_DIR", str(tmp_path))
    for name, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
        "LAST_RUN_PATH": tmp_path / "state/last_run.json",
        "DRAFT_PARAMS_PATH": tmp_path / "state/draft_params.json",
    }.items():
        monkeypatch.setattr(run_storage, name, path)
    run_storage.DRAFT_PARAMS_PATH.parent.mkdir()
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    )
    app.session_state["hf_primary_mode"] = "Research"
    return app


@pytest.mark.parametrize(
    "draft",
    [
        "{broken",
        "[]",
        '{"H0": 0}',
        '{"window_type": "missing"}',
        '{"enable_ede": false, "log10_a_c": 400}',
    ],
)
def test_invalid_draft_recovers_visibly_without_overwriting_source(draft_app, draft):
    run_storage.DRAFT_PARAMS_PATH.write_text(draft)
    draft_app.run()
    assert not draft_app.exception
    assert any(
        "saved draft could not be restored" in warning.value
        for warning in draft_app.warning
    )
    assert draft_app.session_state["params"]["H0"] == DEFAULT_PARAMS["H0"]
    assert run_storage.DRAFT_PARAMS_PATH.read_text() == draft


def test_valid_draft_preserves_custom_values_outside_slider_presets(draft_app):
    overrides = {
        "H0": 85.0,
        "n_s": 1.1,
        "n_EDE": 1,
        "mass_min_exp": 7.5,
        "z_values": [0.0, 3.25],
        "delta_halo": 4000.0,
    }
    draft = json.dumps(dict(DEFAULT_PARAMS, **overrides))
    run_storage.DRAFT_PARAMS_PATH.write_text(draft)
    draft_app.run()
    assert not draft_app.exception
    for key, value in overrides.items():
        assert draft_app.session_state["params"][key] == value, key
    assert not draft_app.session_state["draft_recovery_issues"]
    assert run_storage.DRAFT_PARAMS_PATH.read_text() == draft
