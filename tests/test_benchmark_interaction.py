"""Interaction regression test for the saved-run benchmark workflow."""

from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from config.defaults import DEFAULT_PARAMS
from engine.sigma import compute_sigma_result
from state import run_storage
from state.run_model import create_run_from_current_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark_app(tmp_path, monkeypatch):
    monkeypatch.setenv("HALOFORGE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HALOFORGE_DEPLOYMENT", "local")
    for name, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
        "LAST_RUN_PATH": tmp_path / "state" / "last_run.json",
        "DRAFT_PARAMS_PATH": tmp_path / "state" / "draft_params.json",
    }.items():
        monkeypatch.setattr(run_storage, name, path)

    params = dict(DEFAULT_PARAMS, enable_ede=False, H0=70.0)
    k = np.geomspace(params["k_min"], params["k_max"], params["k_points"])
    z = np.asarray(params["z_values"], dtype=float)
    power = 1e4 * k / (1 + (k / 0.2) ** 3)
    result = {
        "k": k,
        "P": power,
        "P_by_z": power[None, :] / (1 + z[:, None]) ** 2,
        "redshifts": z,
        "growth_class": 1 / (1 + z),
        "derived": {"h": 0.7, "sigma8": 0.8},
        "class_status": "TEST_DATA",
    }
    current = {
        "params": params,
        "power_result": result,
        "sigma_result": compute_sigma_result(result, params),
    }
    run = create_run_from_current_state(current, "Benchmark test run", is_baseline=True)
    run_storage.save_run(run)
    run_storage.set_last_run_id(run["run_id"])

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    return app, run["run_id"]


def test_saved_run_benchmark_workflow_click_through(benchmark_app):
    app, run_id = benchmark_app
    app.session_state["hf_primary_mode"] = "Research"
    app.session_state["hf_research_workspace"] = "Benchmark lab"
    app.session_state["current_run_id"] = run_id
    app.session_state["loaded_run_id"] = run_id
    app.run()
    assert not app.exception, [
        (item.message, item.stack_trace) for item in app.exception
    ]

    benchmark_button = None
    for btn in app.button:
        if btn.label == "Benchmark this run":
            benchmark_button = btn
            break

    assert benchmark_button is not None, (
        "Benchmark button must be present in Benchmark lab"
    )
    benchmark_button.click().run()

    assert not app.exception, [
        (item.message, item.stack_trace) for item in app.exception
    ]
    stored = run_storage.load_run(run_id)
    assert stored is not None
    assert "benchmarks" in stored
    assert len(stored["benchmarks"]) > 0
    assert "scientific_validity" in stored
