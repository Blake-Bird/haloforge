"""Assembled-workspace regression tests using explicitly synthetic saved runs."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from config.defaults import DEFAULT_PARAMS
from engine.sigma import compute_sigma_result
from state import run_storage
from state.run_model import create_run_from_current_state


ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = [
    "Dashboard",
    "Notebook",
    "Design experiment",
    "Benchmark lab",
    "Performance lab",
    "Convergence lab",
    "Known limitations",
    "Teach",
    "Learn the pipeline",
    "Graph studio",
    "Structure field",
    "Fit + window atlas",
    "Runs + export",
    "Diagnostics",
]


@pytest.fixture
def saved_app(tmp_path, monkeypatch):
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
    for index, amplitude in enumerate((1.0, 1.2)):
        params = dict(DEFAULT_PARAMS, enable_ede=False, A_s=2.1e-9 * amplitude)
        k = np.geomspace(params["k_min"], params["k_max"], params["k_points"])
        z = np.asarray(params["z_values"], dtype=float)
        power = amplitude * 1e4 * k / (1 + (k / 0.2) ** 3)
        result = {
            "k": k,
            "P": power,
            "P_by_z": power[None, :] / (1 + z[:, None]) ** 2,
            "redshifts": z,
            "growth_class": 1 / (1 + z),
            "derived": {"h": 0.6781, "sigma8": 0.8},
            "class_status": "TEST_DATA",
        }
        current = {
            "params": params,
            "power_result": result,
            "sigma_result": compute_sigma_result(result, params),
        }
        run = create_run_from_current_state(
            current, f"Synthetic run {index}", is_baseline=index == 0
        )
        run_storage.save_run(run)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)


@pytest.mark.parametrize("workspace", WORKSPACES)
def test_research_workspace_renders_saved_science_without_exception(
    saved_app, workspace
):
    saved_app.session_state["hf_primary_mode"] = "Research"
    saved_app.session_state["hf_research_workspace"] = workspace
    saved_app.run()
    assert not saved_app.exception, [
        (error.message, error.stack_trace) for error in saved_app.exception
    ]


@pytest.mark.parametrize("workspace", ["Compare lab", "Sensitivity explorer"])
def test_comparison_workspace_renders_saved_pair_without_exception(
    saved_app, workspace
):
    saved_app.session_state["hf_primary_mode"] = "Compare"
    saved_app.session_state["hf_compare_workspace"] = workspace
    saved_app.run()
    assert not saved_app.exception, [
        (error.message, error.stack_trace) for error in saved_app.exception
    ]


def test_structure_maps_keep_physical_bounds_and_amplitude_ratios(saved_app):
    import json

    saved_app.session_state["hf_primary_mode"] = "Research"
    saved_app.session_state["hf_research_workspace"] = "Structure field"
    saved_app.run()
    assert not saved_app.exception
    tables = [
        item.value for item in saved_app.dataframe if "RMS δ" in item.value.columns
    ]
    assert len(tables) == 1
    table = tables[0].set_index("run")
    assert table.loc["Synthetic run 1", "RMS δ"] / table.loc[
        "Synthetic run 0", "RMS δ"
    ] == pytest.approx(np.sqrt(1.2), rel=1e-12)
    heatmaps = 0
    for chart in saved_app.get("plotly_chart"):
        figure = json.loads(chart.proto.spec)
        if figure["data"][0]["type"] != "heatmap":
            continue
        heatmaps += 1
        for axis in ("xaxis", "yaxis"):
            assert figure["layout"][axis]["range"] == [0, 300]
            assert figure["layout"][axis]["constrain"] == "domain"
    assert heatmaps == 3


def test_startup_and_comparison_exclude_tampered_saved_run(saved_app):
    runs = run_storage.load_all_runs()
    damaged = runs[-1]
    path = run_storage.RUN_DIR / damaged["arrays_file"]
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["P"][0] *= 2
    np.savez_compressed(path, **arrays)
    run_storage.set_last_run_id(damaged["run_id"])
    saved_app.session_state["hf_primary_mode"] = "Compare"
    saved_app.session_state["hf_compare_workspace"] = "Compare lab"
    saved_app.run()
    assert not saved_app.exception
    assert saved_app.session_state["current_run_id"] == runs[0]["run_id"]
    assert any(
        "excluded" in message.value and "integrity" in message.value
        for message in saved_app.warning
    )
    assert (
        run_storage.load_run(damaged["run_id"])["integrity_status"]["state"]
        == "invalid"
    )
    assert any(
        damaged["run_id"] == run["run_id"] for run in run_storage.load_all_runs()
    )
    saved_app.session_state["hf_primary_mode"] = "Research"
    saved_app.session_state["hf_research_workspace"] = "Runs + export"
    saved_app.session_state["run_vault_selected_id"] = damaged["run_id"]
    saved_app.run()
    assert not saved_app.exception
    assert any("failed integrity" in item.value for item in saved_app.error)
    labels = {item.label for item in saved_app.button}
    assert "Delete selected run" in labels
    assert not {"Load", "Rename", "Duplicate", "Set baseline"} & labels


def test_comparison_handoff_preserves_panel_choice_without_widget_warning(saved_app):
    runs = run_storage.load_all_runs()
    saved_app.session_state["hf_primary_mode"] = "Compare"
    saved_app.session_state["hf_compare_workspace"] = "Compare lab"
    saved_app.session_state["compare_run_ids"] = [run["run_id"] for run in runs]
    saved_app.session_state["compare_baseline_id"] = runs[0]["run_id"]
    saved_app.session_state["compare_mode"] = "Percent difference"
    saved_app.session_state["compare_panel_count"] = "1"
    saved_app.run()
    assert not saved_app.exception
    assert saved_app.session_state["compare_panel_count"] == "1"
    assert not any("default value" in warning.value for warning in saved_app.warning)


def test_comparison_survives_repeated_widget_updates(saved_app):
    saved_app.session_state["hf_primary_mode"] = "Compare"
    saved_app.session_state["hf_compare_workspace"] = "Compare lab"
    saved_app.run()
    assert not saved_app.exception
    for mode in ("Ratio", "Percent difference", "Overlay"):
        saved_app.selectbox(key="compare_mode").select(mode).run()
        assert not saved_app.exception, [
            (item.message, item.stack_trace) for item in saved_app.exception
        ]
    saved_app.selectbox(key="compare_common_redshift").select(2.0).run()
    assert not saved_app.exception
    runs = run_storage.load_all_runs()
    for baseline in reversed(runs):
        saved_app.selectbox(key="compare_baseline_id").select(baseline["run_id"]).run()
        assert not saved_app.exception
        candidate = saved_app.session_state["point_compare_candidate_id"]
        assert isinstance(candidate, str)
        assert candidate != baseline["run_id"]
