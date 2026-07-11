import numpy as np

from config.defaults import DEFAULT_PARAMS
from state.run_model import auto_run_name
from state import run_storage


def test_auto_run_name_baseline_and_high_as():
    params = dict(DEFAULT_PARAMS)
    params["enable_ede"] = False
    assert auto_run_name(params, 1) == "Run 001 — Baseline LCDM"
    params["A_s"] = 2.6e-9
    assert auto_run_name(params, 2) == "Run 002 — High A_s"


def test_unique_run_name_handles_duplicates():
    existing = [{"name": "Run 001 — Baseline LCDM"}, {"name": "Run 001 — Baseline LCDM (2)"}]
    assert run_storage.unique_run_name("Run 001 — Baseline LCDM", existing) == "Run 001 — Baseline LCDM (3)"


def test_generate_run_exports_and_reload_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    monkeypatch.setattr(run_storage, "EXPORT_DIR", tmp_path / "exports")
    run = {
        "run_id": "test-run",
        "name": "Run 001 — Baseline LCDM",
        "run_name": "Run 001 — Baseline LCDM",
        "created_at": "2026-07-07T22:51:00+00:00",
        "updated_at": "2026-07-07T22:51:00+00:00",
        "class_status": "AXICLASS",
        "is_baseline": True,
        "visible": True,
        "color": "#38BDF8",
        "notes": "",
        "params": dict(DEFAULT_PARAMS),
        "derived": {"h": 0.6781},
        "arrays": {
            "k": np.array([0.01, 0.1]),
            "P": np.array([10.0, 2.0]),
            "Delta2": np.array([1.0, 2.0]),
            "M_h": np.array([1e10, 1e11]),
            "M": np.array([1.5e10, 1.5e11]),
            "R": np.array([1.0, 2.0]),
            "sigma": np.array([2.0, 1.0]),
            "dlnsigma_dlnM": np.array([-0.2, -0.2]),
            "hmf_press_schechter_z0": np.array([1e-2, 1e-4]),
            "hmf_sheth_tormen_z0": np.array([2e-2, 2e-4]),
            "cumulative_press_schechter_z0": np.array([1e-2, 1e-4]),
            "cumulative_sheth_tormen_z0": np.array([2e-2, 2e-4]),
        },
        "exports": {},
        "hash": "abc",
    }
    run_storage.save_run(run)
    run_storage.generate_run_exports(run)
    loaded = run_storage.load_run("test-run")
    assert loaded is not None
    assert np.allclose(loaded["arrays"]["k"], [0.01, 0.1])
    export_dir = run_storage.run_export_dir("test-run")
    assert (export_dir / "params.json").exists()
    assert (export_dir / "power_spectrum.csv").exists()
    assert (export_dir / "sigma.csv").exists()
    assert (export_dir / "hmf.csv").exists()
    assert (export_dir / "run_summary.md").exists()
    assert (export_dir / "Run_001_—_Baseline_LCDM_exports.zip").exists()


def test_cache_clear_does_not_delete_saved_runs(tmp_path, monkeypatch):
    from state import cache

    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    run = {
        "run_id": "saved",
        "name": "Saved Run",
        "params": dict(DEFAULT_PARAMS),
        "derived": {},
        "arrays": {"k": np.array([0.1]), "P": np.array([1.0])},
    }
    run_storage.save_run(run)
    cache.CACHE_DIR.mkdir(parents=True)
    (cache.CACHE_DIR / "temporary.npz").write_bytes(b"cache")
    cache.clear_cache()
    assert run_storage.load_run("saved") is not None
