import numpy as np
import json
import pytest
from zipfile import ZipFile
from pypdf import PdfReader
import pyarrow.parquet as pq

from config.defaults import DEFAULT_PARAMS
from state.run_model import auto_run_name
from state import run_storage
from state.provenance import reproducibility_hash, run_provenance
from state.bundle import BundleValidationError, plan_workspace_import


def test_auto_run_name_baseline_and_high_as():
    params = dict(DEFAULT_PARAMS)
    params["enable_ede"] = False
    assert auto_run_name(params, 1) == "Run 001 — Baseline LCDM"
    params["A_s"] = 2.6e-9
    assert auto_run_name(params, 2) == "Run 002 — High A_s"


def test_unique_run_name_handles_duplicates():
    existing = [
        {"name": "Run 001 — Baseline LCDM"},
        {"name": "Run 001 — Baseline LCDM (2)"},
    ]
    assert (
        run_storage.unique_run_name("Run 001 — Baseline LCDM", existing)
        == "Run 001 — Baseline LCDM (3)"
    )


@pytest.mark.parametrize("run_id", ["../outside", "/tmp/outside", "a/b", ".."])
def test_storage_rejects_path_like_run_identifiers(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    monkeypatch.setattr(run_storage, "EXPORT_DIR", tmp_path / "exports")
    with pytest.raises(ValueError, match="safe vault token"):
        run_storage.delete_run(run_id)
    assert not list(tmp_path.rglob("*"))


def test_deleted_run_is_moved_to_trash_and_can_be_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    monkeypatch.setattr(run_storage, "EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(run_storage, "TRASH_DIR", tmp_path / "trash")
    run = {
        "run_id": "restore-me",
        "name": "Restorable run",
        "params": dict(DEFAULT_PARAMS),
        "derived": {"h": 0.7},
        "arrays": {"k": np.array([0.1]), "P": np.array([1.0])},
    }
    run_storage.save_run(run)
    export_dir = run_storage.run_export_dir("restore-me")
    export_dir.mkdir(parents=True)
    (export_dir / "result.txt").write_text("evidence", encoding="utf-8")
    deletion_id = run_storage.delete_run("restore-me")
    assert deletion_id
    assert run_storage.load_run("restore-me") is None
    assert (
        run_storage.TRASH_DIR / deletion_id / "saved_runs" / "restore-me.json"
    ).is_file()
    assert run_storage.restore_deleted_run(deletion_id) == "restore-me"
    assert run_storage.load_run("restore-me") is not None
    assert (export_dir / "result.txt").read_text(encoding="utf-8") == "evidence"


def test_save_uses_a_new_array_generation_before_repointing_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    run = {
        "run_id": "generation-test",
        "name": "Generation test",
        "params": {},
        "arrays": {"P": np.array([1.0])},
    }
    run_storage.save_run(run)
    first = json.loads((run_storage.RUN_DIR / "generation-test.json").read_text())
    run["arrays"] = {"P": np.array([2.0])}
    run_storage.save_run(run)
    second = json.loads((run_storage.RUN_DIR / "generation-test.json").read_text())
    assert first["arrays_file"] != second["arrays_file"]
    with np.load(run_storage.RUN_DIR / first["arrays_file"], allow_pickle=False) as old:
        np.testing.assert_array_equal(old["P"], [1.0])
    assert np.allclose(run_storage.load_run("generation-test")["arrays"]["P"], [2.0])


@pytest.mark.parametrize(
    "run_id, arrays_file",
    [("../outside", None), ("safe-run", "../../sentinel.npz")],
)
def test_import_rejects_saved_run_metadata_path_traversal(
    run_id, arrays_file, tmp_path
):
    from io import BytesIO
    from zipfile import ZipFile

    document = {"run_id": run_id, "params": {}}
    if arrays_file is not None:
        document["arrays_file"] = arrays_file
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("saved_runs/innocent.json", json.dumps(document))
    with pytest.raises(BundleValidationError):
        plan_workspace_import(payload.getvalue(), tmp_path / "vault")
    assert not (tmp_path / "vault").exists()


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
    assert (export_dir / "power_spectrum.parquet").exists()
    assert (export_dir / "sigma.csv").exists()
    assert (export_dir / "sigma.parquet").exists()
    assert (export_dir / "hmf.csv").exists()
    assert (export_dir / "hmf.parquet").exists()
    assert (export_dir / "run_summary.md").exists()
    assert (export_dir / "provenance.json").exists()
    assert (export_dir / "notebook.json").exists()
    assert (export_dir / "audit_trail.json").exists()
    assert (export_dir / "benchmarks.json").exists()
    assert (export_dir / "migration.json").exists()
    assert (export_dir / "performance_benchmarks.json").exists()
    assert (export_dir / "scientific_validity.json").exists()
    assert (export_dir / "integrity.json").exists()
    assert (export_dir / "citation_metadata.json").exists()
    assert (export_dir / "data_dictionary.json").exists()
    assert (export_dir / "run_report.pdf").exists()
    assert (export_dir / "matter_power.pdf").exists()
    assert (export_dir / "matter_power.svg").exists()
    assert (export_dir / "matter_power.png").exists()
    assert (export_dir / "matter_power_grayscale.pdf").exists()
    assert (export_dir / "matter_power_grayscale.svg").exists()
    assert (export_dir / "matter_power_grayscale.png").exists()
    assert (export_dir / "mass_variance.pdf").exists()
    assert (export_dir / "mass_variance.svg").exists()
    assert (export_dir / "mass_variance.png").exists()
    assert (export_dir / "mass_variance_grayscale.pdf").exists()
    assert (export_dir / "mass_variance_grayscale.svg").exists()
    assert (export_dir / "mass_variance_grayscale.png").exists()
    assert (export_dir / "analytic_hmf_reference.pdf").exists()
    assert (export_dir / "analytic_hmf_reference.svg").exists()
    assert (export_dir / "analytic_hmf_reference.png").exists()
    assert (export_dir / "analytic_hmf_reference_grayscale.pdf").exists()
    assert (export_dir / "analytic_hmf_reference_grayscale.svg").exists()
    assert (export_dir / "analytic_hmf_reference_grayscale.png").exists()
    report = PdfReader(export_dir / "run_report.pdf")
    assert len(report.pages) >= 1
    assert "HaloForge" in report.pages[0].extract_text()
    assert (export_dir / "README.md").exists()
    assert (export_dir / "share_card.md").exists()
    assert (export_dir / "recreate.py").exists()
    assert (export_dir / "load_export.jl").exists()
    assert "Recreate" in (export_dir / "README.md").read_text()
    assert (
        len(__import__("pandas").read_parquet(export_dir / "power_spectrum.parquet"))
        == 2
    )
    power_metadata = pq.read_schema(export_dir / "power_spectrum.parquet").metadata
    assert power_metadata[b"haloforge.schema_version"] == b"haloforge-table-metadata-v1"
    assert power_metadata[b"haloforge.table"] == b"linear_matter_power"
    assert json.loads(power_metadata[b"haloforge.units"])["P_Mpc^3"] == "Mpc^3"
    assert json.loads(power_metadata[b"haloforge.descriptions"])["Delta2"].startswith(
        "Dimensionless"
    )
    assert power_metadata[b"haloforge.provenance_file"] == b"provenance.json"
    assert power_metadata[b"haloforge.validity_file"] == b"scientific_validity.json"
    sigma_metadata = pq.read_schema(export_dir / "sigma.parquet").metadata
    assert sigma_metadata[b"haloforge.table"] == b"mass_variance"
    assert json.loads(sigma_metadata[b"haloforge.units"])["sigma"] == "dimensionless"
    hmf_metadata = pq.read_schema(export_dir / "hmf.parquet").metadata
    assert (
        hmf_metadata[b"haloforge.table"] == b"analytic_halo_mass_function_references_z0"
    )
    assert (
        json.loads(hmf_metadata[b"haloforge.units"])["hmf_press_schechter_z0"]
        == "h^3 Mpc^-3"
    )
    assert hmf_metadata[b"haloforge.mass_definition"] == b"analytic_top_hat"
    assert (
        b"not a universal empirical HMF calibration"
        in hmf_metadata[b"haloforge.hmf_scope_limit"]
    )
    compile((export_dir / "recreate.py").read_text(), "recreate.py", "exec")
    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "Run_001_—_Baseline_LCDM_exports.zip").exists()
    manifest = json.loads((export_dir / "manifest.json").read_text())
    assert "params.json" in manifest["files"]
    assert "solver_execution.json" in manifest["files"]
    assert "solver_stdout.txt" in manifest["files"]
    assert "solver_stderr.txt" in manifest["files"]
    citations = json.loads((export_dir / "citation_metadata.json").read_text())
    assert "matter_power.pdf" in citations["figures"]
    assert (
        citations["figures"]["matter_power.svg"]
        == citations["figures"]["matter_power.pdf"]
    )
    assert (
        citations["figures"]["matter_power_grayscale.pdf"]
        == citations["figures"]["matter_power.pdf"]
    )
    dictionary = json.loads((export_dir / "data_dictionary.json").read_text())
    assert dictionary["schema_version"] == "haloforge-data-dictionary-v1"
    power_table = next(
        table
        for table in dictionary["tables"]
        if table["table"] == "linear_matter_power"
    )
    assert power_table["files"] == ["power_spectrum.csv", "power_spectrum.parquet"]
    assert (
        next(
            column for column in power_table["columns"] if column["name"] == "P_Mpc^3"
        )["unit"]
        == "Mpc^3"
    )
    hmf_table = next(
        table
        for table in dictionary["tables"]
        if table["table"] == "analytic_halo_mass_function_references_z0"
    )
    assert hmf_table["table_context"]["mass_definition"] == "analytic_top_hat"
    with ZipFile(export_dir / "Run_001_—_Baseline_LCDM_exports.zip") as archive:
        assert "manifest.json" in archive.namelist()
        assert "solver_execution.json" in archive.namelist()
        assert "solver_stdout.txt" in archive.namelist()
        assert "solver_stderr.txt" in archive.namelist()
        assert "README.md" in archive.namelist()
        assert "share_card.md" in archive.namelist()
        assert "notebook.json" in archive.namelist()
        assert "audit_trail.json" in archive.namelist()
        assert "benchmarks.json" in archive.namelist()
        assert "migration.json" in archive.namelist()
        assert "performance_benchmarks.json" in archive.namelist()
        assert "scientific_validity.json" in archive.namelist()
        assert "integrity.json" in archive.namelist()
        assert "citation_metadata.json" in archive.namelist()
        assert "data_dictionary.json" in archive.namelist()
        assert "run_report.pdf" in archive.namelist()
        assert "matter_power.pdf" in archive.namelist()
        assert "matter_power.svg" in archive.namelist()
        assert "matter_power.png" in archive.namelist()
        assert "matter_power_grayscale.pdf" in archive.namelist()
        assert "mass_variance.pdf" in archive.namelist()
        assert "mass_variance.svg" in archive.namelist()
        assert "mass_variance.png" in archive.namelist()
        assert "mass_variance_grayscale.pdf" in archive.namelist()
        assert "analytic_hmf_reference.pdf" in archive.namelist()
        assert "analytic_hmf_reference.svg" in archive.namelist()
        assert "analytic_hmf_reference.png" in archive.namelist()
        assert "analytic_hmf_reference_grayscale.pdf" in archive.namelist()
        assert "hmf.parquet" in archive.namelist()


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


def test_saved_run_integrity_detects_modified_calculation_identity(
    tmp_path, monkeypatch
):
    for key, path in {
        "DATA_ROOT": tmp_path,
        "RUN_DIR": tmp_path / "saved_runs",
        "EXPORT_DIR": tmp_path / "exports",
        "STATE_DIR": tmp_path / "state",
    }.items():
        monkeypatch.setattr(run_storage, key, path)
    params = dict(DEFAULT_PARAMS)
    settings = {"output": "mPk"}
    provenance = run_provenance(params, settings)
    run = {
        "run_id": "integrity-test",
        "name": "Integrity test",
        "params": params,
        "class_settings": settings,
        "provenance": provenance,
        "reproducibility_hash": reproducibility_hash(params, settings, provenance),
        "arrays": {"k": np.array([0.1]), "P": np.array([1.0])},
    }
    run_storage.save_run(run)
    assert (
        run_storage.load_run("integrity-test")["integrity_status"]["state"]
        == "verified"
    )

    path = run_storage.RUN_DIR / "integrity-test.json"
    tampered = json.loads(path.read_text())
    tampered["params"]["H0"] = 80.0
    path.write_text(json.dumps(tampered))
    loaded = run_storage.load_run("integrity-test")
    assert loaded["integrity_status"]["state"] == "invalid"
    assert "will not hydrate" in loaded["storage_warning"]


def test_damaged_run_cannot_be_rechecksummed_by_mutations(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(run_storage, "RUN_DIR", tmp_path / "saved_runs")
    monkeypatch.setattr(run_storage, "EXPORT_DIR", tmp_path / "exports")
    run = {
        "run_id": "damaged",
        "name": "Original",
        "arrays": {"P": np.array([1.0, 2.0])},
    }
    run_storage.save_run(run)
    metadata = json.loads((run_storage.RUN_DIR / "damaged.json").read_text())
    path = run_storage.RUN_DIR / metadata["arrays_file"]
    np.savez(path, P=np.array([9.0, 9.0]))
    originals = {p: p.read_bytes() for p in run_storage.RUN_DIR.iterdir()}
    loaded = run_storage.load_run("damaged")
    assert loaded["integrity_status"]["state"] == "invalid"
    for action in (
        lambda: run_storage.save_run(loaded),
        lambda: run_storage.save_run(run),  # stale caller has no invalid flag
        lambda: run_storage.generate_run_exports(loaded),
        lambda: run_storage.rename_run("damaged", "Renamed"),
        lambda: run_storage.duplicate_run("damaged"),
        lambda: run_storage.update_run_metadata(
            {"run_id": "damaged", "notes": "New note"}
        ),
        lambda: run_storage.set_baseline("damaged"),
    ):
        with pytest.raises(ValueError, match="integrity|damaged"):
            action()
        assert {p: p.read_bytes() for p in run_storage.RUN_DIR.iterdir()} == originals
        assert not (run_storage.EXPORT_DIR / "damaged").exists()
    healthy = {
        "run_id": "healthy",
        "name": "Healthy",
        "arrays": {"P": np.array([1.0, 2.0])},
    }
    run_storage.save_run(healthy)
    run_storage.set_baseline("healthy")
    assert run_storage.load_run("healthy")["is_baseline"]
    assert path.read_bytes() == originals[path]
    assert run_storage.load_run("damaged")["integrity_status"]["state"] == "invalid"
