"""Durable, atomic storage for HaloForge research runs and exports."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.hmf import cumulative_hmf
from engine.fingerprint import cosmic_fingerprint, fingerprint_markdown
from state.run_model import RUN_COLORS
from state.provenance import file_sha256, reproducibility_hash
from state.notebook import normalize_notebook_entry
from state.storage_policy import default_data_root, require_safe_persistent_storage
from state.schema import RUN_STORAGE_SCHEMA_VERSION, migrate_run_document
from state.pdf_report import build_run_pdf
from state.figure_export import build_figure_exports
from state.citations import export_citations
from state.audit import append_audit_event, normalize_audit_trail

DATA_ROOT = default_data_root()
RUN_DIR = DATA_ROOT / "saved_runs"
EXPORT_DIR = DATA_ROOT / "exports"
STATE_DIR = DATA_ROOT / "state"
LAST_RUN_PATH = STATE_DIR / "last_run.json"
DRAFT_PARAMS_PATH = STATE_DIR / "draft_params.json"
_STORAGE_LOCK = threading.RLock()
INTEGRITY_SCHEMA_VERSION = "haloforge-run-integrity-v1"
TABLE_METADATA_SCHEMA_VERSION = "haloforge-table-metadata-v1"
DATA_DICTIONARY_SCHEMA_VERSION = "haloforge-data-dictionary-v1"


def _ensure_dirs() -> None:
    require_safe_persistent_storage()
    for folder in (DATA_ROOT, RUN_DIR, EXPORT_DIR, STATE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _json_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.json"


def _npz_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.npz"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle, **{key: np.asarray(value) for key, value in arrays.items()}
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_parquet(
    path: Path, frame: pd.DataFrame, metadata: dict[str, str] | None = None
) -> None:
    """Write an atomic Parquet table while retaining portable research metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata.update(
            {
                str(key).encode("utf-8"): str(value).encode("utf-8")
                for key, value in (metadata or {}).items()
            }
        )
        pq.write_table(
            table.replace_schema_metadata(schema_metadata), tmp, compression="zstd"
        )
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _metadata(run: dict) -> dict:
    payload = deepcopy(run)
    arrays = payload.pop("arrays", {})
    # This is a read-time assessment, not durable scientific metadata. It must
    # be recomputed against the files each time a run is loaded.
    payload.pop("integrity_status", None)
    payload["name"] = payload.get("run_name", payload.get("name", "Untitled run"))
    payload["run_name"] = payload["name"]
    payload["arrays_file"] = (
        f"{run['run_id']}.npz" if arrays else payload.get("arrays_file")
    )
    return payload


def _normalize_run(run: dict) -> dict:
    name = run.get("run_name", run.get("name", "Untitled run"))
    run["name"] = name
    run["run_name"] = name
    run.setdefault("visible", True)
    run.setdefault("is_baseline", False)
    run.setdefault("exports", {})
    run.setdefault("benchmarks", {})
    run.setdefault("performance_benchmarks", {})
    run.setdefault("scientific_validity", {})
    run.setdefault("integrity", {})
    run["notebook"] = normalize_notebook_entry(run.get("notebook"))
    run.setdefault(
        "updated_at", run.get("created_at", datetime.now(timezone.utc).isoformat())
    )
    run.setdefault("storage_schema_version", RUN_STORAGE_SCHEMA_VERSION)
    run.setdefault("migration_history", [])
    run.setdefault("migration_status", "current")
    run["audit_trail"] = normalize_audit_trail(run.get("audit_trail"))
    return run


def unique_run_name(base_name: str, existing_runs: list[dict] | None = None) -> str:
    base_name = " ".join(str(base_name).strip().split()) or "Untitled run"
    existing = {
        run.get("run_name", run.get("name", ""))
        for run in (existing_runs if existing_runs is not None else load_all_runs())
    }
    if base_name not in existing:
        return base_name
    counter = 2
    while f"{base_name} ({counter})" in existing:
        counter += 1
    return f"{base_name} ({counter})"


def save_run(run: dict) -> dict:
    """Persist a run using atomic replacement so interrupted writes cannot corrupt it."""
    _ensure_dirs()
    with _STORAGE_LOCK:
        run = _normalize_run(migrate_run_document(run))
        if not run["audit_trail"]:
            action = (
                "run_created"
                if not _json_path(run["run_id"]).exists()
                else "audit_initialized"
            )
            append_audit_event(
                run, action, ["run_id", "params"] if action == "run_created" else []
            )
        run["migration_status"] = "current"
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        arrays = run.get("arrays", {})
        if arrays:
            arrays_path = _npz_path(run["run_id"])
            _atomic_write_npz(arrays_path, arrays)
            run["integrity"] = {
                "schema_version": INTEGRITY_SCHEMA_VERSION,
                "arrays_sha256": file_sha256(arrays_path),
                "scope": "Saved NPZ array payload and declared calculation identity.",
            }
        _atomic_write_text(
            _json_path(run["run_id"]), json.dumps(_metadata(run), indent=2, default=str)
        )
    return run


def _load_arrays(run: dict) -> dict:
    arrays_file = run.get("arrays_file")
    if not arrays_file:
        run["arrays"] = {}
        return run
    path = RUN_DIR / arrays_file
    if not path.exists():
        run["arrays"] = {}
        run["storage_warning"] = f"Missing array file: {arrays_file}"
        return run
    try:
        with np.load(path, allow_pickle=False) as data:
            run["arrays"] = {key: data[key] for key in data.files}
    except Exception as exc:
        run["arrays"] = {}
        run["storage_warning"] = f"Could not read {arrays_file}: {exc}"
    return run


def _verify_run_integrity(run: dict) -> dict:
    """Verify durable payload and declared calculation identity without mutating it.

    Older runs without an integrity envelope remain readable but are clearly
    labelled unverified. A mismatch is a fail-closed loading condition.
    """
    checks: list[dict] = []
    integrity = run.get("integrity", {})
    expected_arrays = integrity.get("arrays_sha256")
    arrays_file = run.get("arrays_file")
    if expected_arrays and arrays_file:
        path = RUN_DIR / arrays_file
        actual = file_sha256(path) if path.is_file() else None
        checks.append(
            {
                "check": "array payload SHA-256",
                "state": "pass" if actual == expected_arrays else "fail",
            }
        )
    else:
        checks.append({"check": "array payload SHA-256", "state": "unavailable"})

    provenance = run.get("provenance", {})
    declared = run.get("reproducibility_hash") or provenance.get("reproducibility_hash")
    if declared and provenance and run.get("params") is not None:
        try:
            actual = reproducibility_hash(
                run["params"], run.get("class_settings", {}), provenance
            )
            checks.append(
                {
                    "check": "declared calculation identity",
                    "state": "pass" if actual == declared else "fail",
                }
            )
        except (KeyError, TypeError, ValueError):
            checks.append(
                {"check": "declared calculation identity", "state": "unavailable"}
            )
    else:
        checks.append(
            {"check": "declared calculation identity", "state": "unavailable"}
        )

    states = {item["state"] for item in checks}
    state = (
        "invalid"
        if "fail" in states
        else ("verified" if states == {"pass"} else "unverified_legacy")
    )
    run["integrity_status"] = {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "state": state,
        "checks": checks,
    }
    if state == "invalid":
        run["storage_warning"] = (
            "Integrity verification failed. HaloForge will not hydrate this saved result; inspect or restore the original files."
        )
    return run


def load_all_runs() -> list[dict]:
    _ensure_dirs()
    runs = []
    with _STORAGE_LOCK:
        for path in sorted(RUN_DIR.glob("*.json")):
            try:
                document = migrate_run_document(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                runs.append(
                    _verify_run_integrity(_normalize_run(_load_arrays(document)))
                )
            except Exception:
                continue
    return sorted(runs, key=lambda run: run.get("created_at", ""))


def load_run(run_id: str) -> dict | None:
    _ensure_dirs()
    path = _json_path(run_id)
    if not path.exists():
        return None
    with _STORAGE_LOCK:
        try:
            document = migrate_run_document(
                json.loads(path.read_text(encoding="utf-8"))
            )
            return _verify_run_integrity(_normalize_run(_load_arrays(document)))
        except Exception:
            return None


def delete_run(run_id: str) -> None:
    with _STORAGE_LOCK:
        for path in (_json_path(run_id), _npz_path(run_id)):
            path.unlink(missing_ok=True)
        export_dir = run_export_dir(run_id)
        if export_dir.exists():
            for child in sorted(export_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            export_dir.rmdir()
        if get_last_run_id() == run_id:
            remaining = load_all_runs()
            set_last_run_id(remaining[-1]["run_id"] if remaining else None)


def duplicate_run(run_id: str) -> dict | None:
    run = load_run(run_id)
    if run is None:
        return None
    from uuid import uuid4

    new_run = deepcopy(run)
    new_run["run_id"] = str(uuid4())
    new_run["name"] = unique_run_name(f"Copy of {run['name']}")
    new_run["run_name"] = new_run["name"]
    new_run["is_baseline"] = False
    new_run["created_at"] = datetime.now(timezone.utc).isoformat()
    new_run["updated_at"] = new_run["created_at"]
    new_run["exports"] = {}
    new_run["audit_trail"] = []
    new_run["notebook"] = normalize_notebook_entry(new_run.get("notebook"))
    new_run["notebook"]["parent_run_id"] = run_id
    append_audit_event(new_run, "run_created", ["run_id", "params"])
    append_audit_event(new_run, "forked_from", ["notebook.parent_run_id"])
    save_run(new_run)
    generate_run_exports(new_run)
    set_last_run_id(new_run["run_id"])
    return new_run


def rename_run(run_id: str, requested_name: str) -> dict | None:
    run = load_run(run_id)
    if run is None:
        return None
    others = [
        candidate for candidate in load_all_runs() if candidate["run_id"] != run_id
    ]
    old_name = run["name"]
    run["name"] = unique_run_name(requested_name, others)
    run["run_name"] = run["name"]
    if run["name"] != old_name:
        append_audit_event(run, "renamed", ["name", "run_name"])
    save_run(run)
    generate_run_exports(run)
    return run


def set_baseline(run_id: str) -> None:
    for run in load_all_runs():
        desired = run["run_id"] == run_id
        if run.get("is_baseline") != desired:
            run["is_baseline"] = desired
            append_audit_event(
                run,
                "baseline_selected" if desired else "baseline_cleared",
                ["is_baseline"],
            )
        save_run(run)


def update_run_metadata(run: dict) -> None:
    existing = load_run(run["run_id"])
    if existing and "arrays" not in run:
        run["arrays"] = existing.get("arrays", {})
    if existing:
        excluded = {
            "arrays",
            "exports",
            "updated_at",
            "integrity",
            "integrity_status",
            "audit_trail",
        }
        changed = sorted(
            key
            for key in set(existing) | set(run)
            if key not in excluded and existing.get(key) != run.get(key)
        )
        if changed:
            append_audit_event(run, "metadata_updated", changed)
    save_run(run)


def get_run_label(run: dict) -> str:
    name = run.get("run_name", run.get("name", "Untitled run"))
    status = run.get("class_status", "UNKNOWN")
    created = str(run.get("created_at", ""))[:16].replace("T", " ")
    params = run.get("params", {})
    extras = []
    if "A_s" in params:
        extras.append(f"A_s={float(params['A_s']):.2e}")
    if params.get("enable_ede"):
        extras.append(f"f_EDE={float(params.get('f_EDE', 0.0)):.2f}")
    suffix = " | ".join(
        part for part in [status, created, ", ".join(extras[:2])] if part
    )
    return f"{name} | {suffix}" if suffix else name


def set_last_run_id(run_id: str | None) -> None:
    _ensure_dirs()
    payload = {"run_id": run_id, "updated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_write_text(LAST_RUN_PATH, json.dumps(payload, indent=2))


def get_last_run_id() -> str | None:
    try:
        value = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8")).get("run_id")
        return str(value) if value else None
    except Exception:
        return None


def save_draft_params(params: dict) -> None:
    _ensure_dirs()
    _atomic_write_text(DRAFT_PARAMS_PATH, json.dumps(params, indent=2, default=str))


def load_draft_params() -> dict | None:
    try:
        value = json.loads(DRAFT_PARAMS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _power_result_from_run(run: dict) -> dict:
    arrays = run.get("arrays", {})
    return {
        "k": arrays.get("k", np.array([], dtype=float)),
        "P": arrays.get("P", np.array([], dtype=float)),
        "derived": run.get("derived", {}),
        "class_status": run.get("class_status", ""),
        "class_error": run.get("class_error", ""),
        "warning": run.get("class_error", ""),
    }


def _sigma_result_from_run(run: dict) -> dict:
    arrays = run.get("arrays", {})
    return {
        "M_h": arrays.get("M_h", np.array([], dtype=float)),
        "M": arrays.get("M", np.array([], dtype=float)),
        "R": arrays.get("R", np.array([], dtype=float)),
        "sigma": arrays.get("sigma", np.array([], dtype=float)),
        "dlnsigma_dlnM": arrays.get(
            "dlnsigma_dlnM", arrays.get("dlog_sigma_dlog_M", np.array([], dtype=float))
        ),
        "rho0": run.get("rho0"),
        "window_type": run.get(
            "window_type", run.get("params", {}).get("window_type", "")
        ),
        "delta_c": float(run.get("params", {}).get("delta_c", 1.686)),
        "numerical_diagnostics": run.get("numerical_diagnostics", {}),
    }


def _pipeline_run_from_saved(run: dict) -> dict:
    # Keep the reproducibility envelope when a persisted run is rendered into
    # an export.  Dropping it here made the human-readable report claim that
    # provenance was unavailable even though provenance.json was present.
    return {
        "params": run.get("params", {}),
        "power_result": _power_result_from_run(run),
        "sigma_result": _sigma_result_from_run(run),
        "provenance": run.get("provenance", {}),
        "reproducibility_hash": run.get("reproducibility_hash", ""),
    }


def run_export_dir(run_id: str) -> Path:
    return EXPORT_DIR / run_id


def _safe_export_stem(name: str) -> str:
    cleaned = re.sub(r"[\s/\\]+", "_", name.strip())
    cleaned = "".join(
        ch
        for ch in cleaned
        if ch.isprintable() and ch not in {":", "*", "?", '"', "<", ">", "|"}
    ).strip("_")
    return cleaned or "haloforge_run"


def generate_run_exports(run: dict) -> dict:
    """Write standard per-run export files and update run metadata."""
    run = _normalize_run(run)
    # Establish the array checksum before emitting the bundle's integrity
    # artifact. This gives direct callers the same guarantee as ordinary
    # calculate-and-save flows.
    save_run(run)
    export_dir = run_export_dir(run["run_id"])
    export_dir.mkdir(parents=True, exist_ok=True)
    arrays = run.get("arrays", {})
    files: dict[str, str] = {}

    params_path = export_dir / "params.json"
    _atomic_write_text(
        params_path, json.dumps(run.get("params", {}), indent=2, default=str)
    )
    files["params.json"] = str(params_path)

    derived_path = export_dir / "derived.json"
    _atomic_write_text(
        derived_path, json.dumps(run.get("derived", {}), indent=2, default=str)
    )
    files["derived.json"] = str(derived_path)

    class_path = export_dir / "class_settings.json"
    _atomic_write_text(
        class_path, json.dumps(run.get("class_settings", {}), indent=2, default=str)
    )
    files["class_settings.json"] = str(class_path)

    provenance_path = export_dir / "provenance.json"
    _atomic_write_text(
        provenance_path, json.dumps(run.get("provenance", {}), indent=2, default=str)
    )
    files["provenance.json"] = str(provenance_path)

    notebook_path = export_dir / "notebook.json"
    _atomic_write_text(
        notebook_path,
        json.dumps(
            normalize_notebook_entry(run.get("notebook")), indent=2, default=str
        ),
    )
    files["notebook.json"] = str(notebook_path)

    audit_path = export_dir / "audit_trail.json"
    _atomic_write_text(
        audit_path,
        json.dumps(
            {
                "schema_version": "haloforge-local-run-audit-v1",
                "events": run.get("audit_trail", []),
                "scope": "Local app-level mutation history only; it does not identify people or secure a shared workspace, and cannot prevent filesystem edits.",
            },
            indent=2,
            default=str,
        ),
    )
    files["audit_trail.json"] = str(audit_path)

    benchmarks_path = export_dir / "benchmarks.json"
    _atomic_write_text(
        benchmarks_path, json.dumps(run.get("benchmarks", {}), indent=2, default=str)
    )
    files["benchmarks.json"] = str(benchmarks_path)

    migration_path = export_dir / "migration.json"
    _atomic_write_text(
        migration_path,
        json.dumps(
            {
                "storage_schema_version": run.get("storage_schema_version"),
                "migration_status": run.get("migration_status"),
                "migration_history": run.get("migration_history", []),
            },
            indent=2,
            default=str,
        ),
    )
    files["migration.json"] = str(migration_path)

    performance_path = export_dir / "performance_benchmarks.json"
    _atomic_write_text(
        performance_path,
        json.dumps(run.get("performance_benchmarks", {}), indent=2, default=str),
    )
    files["performance_benchmarks.json"] = str(performance_path)

    validity_path = export_dir / "scientific_validity.json"
    _atomic_write_text(
        validity_path,
        json.dumps(run.get("scientific_validity", {}), indent=2, default=str),
    )
    files["scientific_validity.json"] = str(validity_path)

    integrity_path = export_dir / "integrity.json"
    _atomic_write_text(
        integrity_path, json.dumps(run.get("integrity", {}), indent=2, default=str)
    )
    files["integrity.json"] = str(integrity_path)

    citations_path = export_dir / "citation_metadata.json"
    _atomic_write_text(
        citations_path, json.dumps(export_citations(run), indent=2, default=str)
    )
    files["citation_metadata.json"] = str(citations_path)

    pdf_path = export_dir / "run_report.pdf"
    _atomic_write_bytes(pdf_path, build_run_pdf(run))
    files["run_report.pdf"] = str(pdf_path)

    for filename, content in build_figure_exports(run).items():
        path = export_dir / filename
        _atomic_write_bytes(path, content)
        files[filename] = str(path)

    readme_path = export_dir / "README.md"
    _atomic_write_text(readme_path, export_readme(run))
    files["README.md"] = str(readme_path)

    share_card_path = export_dir / "share_card.md"
    _atomic_write_text(share_card_path, share_card_markdown(run))
    files["share_card.md"] = str(share_card_path)

    recreation_path = export_dir / "recreate.py"
    _atomic_write_text(recreation_path, python_recreation_script(run))
    files["recreate.py"] = str(recreation_path)

    for filename, content in analysis_loader_snippets().items():
        path = export_dir / filename
        _atomic_write_text(path, content)
        files[filename] = str(path)

    if "k" in arrays and "P" in arrays:
        power_path = export_dir / "power_spectrum.csv"
        _atomic_write_text(power_path, export_power_csv(_power_result_from_run(run)))
        files["power_spectrum.csv"] = str(power_path)
        power_parquet = export_dir / "power_spectrum.parquet"
        _atomic_write_parquet(
            power_parquet,
            pd.read_csv(StringIO(export_power_csv(_power_result_from_run(run)))),
            _power_table_metadata(run),
        )
        files["power_spectrum.parquet"] = str(power_parquet)

    if {"M_h", "M", "R", "sigma"}.issubset(arrays):
        sigma_path = export_dir / "sigma.csv"
        _atomic_write_text(sigma_path, export_sigma_csv(_sigma_result_from_run(run)))
        files["sigma.csv"] = str(sigma_path)
        sigma_parquet = export_dir / "sigma.parquet"
        _atomic_write_parquet(
            sigma_parquet,
            pd.read_csv(StringIO(export_sigma_csv(_sigma_result_from_run(run)))),
            _sigma_table_metadata(run),
        )
        files["sigma.parquet"] = str(sigma_parquet)

    if run.get(
        "window_type", run.get("params", {}).get("window_type")
    ) == "Top-hat" and {
        "M_h",
        "hmf_press_schechter_z0",
        "hmf_sheth_tormen_z0",
    }.issubset(arrays):
        hmf_path = export_dir / "hmf.csv"
        hmf_csv = export_hmf_csv(arrays)
        _atomic_write_text(hmf_path, hmf_csv)
        files["hmf.csv"] = str(hmf_path)
        hmf_parquet = export_dir / "hmf.parquet"
        _atomic_write_parquet(
            hmf_parquet, pd.read_csv(StringIO(hmf_csv)), _hmf_table_metadata(run)
        )
        files["hmf.parquet"] = str(hmf_parquet)

    dictionary_path = export_dir / "data_dictionary.json"
    _atomic_write_text(
        dictionary_path,
        json.dumps(export_data_dictionary(run), indent=2, sort_keys=True),
    )
    files["data_dictionary.json"] = str(dictionary_path)

    summary_path = export_dir / "run_summary.md"
    report_run = _pipeline_run_from_saved(run)
    _atomic_write_text(
        summary_path,
        summary_markdown_report(
            report_run,
            run.get("name", "HaloForge run"),
            run.get("notes", ""),
            [
                "Auto-exported AxiCLASS P(k,z), sigma(M,z), and HMF products when available."
            ],
        ),
    )
    files["run_summary.md"] = str(summary_path)

    manifest_path = export_dir / "manifest.json"
    manifest = {
        "schema_version": run.get("provenance", {}).get(
            "schema_version", "unavailable"
        ),
        "run_id": run["run_id"],
        "reproducibility_hash": run.get(
            "reproducibility_hash",
            run.get("provenance", {}).get("reproducibility_hash", "unavailable"),
        ),
        "files": {
            name: {
                "sha256": file_sha256(Path(path)),
                "bytes": Path(path).stat().st_size,
            }
            for name, path in files.items()
        },
        "integrity_scope": "Checksums cover files inside this bundle before ZIP compression.",
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    files["manifest.json"] = str(manifest_path)

    zip_path = (
        export_dir
        / f"{_safe_export_stem(run.get('name', 'haloforge_run'))}_exports.zip"
    )
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with ZipFile(tmp_zip, "w", ZIP_DEFLATED) as archive:
        for path_str in files.values():
            path = Path(path_str)
            if path.exists():
                archive.write(path, arcname=path.name)
    os.replace(tmp_zip, zip_path)
    files["exports.zip"] = str(zip_path)

    run["exports"] = files
    save_run(run)
    return files


def export_status(run: dict) -> list[dict]:
    export_dir = run_export_dir(run["run_id"])
    names = [
        "README.md",
        "share_card.md",
        "params.json",
        "derived.json",
        "class_settings.json",
        "provenance.json",
        "notebook.json",
        "audit_trail.json",
        "benchmarks.json",
        "migration.json",
        "performance_benchmarks.json",
        "scientific_validity.json",
        "integrity.json",
        "citation_metadata.json",
        "data_dictionary.json",
        "run_report.pdf",
        "matter_power.pdf",
        "matter_power.svg",
        "matter_power.png",
        "matter_power_grayscale.pdf",
        "matter_power_grayscale.svg",
        "matter_power_grayscale.png",
        "mass_variance.pdf",
        "mass_variance.svg",
        "mass_variance.png",
        "mass_variance_grayscale.pdf",
        "mass_variance_grayscale.svg",
        "mass_variance_grayscale.png",
        "analytic_hmf_reference.pdf",
        "analytic_hmf_reference.svg",
        "analytic_hmf_reference.png",
        "analytic_hmf_reference_grayscale.pdf",
        "analytic_hmf_reference_grayscale.svg",
        "analytic_hmf_reference_grayscale.png",
        "recreate.py",
        "load_export.jl",
        "load_export.R",
        "load_export.wl",
        "manifest.json",
        "power_spectrum.csv",
        "power_spectrum.parquet",
        "sigma.csv",
        "sigma.parquet",
        "hmf.csv",
        "hmf.parquet",
        "run_summary.md",
    ]
    return [
        {
            "file": name,
            "exists": (export_dir / name).exists(),
            "path": str(export_dir / name),
        }
        for name in names
    ]


def export_run_json(run_id: str) -> str:
    run = load_run(run_id)
    return (
        json.dumps(_metadata(run), indent=2, default=str) if run is not None else "{}"
    )


def export_run_csv(run_id: str) -> str:
    run = load_run(run_id)
    if run is None or not run.get("arrays"):
        return ""
    arrays = run["arrays"]
    rows = []
    for i in range(len(arrays["M_h"])):
        rows.append(
            {
                "M_h_hinv_Msun": arrays["M_h"][i],
                "M_Msun": arrays["M"][i],
                "R_Mpc": arrays["R"][i],
                "sigma": arrays["sigma"][i],
                "dlnsigma_dlnM": arrays["dlnsigma_dlnM"][i],
            }
        )
    return _rows_to_csv(rows)


def next_color_index() -> int:
    return len(load_all_runs()) % len(RUN_COLORS)


def _rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def export_power_csv(power_result: dict) -> str:
    rows = [
        {
            "k_Mpc^-1": float(k),
            "P_Mpc^3": float(p),
            "Delta2": float(k**3 * p / (2.0 * np.pi**2)),
        }
        for k, p in zip(power_result["k"], power_result["P"])
    ]
    return _rows_to_csv(rows)


def export_sigma_csv(sigma_result: dict) -> str:
    rows = []
    deriv = sigma_result.get("dlnsigma_dlnM", np.zeros_like(sigma_result["sigma"]))
    for M_h, M, R, sigma, slope in zip(
        sigma_result["M_h"],
        sigma_result["M"],
        sigma_result["R"],
        sigma_result["sigma"],
        deriv,
    ):
        rows.append(
            {
                "M_h_hinv_Msun": float(M_h),
                "M_Msun": float(M),
                "R_Mpc": float(R),
                "sigma": float(sigma),
                "dlnsigma_dlnM": float(slope),
            }
        )
    return _rows_to_csv(rows)


def export_hmf_csv(arrays: dict) -> str:
    """Export stored analytic HMF references without upgrading their scope."""
    masses = np.asarray(arrays["M_h"])
    press_schechter = np.asarray(arrays["hmf_press_schechter_z0"])
    sheth_tormen = np.asarray(arrays["hmf_sheth_tormen_z0"])
    cumulative_ps = np.asarray(
        arrays.get(
            "cumulative_press_schechter_z0", cumulative_hmf(masses, press_schechter)
        )
    )
    cumulative_st = np.asarray(
        arrays.get("cumulative_sheth_tormen_z0", cumulative_hmf(masses, sheth_tormen))
    )
    if not (
        masses.shape
        == press_schechter.shape
        == sheth_tormen.shape
        == cumulative_ps.shape
        == cumulative_st.shape
    ):
        raise ValueError("Stored HMF arrays must share one mass-grid shape.")
    rows = [
        {
            "M_h_inv_Msun": float(mass),
            "hmf_press_schechter_z0": float(ps),
            "hmf_sheth_tormen_z0": float(st),
            "cumulative_press_schechter_z0": float(cps),
            "cumulative_sheth_tormen_z0": float(cst),
        }
        for mass, ps, st, cps, cst in zip(
            masses, press_schechter, sheth_tormen, cumulative_ps, cumulative_st
        )
    ]
    return _rows_to_csv(rows)


def _research_table_metadata(
    run: dict, table_name: str, units: dict[str, str], descriptions: dict[str, str]
) -> dict[str, str]:
    """Return self-describing metadata shared by exported scientific tables.

    The companion JSON files remain the authoritative full records.  These
    links make a detached Parquet file interpretable without pretending that
    its numerical values alone establish calibration or publication fitness.
    """
    provenance = run.get("provenance", {})
    reproducibility = run.get(
        "reproducibility_hash", provenance.get("reproducibility_hash", "unavailable")
    )
    return {
        "haloforge.schema_version": TABLE_METADATA_SCHEMA_VERSION,
        "haloforge.table": table_name,
        "haloforge.units": json.dumps(units, sort_keys=True),
        "haloforge.descriptions": json.dumps(descriptions, sort_keys=True),
        "haloforge.run_id": str(run.get("run_id", "unavailable")),
        "haloforge.reproducibility_hash": str(reproducibility),
        "haloforge.provenance_file": "provenance.json",
        "haloforge.validity_file": "scientific_validity.json",
        "haloforge.integrity_file": "integrity.json",
        "haloforge.citations_file": "citation_metadata.json",
        "haloforge.scope_limit": "Table metadata links to the bundle evidence; it does not itself establish physical appropriateness, fit calibration, or publication suitability.",
    }


def _power_table_metadata(run: dict) -> dict[str, str]:
    return _research_table_metadata(
        run,
        "linear_matter_power",
        {"k_Mpc^-1": "Mpc^-1", "P_Mpc^3": "Mpc^3", "Delta2": "dimensionless"},
        {
            "k_Mpc^-1": "Comoving Fourier wavenumber.",
            "P_Mpc^3": "Linear matter power spectrum at the focused redshift.",
            "Delta2": "Dimensionless linear power, k^3 P(k) / (2 pi^2).",
        },
    )


def _sigma_table_metadata(run: dict) -> dict[str, str]:
    return _research_table_metadata(
        run,
        "mass_variance",
        {
            "M_h_hinv_Msun": "h^-1 Msun",
            "M_Msun": "Msun",
            "R_Mpc": "Mpc",
            "sigma": "dimensionless",
            "dlnsigma_dlnM": "dimensionless",
        },
        {
            "M_h_hinv_Msun": "Halo mass in h^-1 solar masses.",
            "M_Msun": "Physical halo mass in solar masses using the saved h.",
            "R_Mpc": "Top-hat-equivalent comoving smoothing radius.",
            "sigma": "Linear density-field RMS after the saved smoothing prescription.",
            "dlnsigma_dlnM": "Logarithmic slope of sigma with respect to mass.",
        },
    )


def _hmf_table_metadata(run: dict) -> dict[str, str]:
    metadata = _research_table_metadata(
        run,
        "analytic_halo_mass_function_references_z0",
        {
            "M_h_inv_Msun": "h^-1 Msun",
            "hmf_press_schechter_z0": "h^3 Mpc^-3",
            "hmf_sheth_tormen_z0": "h^3 Mpc^-3",
            "cumulative_press_schechter_z0": "h^3 Mpc^-3",
            "cumulative_sheth_tormen_z0": "h^3 Mpc^-3",
        },
        {
            "M_h_inv_Msun": "Analytic top-hat smoothing mass in h^-1 solar masses.",
            "hmf_press_schechter_z0": "Press-Schechter 1974 analytic differential reference dn/dlnM at z=0.",
            "hmf_sheth_tormen_z0": "Sheth-Tormen 2001 analytic differential reference dn/dlnM at z=0.",
            "cumulative_press_schechter_z0": "Finite sampled-grid cumulative Press-Schechter reference n(>M) at z=0.",
            "cumulative_sheth_tormen_z0": "Finite sampled-grid cumulative Sheth-Tormen reference n(>M) at z=0.",
        },
    )
    metadata.update(
        {
            "haloforge.mass_definition": "analytic_top_hat",
            "haloforge.reference_models": "Press-Schechter 1974; Sheth-Tormen 2001",
            "haloforge.hmf_scope_limit": "Analytic top-hat references only; these columns are not a universal empirical HMF calibration. Cumulative values integrate only over the stored finite mass grid.",
        }
    )
    return metadata


def _dictionary_table(metadata: dict[str, str], files: list[str]) -> dict:
    units = json.loads(metadata["haloforge.units"])
    descriptions = json.loads(metadata["haloforge.descriptions"])
    return {
        "table": metadata["haloforge.table"],
        "files": files,
        "columns": [
            {
                "name": name,
                "unit": unit,
                "description": descriptions.get(name, "Description not recorded."),
            }
            for name, unit in units.items()
        ],
        "provenance_links": {
            "provenance": metadata["haloforge.provenance_file"],
            "validity": metadata["haloforge.validity_file"],
            "integrity": metadata["haloforge.integrity_file"],
            "citations": metadata["haloforge.citations_file"],
        },
        "scope_limit": metadata["haloforge.scope_limit"],
        "table_context": {
            key.removeprefix("haloforge."): value
            for key, value in metadata.items()
            if key
            in {
                "haloforge.mass_definition",
                "haloforge.reference_models",
                "haloforge.hmf_scope_limit",
            }
        },
    }


def export_data_dictionary(run: dict) -> dict:
    """Create the human- and machine-readable field guide for one bundle."""
    arrays = run.get("arrays", {})
    tables = []
    if "k" in arrays and "P" in arrays:
        tables.append(
            _dictionary_table(
                _power_table_metadata(run),
                ["power_spectrum.csv", "power_spectrum.parquet"],
            )
        )
    if {"M_h", "M", "R", "sigma"}.issubset(arrays):
        tables.append(
            _dictionary_table(
                _sigma_table_metadata(run), ["sigma.csv", "sigma.parquet"]
            )
        )
    if run.get(
        "window_type", run.get("params", {}).get("window_type")
    ) == "Top-hat" and {
        "M_h",
        "hmf_press_schechter_z0",
        "hmf_sheth_tormen_z0",
    }.issubset(arrays):
        tables.append(
            _dictionary_table(_hmf_table_metadata(run), ["hmf.csv", "hmf.parquet"])
        )
    return {
        "schema_version": DATA_DICTIONARY_SCHEMA_VERSION,
        "run_id": str(run.get("run_id", "unavailable")),
        "reproducibility_hash": str(
            run.get(
                "reproducibility_hash",
                run.get("provenance", {}).get("reproducibility_hash", "unavailable"),
            )
        ),
        "tables": tables,
        "scope_limit": "This dictionary documents stored values and their bundle context. It does not independently establish physical appropriateness, empirical-fit calibration, or publication suitability.",
    }


def summary_markdown_report(
    run: dict, name: str, notes: str, bullets: list[str]
) -> str:
    params = run.get("params", {})
    power = run.get("power_result", {})
    sigma = run.get("sigma_result", {})
    diagnostics = sigma.get("numerical_diagnostics", {})
    provenance = run.get("provenance", {})
    lines = [
        f"# {name}",
        "",
        notes.strip(),
        "",
        "## Reproducibility",
        "",
        f"- Backend: {power.get('class_status', 'UNKNOWN')}",
        f"- k range: {params.get('k_min')} to {params.get('k_max')} Mpc^-1",
        f"- k samples: {params.get('k_points')}",
        f"- mass range: 10^{params.get('mass_min_exp')} to 10^{params.get('mass_max_exp')} h^-1 Msun",
        f"- mass samples: {params.get('mass_points')}",
        f"- window: {sigma.get('window_type', params.get('window_type'))}",
        "- HMF availability: "
        + (
            "Top-hat model; calibration and cosmology still require validation."
            if sigma.get("window_type", params.get("window_type")) == "Top-hat"
            else "Unavailable: alternate-window variance has no calibrated halo mass assignment in HaloForge."
        ),
        "- integration: fixed log-k Simpson rule on the sampled CLASS grid",
        f"- numerical sensitivity check: {diagnostics.get('method', 'not recorded')}",
        f"- numerical scope limit: {diagnostics.get('scope_limit', 'not recorded')}",
        f"- run schema: {provenance.get('schema_version', 'not recorded')}",
        f"- reproducibility hash: {provenance.get('reproducibility_hash', 'not recorded')}",
        f"- application revision: {provenance.get('git_revision', 'not recorded')}",
        f"- AxiCLASS commit: {provenance.get('axiclass_commit', 'not recorded')}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines).strip() + "\n"


def export_readme(run: dict) -> str:
    """Human-readable documentation that is deliberately bundled with every run."""
    notebook = normalize_notebook_entry(run.get("notebook"))
    provenance = run.get("provenance", {})
    lines = [
        f"# HaloForge export — {run.get('name', 'Untitled run')}",
        "",
        "This is a local, reproducible research bundle. It is not a claim that every empirical halo-mass-function result is calibrated for this cosmology.",
        "",
        "## What is included",
        "",
        "- `params.json`: submitted cosmological, halo-model, and numerical parameters.",
        "- `class_settings.json`: exact CLASS/AxiCLASS settings sent to the solver.",
        "- `power_spectrum.csv` / `.parquet`: `k_Mpc^-1`, linear `P_Mpc^3`, and dimensionless `Delta2` at the focused redshift. The Parquet schema embeds field units, descriptions, the reproducibility hash, and links to this bundle's evidence records.",
        "- `sigma.csv` / `.parquet`: `M_h_hinv_Msun`, physical `M_Msun`, top-hat-equivalent `R_Mpc`, `sigma`, and `dlnsigma_dlnM`. The Parquet schema embeds field units, descriptions, the reproducibility hash, and links to this bundle's evidence records.",
        "- `hmf.csv` / `.parquet`: analytic Press-Schechter and Sheth-Tormen top-hat reference HMF products at z=0 where available. The Parquet schema embeds units, finite-grid cumulative semantics, provenance links, and an explicit warning not to treat these columns as a universal empirical calibration.",
        "- `data_dictionary.json`: one versioned field-by-field guide to every included scientific table, including files, units, definitions, provenance links, and table-specific scope limits.",
        "- `notebook.json`: question, prior hypothesis, conclusion, caveats, citations, chart-region annotations, branch parent, and explicit linked follow-up experiment IDs.",
        "- `run_report.pdf`: a human-readable provenance, parameter, validity, notebook, and caveat summary; it does not replace the machine-readable data.",
        "- `citation_metadata.json`: figure-specific scientific-source and method metadata plus any user-supplied notebook citations.",
        "- `matter_power`, `mass_variance`, and `analytic_hmf_reference` figures when available: matching PDF and SVG vector versions plus 1920×1300 PNG files, each with captions, citation pointers, and explicit validity limits. `_grayscale` counterparts use contrast and line patterns for print-safe interpretation.",
        "- `provenance.json`, `migration.json`, `scientific_validity.json`, `integrity.json`, and `manifest.json`: calculation provenance, recoverable schema history, separate validity claims, and integrity hashes.",
        "- `recreate.py`: a one-command AxiCLASS smoke recreation of the submitted solver settings.",
        "- `load_export.jl`, `load_export.R`, and `load_export.wl`: table-loading starters for Julia, R, and Mathematica.",
        "",
        "## Recreate",
        "",
        "Use the same AxiCLASS-compatible `classy` build recorded in `provenance.json`, then run:",
        "",
        "```bash",
        "python recreate.py",
        "```",
        "",
        "The script writes `recreated_power_spectrum.csv` and compares it to the stored table. Exact byte-for-byte agreement is not promised across solver builds or platforms; discrepancies must be investigated, not ignored.",
        "",
        "## Scientific interpretation limits",
        "",
        f"- Reproducibility hash: `{run.get('reproducibility_hash', provenance.get('reproducibility_hash', 'unavailable'))}`",
        f"- HMF status: `{run.get('hmf_status', 'unavailable')}`",
        "- Numerical endpoint checks describe sampled-range sensitivity only; they are not a physical or solver-convergence proof.",
        "- The structure field in HaloForge is a linear Gaussian illustration, not an N-body simulation or halo catalogue.",
        "",
        "## Notebook",
        "",
        f"- Research question: {notebook['research_question'] or 'not recorded'}",
        f"- Hypothesis: {notebook['hypothesis'] or 'not recorded'}",
        f"- Conclusion: {notebook['conclusion'] or 'not recorded'}",
    ]
    if notebook["caveats"]:
        lines.extend(
            ["", "### Caveats", ""] + [f"- {item}" for item in notebook["caveats"]]
        )
    if notebook["citations"]:
        lines.extend(
            ["", "### User-supplied citations", ""]
            + [f"- {item}" for item in notebook["citations"]]
        )
    return "\n".join(lines).strip() + "\n"


def share_card_markdown(run: dict, baseline: dict | None = None) -> str:
    """A compact share artifact that carries the model caveat with the claim."""
    notebook = normalize_notebook_entry(run.get("notebook"))
    params = run.get("params", {})
    deltas = []
    if baseline:
        for key in ("A_s", "n_s", "Omega_m", "f_EDE", "log10_a_c"):
            if baseline.get("params", {}).get(key) != params.get(key):
                deltas.append(
                    f"{key}: {baseline['params'].get(key)} → {params.get(key)}"
                )
    if not deltas:
        deltas.append(
            "No named baseline comparison was embedded in this standalone card."
        )
    caveat = (
        notebook["caveats"][0]
        if notebook["caveats"]
        else "This is a linear-theory calculation; empirical halo-mass-function claims depend on fit validity and calibration domain."
    )
    fingerprint = cosmic_fingerprint(run, baseline)
    return (
        "\n".join(
            [
                "# What universe did you make?",
                "",
                f"## {run.get('name', 'Untitled HaloForge experiment')}",
                "",
                f"**Question:** {notebook['research_question'] or 'Explore how a changed cosmology propagates into structure.'}",
                f"**Prediction before running:** {notebook['prediction'] or notebook['hypothesis'] or 'Not recorded.'}",
                "",
                "### Parameter fingerprint",
                "",
                *[f"- {delta}" for delta in deltas],
                "",
                fingerprint_markdown(fingerprint).strip(),
                "",
                f"**Honest takeaway:** {notebook['conclusion'] or 'Inspect the matched baseline comparison before drawing a quantitative conclusion.'}",
                "",
                f"**Caveat:** {caveat}",
                "",
                "Reproducibility hash: `"
                + str(run.get("reproducibility_hash", "unavailable"))
                + "`",
                "",
                "HaloForge outputs are calculations with stated assumptions—not literal simulated universes or observational discoveries.",
            ]
        )
        + "\n"
    )


def python_recreation_script(run: dict) -> str:
    settings = json.dumps(
        run.get("class_settings", {}), indent=2, sort_keys=True, default=str
    )
    return (
        '''#!/usr/bin/env python3
"""Recreate the stored AxiCLASS linear P(k) sample for this HaloForge run."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

SETTINGS = json.loads(r'''
        + repr(settings)
        + """)
ROOT = Path(__file__).resolve().parent

try:
    from classy import Class
except ImportError as exc:
    raise SystemExit("This export requires the AxiCLASS-compatible classy build recorded in provenance.json.") from exc

stored = np.genfromtxt(ROOT / "power_spectrum.csv", delimiter=",", names=True)
cosmo = Class()
cosmo.set(SETTINGS)
cosmo.compute()
try:
    rows = []
    for k in np.atleast_1d(stored["k_Mpc^-1"]):
        value = float(cosmo.pk_lin(float(k), 0.0))
        rows.append((float(k), value, float(k**3 * value / (2.0 * np.pi**2))))
    with (ROOT / "recreated_power_spectrum.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k_Mpc^-1", "P_Mpc^3", "Delta2"])
        writer.writerows(rows)
    reference = np.atleast_1d(stored["P_Mpc3"])
    recreated = np.asarray([row[1] for row in rows])
    fractional = np.abs(recreated / reference - 1.0)
    print(f"max fractional P(k) discrepancy at z=0: {{fractional.max():.3e}}")
    print("Interpret discrepancies using provenance.json; never discard a failed agreement.")
finally:
    cosmo.struct_cleanup()
    cosmo.empty()
"""
    )


def analysis_loader_snippets() -> dict[str, str]:
    return {
        "load_export.jl": """# Julia: inspect the exported table (CSV.jl, DataFrames.jl)\nusing CSV, DataFrames\npower = CSV.read("power_spectrum.csv", DataFrame)\nprintln(first(power, 5))\n""",
        "load_export.R": """# R: inspect the exported table\npower <- read.csv("power_spectrum.csv")\nprint(head(power))\n""",
        "load_export.wl": """(* Mathematica / Wolfram Language: inspect the exported table *)\npower = Import["power_spectrum.csv", "Dataset"];\npower[Take[All, UpTo[5]]]\n""",
    }
