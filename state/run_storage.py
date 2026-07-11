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

from engine.hmf import cumulative_hmf
from state.run_model import RUN_COLORS

DATA_ROOT = Path(os.environ.get("HALOFORGE_DATA_DIR", "data")).expanduser().resolve()
RUN_DIR = DATA_ROOT / "saved_runs"
EXPORT_DIR = DATA_ROOT / "exports"
STATE_DIR = DATA_ROOT / "state"
LAST_RUN_PATH = STATE_DIR / "last_run.json"
DRAFT_PARAMS_PATH = STATE_DIR / "draft_params.json"
_STORAGE_LOCK = threading.RLock()


def _ensure_dirs() -> None:
    for folder in (DATA_ROOT, RUN_DIR, EXPORT_DIR, STATE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _json_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.json"


def _npz_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.npz"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _metadata(run: dict) -> dict:
    payload = deepcopy(run)
    arrays = payload.pop("arrays", {})
    payload["name"] = payload.get("run_name", payload.get("name", "Untitled run"))
    payload["run_name"] = payload["name"]
    payload["arrays_file"] = f"{run['run_id']}.npz" if arrays else payload.get("arrays_file")
    return payload


def _normalize_run(run: dict) -> dict:
    name = run.get("run_name", run.get("name", "Untitled run"))
    run["name"] = name
    run["run_name"] = name
    run.setdefault("visible", True)
    run.setdefault("is_baseline", False)
    run.setdefault("exports", {})
    run.setdefault("updated_at", run.get("created_at", datetime.now(timezone.utc).isoformat()))
    return run


def unique_run_name(base_name: str, existing_runs: list[dict] | None = None) -> str:
    base_name = " ".join(str(base_name).strip().split()) or "Untitled run"
    existing = {run.get("run_name", run.get("name", "")) for run in (existing_runs if existing_runs is not None else load_all_runs())}
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
        run = _normalize_run(run)
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        arrays = run.get("arrays", {})
        if arrays:
            _atomic_write_npz(_npz_path(run["run_id"]), arrays)
        _atomic_write_text(_json_path(run["run_id"]), json.dumps(_metadata(run), indent=2, default=str))
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


def load_all_runs() -> list[dict]:
    _ensure_dirs()
    runs = []
    with _STORAGE_LOCK:
        for path in sorted(RUN_DIR.glob("*.json")):
            try:
                runs.append(_normalize_run(_load_arrays(json.loads(path.read_text(encoding="utf-8")))))
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
            return _normalize_run(_load_arrays(json.loads(path.read_text(encoding="utf-8"))))
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
    save_run(new_run)
    generate_run_exports(new_run)
    set_last_run_id(new_run["run_id"])
    return new_run


def rename_run(run_id: str, requested_name: str) -> dict | None:
    run = load_run(run_id)
    if run is None:
        return None
    others = [candidate for candidate in load_all_runs() if candidate["run_id"] != run_id]
    run["name"] = unique_run_name(requested_name, others)
    run["run_name"] = run["name"]
    save_run(run)
    generate_run_exports(run)
    return run


def set_baseline(run_id: str) -> None:
    for run in load_all_runs():
        run["is_baseline"] = run["run_id"] == run_id
        save_run(run)


def update_run_metadata(run: dict) -> None:
    existing = load_run(run["run_id"])
    if existing and "arrays" not in run:
        run["arrays"] = existing.get("arrays", {})
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
    suffix = " | ".join(part for part in [status, created, ", ".join(extras[:2])] if part)
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
        "dlnsigma_dlnM": arrays.get("dlnsigma_dlnM", arrays.get("dlog_sigma_dlog_M", np.array([], dtype=float))),
        "rho0": run.get("rho0"),
        "window_type": run.get("window_type", run.get("params", {}).get("window_type", "")),
        "delta_c": float(run.get("params", {}).get("delta_c", 1.686)),
    }


def _pipeline_run_from_saved(run: dict) -> dict:
    return {"params": run.get("params", {}), "power_result": _power_result_from_run(run), "sigma_result": _sigma_result_from_run(run)}


def run_export_dir(run_id: str) -> Path:
    return EXPORT_DIR / run_id


def _safe_export_stem(name: str) -> str:
    cleaned = re.sub(r"[\s/\\]+", "_", name.strip())
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() and ch not in {":", "*", "?", '"', "<", ">", "|"}).strip("_")
    return cleaned or "haloforge_run"


def generate_run_exports(run: dict) -> dict:
    """Write standard per-run export files and update run metadata."""
    run = _normalize_run(run)
    export_dir = run_export_dir(run["run_id"])
    export_dir.mkdir(parents=True, exist_ok=True)
    arrays = run.get("arrays", {})
    files: dict[str, str] = {}

    params_path = export_dir / "params.json"
    _atomic_write_text(params_path, json.dumps(run.get("params", {}), indent=2, default=str))
    files["params.json"] = str(params_path)

    derived_path = export_dir / "derived.json"
    _atomic_write_text(derived_path, json.dumps(run.get("derived", {}), indent=2, default=str))
    files["derived.json"] = str(derived_path)

    class_path = export_dir / "class_settings.json"
    _atomic_write_text(class_path, json.dumps(run.get("class_settings", {}), indent=2, default=str))
    files["class_settings.json"] = str(class_path)

    if "k" in arrays and "P" in arrays:
        power_path = export_dir / "power_spectrum.csv"
        _atomic_write_text(power_path, export_power_csv(_power_result_from_run(run)))
        files["power_spectrum.csv"] = str(power_path)

    if {"M_h", "M", "R", "sigma"}.issubset(arrays):
        sigma_path = export_dir / "sigma.csv"
        _atomic_write_text(sigma_path, export_sigma_csv(_sigma_result_from_run(run)))
        files["sigma.csv"] = str(sigma_path)

    if {"M_h", "hmf_press_schechter_z0", "hmf_sheth_tormen_z0"}.issubset(arrays):
        hmf_path = export_dir / "hmf.csv"
        rows = []
        cumulative_ps = arrays.get("cumulative_press_schechter_z0", cumulative_hmf(arrays["M_h"], arrays["hmf_press_schechter_z0"]))
        cumulative_st = arrays.get("cumulative_sheth_tormen_z0", cumulative_hmf(arrays["M_h"], arrays["hmf_sheth_tormen_z0"]))
        for i, mass in enumerate(arrays["M_h"]):
            rows.append({
                "M_h_inv_Msun": float(mass),
                "hmf_press_schechter_z0": float(arrays["hmf_press_schechter_z0"][i]),
                "hmf_sheth_tormen_z0": float(arrays["hmf_sheth_tormen_z0"][i]),
                "cumulative_press_schechter_z0": float(cumulative_ps[i]),
                "cumulative_sheth_tormen_z0": float(cumulative_st[i]),
            })
        _atomic_write_text(hmf_path, _rows_to_csv(rows))
        files["hmf.csv"] = str(hmf_path)

    summary_path = export_dir / "run_summary.md"
    report_run = _pipeline_run_from_saved(run)
    _atomic_write_text(summary_path, summary_markdown_report(
        report_run,
        run.get("name", "HaloForge run"),
        run.get("notes", ""),
        ["Auto-exported AxiCLASS P(k,z), sigma(M,z), and HMF products when available."],
    ))
    files["run_summary.md"] = str(summary_path)

    zip_path = export_dir / f"{_safe_export_stem(run.get('name', 'haloforge_run'))}_exports.zip"
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
    names = ["params.json", "derived.json", "class_settings.json", "power_spectrum.csv", "sigma.csv", "hmf.csv", "run_summary.md"]
    return [{"file": name, "exists": (export_dir / name).exists(), "path": str(export_dir / name)} for name in names]


def export_run_json(run_id: str) -> str:
    run = load_run(run_id)
    return json.dumps(_metadata(run), indent=2, default=str) if run is not None else "{}"


def export_run_csv(run_id: str) -> str:
    run = load_run(run_id)
    if run is None or not run.get("arrays"):
        return ""
    arrays = run["arrays"]
    rows = []
    for i in range(len(arrays["M_h"])):
        rows.append({
            "M_h_hinv_Msun": arrays["M_h"][i],
            "M_Msun": arrays["M"][i],
            "R_Mpc": arrays["R"][i],
            "sigma": arrays["sigma"][i],
            "dlnsigma_dlnM": arrays["dlnsigma_dlnM"][i],
        })
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
    rows = [{"k_Mpc^-1": float(k), "P_Mpc^3": float(p), "Delta2": float(k**3 * p / (2.0 * np.pi**2))} for k, p in zip(power_result["k"], power_result["P"])]
    return _rows_to_csv(rows)


def export_sigma_csv(sigma_result: dict) -> str:
    rows = []
    deriv = sigma_result.get("dlnsigma_dlnM", np.zeros_like(sigma_result["sigma"]))
    for M_h, M, R, sigma, slope in zip(sigma_result["M_h"], sigma_result["M"], sigma_result["R"], sigma_result["sigma"], deriv):
        rows.append({"M_h_hinv_Msun": float(M_h), "M_Msun": float(M), "R_Mpc": float(R), "sigma": float(sigma), "dlnsigma_dlnM": float(slope)})
    return _rows_to_csv(rows)


def summary_markdown_report(run: dict, name: str, notes: str, bullets: list[str]) -> str:
    params = run.get("params", {})
    power = run.get("power_result", {})
    sigma = run.get("sigma_result", {})
    lines = [
        f"# {name}", "", notes.strip(), "", "## Reproducibility", "",
        f"- Backend: {power.get('class_status', 'UNKNOWN')}",
        f"- k range: {params.get('k_min')} to {params.get('k_max')} Mpc^-1",
        f"- k samples: {params.get('k_points')}",
        f"- mass range: 10^{params.get('mass_min_exp')} to 10^{params.get('mass_max_exp')} h^-1 Msun",
        f"- mass samples: {params.get('mass_points')}",
        f"- window: {sigma.get('window_type', params.get('window_type'))}",
        f"- integration: fixed log-k Simpson rule on the sampled CLASS grid",
        "", "## Notes", "",
    ]
    lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines).strip() + "\n"
