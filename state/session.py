"""Transactional Streamlit session state for stable HaloForge runs."""

from __future__ import annotations

import json
import threading
from copy import deepcopy

import numpy as np
import streamlit as st

from config.defaults import DEFAULT_PARAMS
from engine.contracts import validate_fit_configuration
from engine.class_runner import ClassRuntimeError, compute_matter_power
from engine.sigma import compute_sigma_result
from engine.parameter_validation import solver_parameter_errors
from state.cache import cache_key, load_cached_power, save_cached_power
from state.run_model import auto_run_name, create_run_from_current_state
from state.run_storage import (
    generate_run_exports,
    get_last_run_id,
    load_all_runs,
    load_draft_params,
    load_run,
    next_color_index,
    save_draft_params,
    save_run,
    set_last_run_id,
    unique_run_name,
)

_COMPUTE_LOCK = threading.Lock()


def _merged_params(source: dict | None) -> dict:
    merged = deepcopy(DEFAULT_PARAMS)
    if source:
        merged.update(source)
    return merged


def _safe_widget_state_update(key: str, value) -> None:
    try:
        st.session_state[key] = value
    except Exception:
        pass


def _run_reference(run: dict) -> dict:
    return {
        "run_id": str(run["run_id"]),
        "name": str(
            run.get(
                "name",
                run.get(
                    "run_name",
                    "Untitled run",
                ),
            )
        ),
        "is_baseline": bool(
            run.get(
                "is_baseline",
                False,
            )
        ),
    }


def _hydrate_run(run: dict) -> None:
    if run.get("integrity_status", {}).get("state") == "invalid":
        raise ValueError(
            "A saved run with failed integrity checks cannot be loaded for analysis."
        )
    arrays = run.get("arrays", {})
    params = _merged_params(run.get("params"))
    power_result = {
        "k": arrays.get("k"),
        "P": arrays.get("P"),
        "P_by_z": arrays.get("P_by_z", np.asarray([arrays.get("P")])),
        "redshifts": arrays.get("redshifts", np.asarray([0.0])),
        "growth_class": arrays.get("growth_class", np.asarray([1.0])),
        "background_omega_m_by_z": arrays.get(
            "background_omega_m_by_z", np.asarray([])
        ),
        "derived": {
            "h": run.get("derived", {}).get("h", float(params["H0"]) / 100.0),
            "Omega_m": params.get("Omega_m"),
            "sigma8": run.get("sigma8"),
        },
        "class_status": run.get("class_status", "UNKNOWN"),
        "warning": run.get("storage_warning", ""),
        "class_error": run.get("class_error", ""),
        "background_warning": run.get("background_warning", ""),
        "class_settings": run.get("class_settings", {}),
        "from_cache": False,
        "loaded_from_run": run["run_id"],
    }
    sigma_result = {
        "M_h": arrays.get("M_h"),
        "M": arrays.get("M"),
        "R": arrays.get("R"),
        "sigma": arrays.get("sigma"),
        "sigma_by_z": arrays.get("sigma_by_z", np.asarray([arrays.get("sigma")])),
        "redshifts": arrays.get("redshifts", np.asarray([0.0])),
        "sigma8_pipeline_by_z": arrays.get("sigma8_pipeline_by_z", np.asarray([])),
        "dlnsigma_dlnM": arrays.get("dlnsigma_dlnM"),
        "dlnsigma_dlnM_by_z": arrays.get(
            "dlnsigma_dlnM_by_z", np.asarray([arrays.get("dlnsigma_dlnM")])
        ),
        "rho0": run.get("rho0"),
        "window_type": run.get("window_type", params.get("window_type", "Top-hat")),
        "delta_c": float(params.get("delta_c", 1.686)),
        "integration_method": run.get("integration_method", "log-k Simpson"),
        "numerical_diagnostics": run.get("numerical_diagnostics", {}),
    }
    st.session_state["params"] = deepcopy(params)
    st.session_state["completed_params"] = deepcopy(params)
    st.session_state["matter_power_result"] = power_result
    st.session_state["matter_power_cache_key"] = cache_key(params)
    st.session_state["sigma_result"] = sigma_result
    st.session_state["loaded_run_id"] = run["run_id"]
    st.session_state["current_run_id"] = run["run_id"]
    st.session_state["last_auto_saved_run"] = _run_reference(run)
    st.session_state["run_name_draft"] = ""


def init_session() -> None:
    if st.session_state.get("haloforge_initialized"):
        return
    st.session_state["draft_recovery_issues"] = []
    try:
        draft = load_draft_params()
        candidate = _merged_params(draft)
        issues = validate_params(candidate) if draft is not None else []
    except (ValueError, TypeError, OverflowError) as exc:
        candidate, issues = _merged_params(None), [str(exc)]
    if issues:
        st.session_state["draft_recovery_issues"] = issues
        candidate = _merged_params(None)
    st.session_state["params"] = candidate
    st.session_state["completed_params"] = None
    st.session_state["matter_power_result"] = None
    st.session_state["matter_power_cache_key"] = None
    st.session_state["loaded_run_id"] = None
    st.session_state["current_run_id"] = None
    st.session_state["last_auto_saved_run"] = None
    st.session_state["sigma_result"] = None
    st.session_state["saved_runs"] = load_all_runs()
    st.session_state["run_name_draft"] = ""
    st.session_state["save_run_notes"] = ""
    st.session_state["haloforge_initialized"] = True

    runs = st.session_state["saved_runs"]
    target_id = get_last_run_id()
    target = load_run(target_id) if target_id else None
    usable = [
        run
        for run in runs
        if run.get("arrays")
        and run.get("integrity_status", {}).get("state") != "invalid"
    ]
    if (
        target is None
        or not target.get("arrays")
        or target.get("integrity_status", {}).get("state") == "invalid"
    ):
        target = usable[-1] if usable else None
    if target is not None and target.get("arrays"):
        _hydrate_run(target)


def get_params() -> dict:
    init_session()
    return st.session_state["params"]


def get_active_params() -> dict:
    init_session()
    return st.session_state.get("completed_params") or st.session_state["params"]


def reset_params() -> None:
    init_session()
    st.session_state["params"] = deepcopy(DEFAULT_PARAMS)
    st.session_state["run_name_draft"] = ""

    save_draft_params(st.session_state["params"])


def clear_loaded_run() -> None:
    init_session()

    st.session_state["completed_params"] = None
    st.session_state["matter_power_result"] = None
    st.session_state["matter_power_cache_key"] = None
    st.session_state["sigma_result"] = None
    st.session_state["loaded_run_id"] = None
    st.session_state["current_run_id"] = None
    st.session_state["last_auto_saved_run"] = None
    st.session_state["run_name_draft"] = ""

    set_last_run_id(None)


def validate_params(params: dict) -> list[str]:
    errors: list[str] = solver_parameter_errors(params)

    def numeric(key: str) -> float | None:
        try:
            if isinstance(params[key], bool):
                raise ValueError
            value = float(params[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"{key} must be a finite numeric value.")
            return None
        if not np.isfinite(value):
            errors.append(f"{key} must be a finite numeric value.")
            return None
        return value

    for key in ("delta_c", "delta_halo"):
        value = numeric(key)
        if value is not None and value <= 0:
            errors.append(f"{key} must be positive.")
    batch = numeric("quad_limit")
    if batch is not None and (not batch.is_integer() or batch < 1):
        errors.append("quad_limit must be a positive integer batch size.")
    if params.get("window_type") not in {"Top-hat", "Gaussian", "Sharp-k"}:
        errors.append("Choose a supported window: Top-hat, Gaussian, or Sharp-k.")
    epoch = numeric("log10_a_c")
    if epoch is not None:
        try:
            scale = 10.0**epoch
            valid_epoch = 0 < scale <= 1 and np.isfinite(1 / scale)
        except (OverflowError, ZeroDivisionError):
            valid_epoch = False
        if not valid_epoch:
            errors.append(
                "log10_a_c must give a positive scale factor no later than today and a finite critical redshift."
            )
    index = numeric("n_EDE")
    if index is not None and (not index.is_integer() or index < 1):
        errors.append("n_EDE must be a positive integer.")
    delta_halo = numeric("delta_halo")
    omega_b, omega_m = numeric("Omega_b"), numeric("Omega_m")
    k_min, k_max = numeric("k_min"), numeric("k_max")
    mass_min, mass_max, selected_mass = (
        numeric("mass_min_exp"),
        numeric("mass_max_exp"),
        numeric("selected_mass_exp"),
    )
    f_ede = numeric("f_EDE") if params.get("enable_ede") else None
    if params.get("fitting") == "Tinker 2008" and (
        delta_halo is None or not 200 <= delta_halo <= 3200
    ):
        errors.append(
            "Tinker 2008 requires halo overdensity between 200 and 3200 relative to mean matter density."
        )
    try:
        validate_fit_configuration(
            params.get("fitting"),
            params.get("mass_definition", "analytic_top_hat"),
            delta_halo if delta_halo is not None else float("nan"),
        )
    except ValueError as exc:
        errors.append(str(exc))
    if omega_b is not None and omega_m is not None and omega_b >= omega_m:
        errors.append(
            "Omega_b must be smaller than Omega_m so Omega_cdm remains positive."
        )
    if k_min is not None and k_max is not None and (k_min <= 0 or k_max <= k_min):
        errors.append("The k range must satisfy 0 < k_min < k_max.")
    k_points = numeric("k_points")
    if k_points is not None and (not k_points.is_integer() or k_points < 50):
        errors.append("At least 50 k samples are required.")
    if mass_min is not None and mass_max is not None and mass_min >= mass_max:
        errors.append("The minimum halo mass must be below the maximum halo mass.")
    mass_points = numeric("mass_points")
    if mass_points is not None and (not mass_points.is_integer() or mass_points < 20):
        errors.append("At least 20 mass samples are required.")
    if (
        selected_mass is not None
        and mass_min is not None
        and mass_max is not None
        and (selected_mass < mass_min or selected_mass > mass_max)
    ):
        errors.append(
            "The inspection mass must lie inside the selected halo-mass range."
        )
    try:
        negative_redshift = any(
            not np.isfinite(float(z)) or float(z) < 0
            for z in params.get("z_values", [])
        )
    except (TypeError, ValueError):
        negative_redshift = True
    if negative_redshift:
        errors.append("Redshifts must be non-negative.")
    if params.get("enable_ede") and (f_ede is None or not 0.0 <= f_ede <= 0.3):
        errors.append("f_EDE must lie between 0 and 0.3.")
    return list(dict.fromkeys(errors))


def _comparison_payload(params: dict) -> str:
    ignored = {"mode"}
    return json.dumps(
        {key: params[key] for key in sorted(params) if key not in ignored},
        sort_keys=True,
        default=str,
    )


def slow_parameters_changed() -> bool:
    init_session()
    completed = st.session_state.get("completed_params")
    return completed is not None and _comparison_payload(
        get_params()
    ) != _comparison_payload(completed)


def run_new_cosmology(
    requested_name: str = "", notes: str = "", notebook: dict | None = None
) -> dict:
    """Compute P(k,z) and sigma(M,z) transactionally, then atomically auto-save."""
    init_session()
    params = deepcopy(get_params())
    errors = validate_params(params)
    if errors:
        raise ValueError("\n".join(errors))
    save_draft_params(params)

    if not _COMPUTE_LOCK.acquire(blocking=False):
        raise ClassRuntimeError(
            "Another AxiCLASS solve is already running. Let it finish before starting a second run."
        )
    try:
        cached = load_cached_power(params)
        if cached is not None:
            power_result = cached
            power_result["from_cache"] = True
        else:
            power_result = compute_matter_power(params)
            power_result["from_cache"] = False
            save_cached_power(params, power_result)

        sigma_result = compute_sigma_result(power_result, params)
        current = {
            "params": deepcopy(params),
            "power_result": power_result,
            "sigma_result": sigma_result,
            "hash": cache_key(params),
        }
        runs = load_all_runs()
        base_name = requested_name.strip() or auto_run_name(params, len(runs) + 1)
        name = unique_run_name(base_name, runs)
        run = create_run_from_current_state(
            current,
            name=name,
            notes=notes.strip() or "Auto-saved after a completed AxiCLASS run.",
            color_index=next_color_index(),
            is_baseline=not any(existing.get("is_baseline") for existing in runs),
            notebook=notebook,
        )
        run["hash"] = cache_key(params)
        save_run(run)
        generate_run_exports(run)
        set_last_run_id(run["run_id"])

        st.session_state["completed_params"] = deepcopy(params)
        st.session_state["matter_power_result"] = power_result
        st.session_state["matter_power_cache_key"] = cache_key(params)
        st.session_state["sigma_result"] = sigma_result
        st.session_state["loaded_run_id"] = run["run_id"]
        st.session_state["current_run_id"] = run["run_id"]
        st.session_state["last_auto_saved_run"] = _run_reference(run)
        st.session_state["run_name_draft"] = ""
        refresh_saved_runs()
        power_result["saved_run"] = {
            "run_id": run["run_id"],
            "name": run["name"],
            "is_baseline": run.get("is_baseline", False),
        }
        return power_result
    finally:
        _COMPUTE_LOCK.release()


def get_matter_power_result() -> dict | None:
    init_session()
    return st.session_state.get("matter_power_result")


def get_sigma_result() -> dict | None:
    init_session()
    return st.session_state.get("sigma_result")


def current_pipeline_run() -> dict | None:
    power_result = get_matter_power_result()
    sigma_result = get_sigma_result()
    completed = st.session_state.get("completed_params")
    if power_result is None or sigma_result is None or completed is None:
        return None
    return {
        "params": deepcopy(completed),
        "power_result": power_result,
        "sigma_result": sigma_result,
    }


def refresh_saved_runs() -> list[dict]:
    st.session_state["saved_runs"] = load_all_runs()
    return st.session_state["saved_runs"]


def save_current_run(name: str, notes: str = "") -> dict | None:
    current = current_pipeline_run()
    if current is None:
        return None
    runs = load_all_runs()
    unique_name = unique_run_name(
        name.strip() or auto_run_name(current["params"], len(runs) + 1), runs
    )
    run = create_run_from_current_state(
        current,
        unique_name,
        notes,
        next_color_index(),
        not any(r.get("is_baseline") for r in runs),
    )
    save_run(run)
    generate_run_exports(run)
    set_last_run_id(run["run_id"])
    refresh_saved_runs()
    st.session_state["current_run_id"] = run["run_id"]
    st.session_state["loaded_run_id"] = run["run_id"]
    st.session_state["last_auto_saved_run"] = _run_reference(run)
    return run


def load_run_into_session(run_id: str) -> dict | None:
    init_session()
    run = load_run(run_id)
    if (
        run is None
        or not run.get("arrays")
        or run.get("integrity_status", {}).get("state") == "invalid"
    ):
        return None
    _hydrate_run(run)
    set_last_run_id(run_id)
    save_draft_params(run.get("params", {}))
    _safe_widget_state_update("save_run_notes", run.get("notes", ""))
    return run


def get_all_available_runs() -> list[dict]:
    return refresh_saved_runs()


def set_current_run(run_id: str) -> dict | None:
    return load_run_into_session(run_id)
