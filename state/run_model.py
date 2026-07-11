"""Saved run model construction."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import sys
from uuid import uuid4

import numpy as np

from config.defaults import DEFAULT_PARAMS
from config.style import AMBER, CYAN, GREEN, ROSE, VIOLET
from engine.cosmology import derived_quantities
from engine.hmf import cumulative_hmf, hmf_z

RUN_COLORS = [CYAN, VIOLET, AMBER, ROSE, GREEN]


def _array_payload(current: dict) -> dict:
    power = current["power_result"]
    sigma = current["sigma_result"]
    ps_run = {"params": current["params"], "power_result": power, "sigma_result": sigma}
    ps_hmf = hmf_z(ps_run, 0.0, "Press-Schechter 1974")
    st_hmf = hmf_z(ps_run, 0.0, "Sheth-Tormen 2001")
    k = np.asarray(power["k"], dtype=float)
    p = np.asarray(power["P"], dtype=float)
    return {
        "k": k,
        "P": p,
        "P_by_z": np.asarray(power.get("P_by_z", [p]), dtype=float),
        "redshifts": np.asarray(power.get("redshifts", [0.0]), dtype=float),
        "growth_class": np.asarray(power.get("growth_class", [1.0]), dtype=float),
        "Delta2": k**3 * p / (2.0 * np.pi**2),
        "M_h": np.asarray(sigma["M_h"], dtype=float),
        "M": np.asarray(sigma["M"], dtype=float),
        "R": np.asarray(sigma["R"], dtype=float),
        "sigma": np.asarray(sigma["sigma"], dtype=float),
        "sigma_by_z": np.asarray(sigma.get("sigma_by_z", [sigma["sigma"]]), dtype=float),
        "sigma8_pipeline_by_z": np.asarray(sigma.get("sigma8_pipeline_by_z", []), dtype=float),
        "dlnsigma_dlnM": np.asarray(sigma["dlnsigma_dlnM"], dtype=float),
        "dlnsigma_dlnM_by_z": np.asarray(sigma.get("dlnsigma_dlnM_by_z", [sigma["dlnsigma_dlnM"]]), dtype=float),
        "dlog_sigma_dlog_M": np.asarray(sigma["dlnsigma_dlnM"], dtype=float),
        "hmf_press_schechter_z0": np.asarray(ps_hmf["hmf"], dtype=float),
        "hmf_sheth_tormen_z0": np.asarray(st_hmf["hmf"], dtype=float),
        "cumulative_press_schechter_z0": cumulative_hmf(ps_hmf["M_h"], ps_hmf["hmf"]),
        "cumulative_sheth_tormen_z0": cumulative_hmf(st_hmf["M_h"], st_hmf["hmf"]),
    }


def auto_run_name(params: dict, run_number: int) -> str:
    """Build a readable unique-ish default name from the changed cosmology knobs."""
    parts: list[str] = []
    default = DEFAULT_PARAMS
    if abs(float(params.get("A_s", default["A_s"])) - float(default["A_s"])) / float(default["A_s"]) > 0.03:
        parts.append("High A_s" if float(params["A_s"]) > float(default["A_s"]) else "Low A_s")
    if abs(float(params.get("n_s", default["n_s"])) - float(default["n_s"])) > 0.01:
        parts.append("Blue Tilt" if float(params["n_s"]) > float(default["n_s"]) else "Red Tilt")
    if bool(params.get("enable_ede")):
        z_c = 10 ** (-float(params.get("log10_a_c", default["log10_a_c"]))) - 1.0
        parts.append(f"EDE f{float(params.get('f_EDE', 0.0)):.2f} zc{round(z_c):g}")
    if abs(float(params.get("Omega_m", default["Omega_m"])) - float(default["Omega_m"])) > 0.015:
        parts.append(f"Omega_m {float(params['Omega_m']):.3f}")
    if not parts:
        label = "Baseline LCDM"
    elif len(parts) <= 2:
        label = " + ".join(parts)
    else:
        label = "Custom Cosmology"
    return f"Run {run_number:03d} — {label}"


def create_run_from_current_state(
    current: dict,
    name: str,
    notes: str = "",
    color_index: int = 0,
    is_baseline: bool = False,
) -> dict:
    """Create a serializable run metadata object plus in-memory arrays."""
    params = deepcopy(current["params"])
    power = current["power_result"]
    sigma = current["sigma_result"]
    run_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    run_name = name.strip() or "Untitled run"
    return {
        "run_id": run_id,
        "run_name": run_name,
        "name": run_name,
        "color": RUN_COLORS[color_index % len(RUN_COLORS)],
        "created_at": now,
        "updated_at": now,
        "params": params,
        "derived": derived_quantities(params),
        "arrays": _array_payload(current),
        "notes": notes,
        "visible": True,
        "is_baseline": is_baseline,
        "class_status": power.get("class_status", "UNKNOWN"),
        "classy_path": power.get("classy_path", ""),
        "python_executable": sys.executable,
        "class_error": power.get("class_error", power.get("warning", "")),
        "class_settings": deepcopy(power.get("class_settings", {})),
        "sigma8": power.get("derived", {}).get("sigma8"),
        "rho0": sigma.get("rho0"),
        "window_type": sigma.get("window_type"),
        "exports": {},
        "hash": current.get("hash", ""),
    }
