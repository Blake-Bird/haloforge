"""Crash-isolated strict AxiCLASS/CLASS execution layer."""

from __future__ import annotations

import gc
import importlib
import os
import json
import subprocess
import tempfile
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from engine.cosmology import omega_cdm


class ClassRuntimeError(RuntimeError):
    """Raised when the real CLASS/AxiCLASS backend cannot complete a run."""


def _z_values(params: dict) -> list[float]:
    values = {0.0, float(params.get("single_z", 0.0))}
    values.update(float(z) for z in params.get("z_values", [0.0]))
    return sorted(z for z in values if z >= 0.0)


def build_class_settings(params: dict) -> dict[str, Any]:
    z_values = _z_values(params)
    settings: dict[str, Any] = {
        "output": "mPk",
        "H0": float(params["H0"]),
        "Omega_b": float(params["Omega_b"]),
        "Omega_cdm": omega_cdm(params),
        "Omega_k": float(params.get("Omega_k", 0.0)),
        "A_s": float(params["A_s"]),
        "n_s": float(params["n_s"]),
        "k_pivot": float(params.get("k_pivot", 0.05)),
        "tau_reio": float(params["tau_reio"]),
        "T_cmb": float(params.get("Tcmb", 2.7255)),
        "N_ur": float(params.get("N_eff", 3.046)),
        "P_k_max_1/Mpc": 1.05 * float(params["k_max"]),
        "z_max_pk": max(z_values),
        "z_pk": ",".join(f"{z:.12g}" for z in z_values),
        "modes": "s",
        "gauge": "synchronous",
    }
    if params.get("enable_ede", False):
        settings.update({
            "scf_potential": "axion",
            "n_axion": int(params["n_EDE"]),
            "log10_axion_ac": float(params["log10_a_c"]),
            "fraction_axion_ac": float(params["f_EDE"]),
            "scf_parameters": str(params.get("scf_parameters", "2.806,0.0")),
            "do_shooting": "yes",
            "do_shooting_scf": "yes",
            "scf_has_perturbations": "yes",
            "attractor_ic_scf": "no",
        })
    return settings


def _cleanup(cosmo: Any) -> None:
    for name in ("struct_cleanup", "empty"):
        method = getattr(cosmo, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    gc.collect()


def _compute_direct(params: dict) -> dict[str, Any]:
    try:
        import classy  # type: ignore
        Class = classy.Class
    except Exception as exc:
        raise ClassRuntimeError(f"The classy binding could not be imported: {exc}") from exc

    k = np.logspace(np.log10(float(params["k_min"])), np.log10(float(params["k_max"])), int(params["k_points"]))
    redshifts = np.asarray(_z_values(params), dtype=float)
    settings = build_class_settings(params)
    cosmo = Class()
    try:
        cosmo.set(settings)
        cosmo.compute()
        p_by_z = np.empty((len(redshifts), len(k)), dtype=float)
        for j, z in enumerate(redshifts):
            p_by_z[j] = np.fromiter((cosmo.pk(float(ki), float(z)) for ki in k), dtype=float, count=len(k))
        if not np.all(np.isfinite(p_by_z)) or np.any(p_by_z <= 0.0):
            raise ClassRuntimeError("AxiCLASS returned non-finite or non-positive P(k,z).")

        derived = cosmo.get_current_derived_parameters(["h", "Omega_m", "sigma8"])
        growth0 = float(cosmo.scale_independent_growth_factor(0.0))
        growth = np.asarray([float(cosmo.scale_independent_growth_factor(float(z))) / growth0 for z in redshifts])
        return {
            "k": k,
            "P": p_by_z[0].copy(),
            "P_by_z": p_by_z,
            "redshifts": redshifts,
            "growth_class": growth,
            "derived": {"h": float(derived["h"]), "Omega_m": float(derived["Omega_m"]), "sigma8": float(derived["sigma8"])},
            "class_status": "AXICLASS" if params.get("enable_ede") else "CLASS",
            "class_settings": settings,
            "classy_path": getattr(classy, "__file__", ""),
            "class_error": "",
        }
    except ClassRuntimeError:
        raise
    except Exception as exc:
        model = "AxiCLASS EDE" if params.get("enable_ede") else "CLASS LCDM"
        raise ClassRuntimeError(f"{model} failed: {exc}") from exc
    finally:
        _cleanup(cosmo)


def compute_matter_power(params: dict) -> dict[str, Any]:
    """Run CLASS in a dedicated worker process so native failures cannot kill Streamlit."""
    loaded_classy = sys.modules.get("classy")
    if loaded_classy is not None and not getattr(loaded_classy, "__file__", None):
        return _compute_direct(params)

    timeout = int(os.environ.get("HALOFORGE_CLASS_TIMEOUT_SECONDS", "1800"))
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="haloforge_class_") as temp_dir:
        temp = Path(temp_dir)
        params_path = temp / "params.json"
        result_path = temp / "result.npz"
        params_path.write_text(json.dumps(params, default=str), encoding="utf-8")
        command = [sys.executable, "-m", "engine.class_worker", str(params_path), str(result_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClassRuntimeError(
                f"AxiCLASS exceeded the {timeout}-second safety timeout and was stopped; the previous completed run is still intact."
            ) from exc

        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip().splitlines()
            final_line = details[-1] if details else f"worker exit code {completed.returncode}"
            if completed.returncode < 0:
                final_line = f"worker terminated by signal {-completed.returncode}"
            raise ClassRuntimeError(
                f"The isolated CLASS worker failed without crashing HaloForge: {final_line}"
            )
        if not result_path.exists():
            raise ClassRuntimeError("The isolated CLASS worker exited without producing a result file.")
        try:
            with np.load(result_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                return {
                    "k": data["k"],
                    "P": data["P"],
                    "P_by_z": data["P_by_z"],
                    "redshifts": data["redshifts"],
                    "growth_class": data["growth_class"],
                    **metadata,
                }
        except Exception as exc:
            raise ClassRuntimeError(f"The isolated CLASS result could not be read: {exc}") from exc

def classy_import_diagnostics() -> dict[str, Any]:
    try:
        module = importlib.import_module("classy")
        return {"imports": hasattr(module, "Class"), "path": getattr(module, "__file__", ""), "error": ""}
    except Exception as exc:
        return {"imports": False, "path": "", "error": "".join(traceback.format_exception_only(type(exc), exc)).strip()}


def environment_diagnostics() -> dict[str, Any]:
    classy = classy_import_diagnostics()
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "classy_imports": classy["imports"],
        "classy_path": classy["path"],
        "classy_error": classy["error"],
        "cwd": str(Path.cwd()),
        "class_isolation": "dedicated subprocess worker",
        "class_timeout_seconds": int(os.environ.get("HALOFORGE_CLASS_TIMEOUT_SECONDS", "1800")),
    }


def tiny_class_smoke_test(params: dict) -> dict[str, Any]:
    smoke = dict(params)
    smoke.update(enable_ede=False, k_min=1e-3, k_max=1.0, k_points=8, z_values=[0.0])
    try:
        result = compute_matter_power(smoke)
        value = float(np.interp(0.1, result["k"], result["P"]))
        return {"ok": True, "message": f"Real CLASS passed in its isolated worker: P(0.1,0)={value:.6e} Mpc^3"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}
