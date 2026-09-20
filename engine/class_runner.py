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
from engine.parameter_validation import solver_parameter_errors
from engine.power_contract import validate_power_arrays


class ClassRuntimeError(RuntimeError):
    """Raised when the real CLASS/AxiCLASS backend cannot complete a run."""


def _positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a bounded integer setting without making a malformed env fatal."""
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _z_values(params: dict) -> list[float]:
    values = {0.0, float(params.get("single_z", 0.0))}
    values.update(float(z) for z in params.get("z_values", [0.0]))
    if any(not np.isfinite(z) or z < 0 for z in values):
        raise ValueError("CLASS redshifts must be finite and nonnegative")
    return sorted(values)


def _has_ede(params: dict) -> bool:
    """The exact zero-fraction boundary is ΛCDM, without scalar-field shooting."""
    if not params.get("enable_ede", False):
        return False
    fraction = float(params["f_EDE"])
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError("EDE fraction must be finite and nonnegative")
    return fraction > 0


def build_class_settings(params: dict) -> dict[str, Any]:
    errors = solver_parameter_errors(params)
    if errors:
        raise ValueError("\n".join(errors))
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
    if _has_ede(params):
        settings.update(
            {
                "scf_potential": "axion",
                "n_axion": int(params["n_EDE"]),
                "log10_axion_ac": float(params["log10_a_c"]),
                "fraction_axion_ac": float(params["f_EDE"]),
                "scf_parameters": str(params.get("scf_parameters", "2.806,0.0")),
                "do_shooting": "yes",
                "do_shooting_scf": "yes",
                "scf_has_perturbations": "yes",
                "attractor_ic_scf": "no",
            }
        )
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


def _background_omega_m_by_z(
    cosmo: Any, redshifts: np.ndarray, omega_m0: float
) -> np.ndarray:
    """Obtain Ωm(z) from CLASS/AxiCLASS's own expansion history.

    The ratio H(z)/H(0) is unit-free, so this handles the CLASS background
    H column regardless of whether a build reports Mpc^-1 or km/s/Mpc.
    """
    get_background = getattr(cosmo, "get_background", None)
    if not callable(get_background):
        raise ClassRuntimeError(
            "CLASS did not expose a background table needed for Ωm(z)."
        )
    background = get_background()
    z_key = next((key for key in background if key.strip() == "z"), None)
    h_key = next((key for key in background if key.startswith("H [")), None)
    if z_key is None or h_key is None:
        raise ClassRuntimeError(
            "CLASS background table is missing z or H(z), so Ωm(z) cannot be recovered."
        )
    z_grid = np.asarray(background[z_key], dtype=float)
    hubble = np.asarray(background[h_key], dtype=float)
    if z_grid.ndim != 1 or hubble.shape != z_grid.shape or z_grid.size < 2:
        raise ClassRuntimeError("CLASS background table has an invalid H(z) grid.")
    order = np.argsort(z_grid)
    z_grid, hubble = z_grid[order], hubble[order]
    if (
        np.any(~np.isfinite(z_grid))
        or np.any(~np.isfinite(hubble))
        or np.any(hubble <= 0)
    ):
        raise ClassRuntimeError("CLASS background table contains invalid H(z) values.")
    if float(redshifts.min()) < z_grid[0] or float(redshifts.max()) > z_grid[-1]:
        raise ClassRuntimeError(
            "Requested redshift falls outside the CLASS background table."
        )
    h0 = float(np.interp(0.0, z_grid, hubble))
    hz = np.interp(redshifts, z_grid, hubble)
    omega_m_z = float(omega_m0) * (1.0 + redshifts) ** 3 / (hz / h0) ** 2
    if np.any(~np.isfinite(omega_m_z)) or np.any(omega_m_z <= 0):
        raise ClassRuntimeError("CLASS background produced an invalid Ωm(z) sequence.")
    return omega_m_z


def _compute_direct(params: dict) -> dict[str, Any]:
    try:
        import classy  # type: ignore

        Class = classy.Class
    except Exception as exc:
        raise ClassRuntimeError(
            f"The classy binding could not be imported: {exc}"
        ) from exc

    k = np.logspace(
        np.log10(float(params["k_min"])),
        np.log10(float(params["k_max"])),
        int(params["k_points"]),
    )
    redshifts = np.asarray(_z_values(params), dtype=float)
    settings = build_class_settings(params)
    cosmo = Class()
    try:
        cosmo.set(settings)
        cosmo.compute()
        p_by_z = np.empty((len(redshifts), len(k)), dtype=float)
        for j, z in enumerate(redshifts):
            p_by_z[j] = np.fromiter(
                (cosmo.pk(float(ki), float(z)) for ki in k), dtype=float, count=len(k)
            )
        if not np.all(np.isfinite(p_by_z)) or np.any(p_by_z <= 0.0):
            raise ClassRuntimeError(
                "AxiCLASS returned non-finite or non-positive P(k,z)."
            )

        derived = cosmo.get_current_derived_parameters(["h", "Omega_m", "sigma8"])
        growth0 = float(cosmo.scale_independent_growth_factor(0.0))
        growth = np.asarray(
            [
                float(cosmo.scale_independent_growth_factor(float(z))) / growth0
                for z in redshifts
            ]
        )
        try:
            background_omega_m = _background_omega_m_by_z(
                cosmo, redshifts, float(derived["Omega_m"])
            )
            background_warning = ""
        except ClassRuntimeError as exc:
            # Most outputs remain valid without a background table. Individual
            # consumers such as EDE Watson SO enforce this dependency.
            background_omega_m = np.asarray([])
            background_warning = str(exc)
        return {
            "k": k,
            "P": p_by_z[0].copy(),
            "P_by_z": p_by_z,
            "redshifts": redshifts,
            "growth_class": growth,
            "background_omega_m_by_z": background_omega_m,
            "background_warning": background_warning,
            "derived": {
                "h": float(derived["h"]),
                "Omega_m": float(derived["Omega_m"]),
                "sigma8": float(derived["sigma8"]),
            },
            "class_status": "AXICLASS" if _has_ede(params) else "CLASS",
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


def _read_worker_result(result_path: Path) -> dict[str, Any]:
    """Load a completed worker artifact, rejecting incomplete/corrupt results."""
    if not result_path.exists():
        raise ClassRuntimeError(
            "The isolated CLASS worker exited without producing a result file."
        )
    try:
        with np.load(result_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if not isinstance(metadata, dict) or any(
                key in metadata
                for key in (
                    "k",
                    "P",
                    "P_by_z",
                    "redshifts",
                    "growth_class",
                    "background_omega_m_by_z",
                )
            ):
                raise ValueError("Worker metadata must not redefine scientific arrays")
            result = {
                "k": data["k"],
                "P": data["P"],
                "P_by_z": data["P_by_z"],
                "redshifts": data["redshifts"],
                "growth_class": data["growth_class"],
                "background_omega_m_by_z": data["background_omega_m_by_z"]
                if "background_omega_m_by_z" in data.files
                else np.asarray([]),
                **metadata,
            }
        validate_power_arrays(result)
        return result
    except ClassRuntimeError:
        raise
    except Exception as exc:
        raise ClassRuntimeError(
            f"The isolated CLASS result could not be read: {exc}"
        ) from exc


def compute_matter_power(params: dict) -> dict[str, Any]:
    """Run CLASS in a dedicated worker process so native failures cannot kill Streamlit."""
    errors = solver_parameter_errors(params)
    if errors:
        raise ValueError("\n".join(errors))
    loaded_classy = sys.modules.get("classy")
    if loaded_classy is not None and not getattr(loaded_classy, "__file__", None):
        return _compute_direct(params)

    timeout = _positive_int_env("HALOFORGE_CLASS_TIMEOUT_SECONDS", 1800)
    # A retry is deliberately limited to infrastructure-like failures.  A CLASS
    # input error is deterministic and must be shown immediately, not hidden by
    # rerunning the same invalid cosmology.
    transient_retries = _positive_int_env(
        "HALOFORGE_CLASS_TRANSIENT_RETRIES", 1, minimum=0
    )
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="haloforge_class_") as temp_dir:
        temp = Path(temp_dir)
        params_path = temp / "params.json"
        result_path = temp / "result.npz"
        params_path.write_text(json.dumps(params, default=str), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "engine.class_worker",
            str(params_path),
            str(result_path),
        ]
        for attempt in range(1, transient_retries + 2):
            # A prior attempt cannot leave a stale artifact that looks valid.
            result_path.unlink(missing_ok=True)
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
                    f"AxiCLASS exceeded the {timeout}-second safety timeout and was stopped; no retry was attempted to avoid repeating an over-budget calculation. The previous completed run is still intact."
                ) from exc

            if completed.returncode == 0:
                try:
                    result = _read_worker_result(result_path)
                    try:
                        validate_power_arrays(result, params)
                    except ValueError as exc:
                        raise ClassRuntimeError(str(exc)) from exc
                    result["solver_execution"] = {
                        "isolation": "dedicated subprocess worker",
                        "timeout_seconds": timeout,
                        "attempts": attempt,
                        "transient_retry_limit": transient_retries,
                    }
                    return result
                except ClassRuntimeError as exc:
                    failure = str(exc)
            else:
                details = (
                    (completed.stderr or completed.stdout or "").strip().splitlines()
                )
                final_line = (
                    details[-1]
                    if details
                    else f"worker exit code {completed.returncode}"
                )
                if completed.returncode < 0:
                    final_line = f"worker terminated by signal {-completed.returncode}"
                    failure = f"The isolated CLASS worker failed without crashing HaloForge: {final_line}"
                else:
                    # Worker-reported failures come from CLASS itself and are
                    # never retried: retrying invalid settings masks the cause.
                    raise ClassRuntimeError(
                        f"The isolated CLASS worker failed without crashing HaloForge: {final_line}"
                    )

            if attempt > transient_retries:
                raise ClassRuntimeError(
                    f"CLASS worker infrastructure failed after {attempt} attempt(s) (retry limit {transient_retries}): {failure}"
                )


def classy_import_diagnostics() -> dict[str, Any]:
    try:
        module = importlib.import_module("classy")
        return {
            "imports": hasattr(module, "Class"),
            "path": getattr(module, "__file__", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "imports": False,
            "path": "",
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }


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
        "class_timeout_seconds": _positive_int_env(
            "HALOFORGE_CLASS_TIMEOUT_SECONDS", 1800
        ),
        "class_transient_retry_limit": _positive_int_env(
            "HALOFORGE_CLASS_TRANSIENT_RETRIES", 1, minimum=0
        ),
    }


def tiny_class_smoke_test(params: dict) -> dict[str, Any]:
    smoke = dict(params)
    smoke.update(
        enable_ede=False,
        k_min=1e-3,
        k_max=1.0,
        k_points=10,
        single_z=0.0,
        z_values=[0.0],
    )
    try:
        result = compute_matter_power(smoke)
        index = int(np.argmin(np.abs(result["k"] - 0.1)))
        if not np.isclose(result["k"][index], 0.1, rtol=1e-12, atol=0):
            raise ValueError(
                "The smoke-test output is missing its requested k = 0.1 Mpc^-1 sample"
            )
        value = float(result["P"][index])
        return {
            "ok": True,
            "message": f"Real CLASS passed in its isolated worker: P(0.1,0)={value:.6e} Mpc^3",
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}
