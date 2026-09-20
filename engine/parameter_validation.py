"""Physical input contracts shared by the UI and direct solver entry points."""

from __future__ import annotations

import math


def solver_parameter_errors(params: dict) -> list[str]:
    """Reject malformed inputs without imposing the sidebar's exploration ranges."""
    errors: list[str] = []

    def number(key, default=None, *, positive=False, nonnegative=False):
        raw = params.get(key, default)
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{key} must be a finite numeric value.")
            return None
        if (positive and value <= 0) or (nonnegative and value < 0):
            bound = "positive" if positive else "nonnegative"
            errors.append(f"{key} must be {bound}.")
        return value

    for key in ("H0", "A_s"):
        number(key, positive=True)
    number("k_pivot", 0.05, positive=True)
    number("Tcmb", 2.7255, positive=True)
    number("n_s")
    number("Omega_k", 0.0)
    number("N_eff", 3.046, nonnegative=True)
    number("tau_reio", nonnegative=True)
    matter = number("Omega_m", positive=True)
    baryons = number("Omega_b", nonnegative=True)
    if matter is not None and baryons is not None and baryons >= matter:
        errors.append(
            "Omega_b must be smaller than Omega_m so Omega_cdm remains positive."
        )
    lower, upper = number("k_min", positive=True), number("k_max", positive=True)
    if lower is not None and upper is not None and lower >= upper:
        errors.append("The k range must satisfy 0 < k_min < k_max.")
    samples = number("k_points")
    if samples is not None and (not samples.is_integer() or samples < 2):
        errors.append("k_points must be an integer of at least 2.")
    number("single_z", 0.0, nonnegative=True)
    try:
        redshifts = params.get("z_values", [0.0])
        if isinstance(redshifts, (str, bytes, dict)):
            raise ValueError
        if any(
            isinstance(z, bool) or not math.isfinite(float(z)) or float(z) < 0
            for z in redshifts
        ):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        errors.append("CLASS redshifts must be finite and nonnegative.")
    enabled = params.get("enable_ede", False)
    if not isinstance(enabled, bool):
        errors.append("enable_ede must be a boolean.")
    if enabled is True:
        fraction = number("f_EDE", nonnegative=True)
        if fraction is not None and fraction >= 1:
            errors.append("f_EDE must be smaller than 1.")
        if fraction is not None and fraction > 0:
            index = number("n_EDE")
            if index is not None and (not index.is_integer() or index < 1):
                errors.append("n_EDE must be a positive integer.")
            epoch = number("log10_a_c")
            if epoch is not None:
                try:
                    scale = 10.0**epoch
                    valid_epoch = 0 < scale <= 1 and math.isfinite(1 / scale)
                except (OverflowError, ZeroDivisionError):
                    valid_epoch = False
                if not valid_epoch:
                    errors.append(
                        "log10_a_c must give a positive scale factor no later than today and a finite critical redshift."
                    )
            try:
                fields = str(params.get("scf_parameters", "2.806,0.0")).split(",")
                if len(fields) != 2 or not all(
                    math.isfinite(float(value)) for value in fields
                ):
                    raise ValueError
            except (ValueError, OverflowError):
                errors.append(
                    "scf_parameters must contain two finite numbers: initial field and its derivative."
                )
    return errors
