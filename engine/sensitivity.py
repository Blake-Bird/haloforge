"""Transparent finite-difference sensitivity analysis over saved experiments."""

from __future__ import annotations

import numpy as np


_IGNORED_PARAMETERS = {"mode", "z_presets_selected", "custom_z_list", "z_values"}


def changed_parameter_names(baseline: dict, candidate: dict) -> set[str]:
    """Return submitted parameter changes, excluding UI-only redshift selection."""
    b = baseline.get("params", baseline)
    c = candidate.get("params", candidate)
    return {
        key
        for key in set(b) | set(c)
        if key not in _IGNORED_PARAMETERS and b.get(key) != c.get(key)
    }


def sigma_at_mass(run: dict, mass_hinv_msun: float, redshift: float) -> float:
    arrays = run.get("arrays", {})
    mass = np.asarray(arrays.get("M_h"), dtype=float)
    redshifts = np.asarray(arrays.get("redshifts", [0.0]), dtype=float)
    sigma = np.asarray(arrays.get("sigma_by_z", [arrays.get("sigma")]), dtype=float)
    if (
        mass.ndim != 1
        or mass.size < 2
        or np.any(mass <= 0)
        or np.any(np.diff(mass) <= 0)
    ):
        raise ValueError("A strictly increasing saved mass grid is required")
    index = int(np.argmin(np.abs(redshifts - float(redshift))))
    if not np.isclose(redshifts[index], redshift, rtol=0.0, atol=1e-9):
        raise ValueError("This run does not include the requested redshift")
    row = sigma[index]
    if row.shape != mass.shape or np.any(row <= 0) or np.any(~np.isfinite(row)):
        raise ValueError("The saved sigma array is unavailable or invalid")
    return float(np.exp(np.interp(np.log(mass_hinv_msun), np.log(mass), np.log(row))))


def sigma8_at_redshift(run: dict, redshift: float) -> float:
    arrays = run.get("arrays", {})
    redshifts = np.asarray(arrays.get("redshifts", [0.0]), dtype=float)
    sigma8 = np.asarray(arrays.get("sigma8_pipeline_by_z"), dtype=float)
    index = int(np.argmin(np.abs(redshifts - float(redshift))))
    if sigma8.shape != redshifts.shape or not np.isclose(
        redshifts[index], redshift, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "This run does not include a saved sigma8 value at the requested redshift"
        )
    return float(sigma8[index])


def controlled_sensitivity(
    baseline: dict,
    candidates: list[dict],
    parameter: str,
    observable: str,
    mass_hinv_msun: float | None = None,
    redshift: float = 0.0,
) -> dict:
    """Compute finite changes only for runs that vary exactly one parameter.

    Sensitivity is reported as a finite fractional response, not a derivative
    claim outside the sampled parameter interval.
    """
    base_params = baseline.get("params", {})
    base_value = base_params.get(parameter)
    if (
        not isinstance(base_value, (int, float))
        or isinstance(base_value, bool)
        or float(base_value) == 0
    ):
        raise ValueError("Sensitivity requires a non-zero numeric baseline parameter")
    if observable == "σ(M)":
        if mass_hinv_msun is None or mass_hinv_msun <= 0:
            raise ValueError("σ(M) sensitivity requires a positive selected mass")

        def evaluate(run: dict) -> float:
            return sigma_at_mass(run, mass_hinv_msun, redshift)

    elif observable == "σ₈":

        def evaluate(run: dict) -> float:
            return sigma8_at_redshift(run, redshift)

    else:
        raise ValueError("Unsupported sensitivity observable")
    baseline_observable = evaluate(baseline)
    rows, excluded = [], []
    for candidate in candidates:
        changes = changed_parameter_names(baseline, candidate)
        if changes != {parameter}:
            excluded.append(
                {
                    "run": candidate.get("name", "Untitled run"),
                    "reason": "Not a one-parameter comparison: "
                    + (", ".join(sorted(changes)) or "no changed parameter"),
                }
            )
            continue
        candidate_value = candidate.get("params", {}).get(parameter)
        if not isinstance(candidate_value, (int, float)) or isinstance(
            candidate_value, bool
        ):
            excluded.append(
                {
                    "run": candidate.get("name", "Untitled run"),
                    "reason": "The selected parameter is not numeric in this candidate.",
                }
            )
            continue
        try:
            y = evaluate(candidate)
        except ValueError as exc:
            excluded.append(
                {"run": candidate.get("name", "Untitled run"), "reason": str(exc)}
            )
            continue
        parameter_fraction = (float(candidate_value) / float(base_value)) - 1.0
        response_fraction = (y / baseline_observable) - 1.0
        rows.append(
            {
                "run": candidate.get("name", "Untitled run"),
                "parameter_value": float(candidate_value),
                "parameter_fractional_change": parameter_fraction,
                "observable_value": y,
                "observable_fractional_change": response_fraction,
                "finite_response_ratio": response_fraction / parameter_fraction
                if parameter_fraction
                else float("nan"),
            }
        )
    return {
        "parameter": parameter,
        "observable": observable,
        "baseline_value": float(base_value),
        "baseline_observable": baseline_observable,
        "redshift": float(redshift),
        "mass_hinv_msun": float(mass_hinv_msun) if mass_hinv_msun else None,
        "rows": sorted(rows, key=lambda row: row["parameter_value"]),
        "excluded": excluded,
        "scope_limit": "Finite responses describe only saved, controlled one-parameter experiments. They are not a global derivative, uncertainty posterior, or surrogate-model prediction.",
    }


def smallest_saved_counterfactual(
    report: dict, target_fractional_change: float, direction: str = "either"
) -> dict:
    """Find the least parameter movement among saved runs that reaches a target.

    This deliberately searches actual controlled experiments only. It neither
    interpolates a new parameter setting nor claims the threshold would hold
    outside the stored finite-difference points.
    """
    target = float(target_fractional_change)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Target fractional change must be finite and positive")
    if direction not in {"increase", "decrease", "either"}:
        raise ValueError("Direction must be increase, decrease, or either")
    qualifying = []
    for row in report.get("rows", []):
        response = float(row.get("observable_fractional_change", float("nan")))
        parameter_change = float(row.get("parameter_fractional_change", float("nan")))
        if not (np.isfinite(response) and np.isfinite(parameter_change)):
            continue
        reaches_target = (
            response >= target
            if direction == "increase"
            else response <= -target
            if direction == "decrease"
            else abs(response) >= target
        )
        if reaches_target:
            qualifying.append(row)
    if not qualifying:
        return {
            "status": "not_reached",
            "target_fractional_change": target,
            "direction": direction,
            "scope_limit": "No saved controlled experiment reached this target. HaloForge does not interpolate or extrapolate an unrun parameter value.",
        }
    selected = min(
        qualifying,
        key=lambda row: (
            abs(float(row["parameter_fractional_change"])),
            abs(float(row["observable_fractional_change"]) - target),
        ),
    )
    return {
        "status": "saved_match",
        "target_fractional_change": target,
        "direction": direction,
        "selected": selected,
        "scope_limit": "This is the smallest qualifying parameter movement among saved controlled experiments only; it is not an inferred continuous threshold or an emulator prediction.",
    }
