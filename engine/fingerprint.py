"""Compact, provenance-friendly summaries of a controlled cosmology change."""

from __future__ import annotations

import numpy as np


FINGERPRINT_VERSION = "haloforge-cosmic-fingerprint-v1"
_PARAMETER_LABELS = {
    "A_s": "Primordial amplitude Aₛ",
    "n_s": "Primordial tilt nₛ",
    "Omega_m": "Matter density Ωₘ",
    "f_EDE": "Early-dark-energy fraction",
    "log10_a_c": "EDE critical epoch log₁₀aᶜ",
    "Omega_k": "Curvature Ωₖ",
}


def _finite_scalar(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _sample_positive(x, y, point: float) -> float | None:
    x, y = (np.asarray(values, dtype=float) for values in (x, y))
    if (
        x.ndim != 1
        or y.shape != x.shape
        or x.size < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        return None
    if point < x[0] or point > x[-1]:
        return None
    if point > 0 and np.all(x > 0) and np.all(y > 0):
        return float(np.exp(np.interp(np.log(point), np.log(x), np.log(y))))
    return float(np.interp(point, x, y))


def _relative_signal(
    label: str, baseline: float | None, candidate: float | None, context: str
) -> dict | None:
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return {
            "label": label,
            "baseline": baseline,
            "candidate": candidate,
            "fractional_change": None,
            "context": context,
        }
    return {
        "label": label,
        "baseline": baseline,
        "candidate": candidate,
        "fractional_change": float(candidate / baseline - 1),
        "context": context,
    }


def cosmic_fingerprint(candidate: dict, baseline: dict | None = None) -> dict:
    """Summarize only stored baseline/candidate differences.

    The returned normalized signal rows are visual-summary inputs, not a
    posterior, calibration statement, or literal universe simulation.
    """
    candidate_params = candidate.get("params", {})
    baseline_params = (baseline or {}).get("params", {})
    parameter_changes = []
    if baseline:
        for key, label in _PARAMETER_LABELS.items():
            before, after = (
                _finite_scalar(baseline_params.get(key)),
                _finite_scalar(candidate_params.get(key)),
            )
            if before is not None and after is not None and before != after:
                parameter_changes.append(
                    {
                        "parameter": key,
                        "label": label,
                        "baseline": before,
                        "candidate": after,
                        "fractional_change": None
                        if before == 0
                        else float(after / before - 1),
                        "absolute_change": float(after - before),
                    }
                )

    signals = []
    if baseline:
        baseline_arrays, candidate_arrays = (
            baseline.get("arrays", {}),
            candidate.get("arrays", {}),
        )
        for label, key, context in (
            ("σ₈", "sigma8", "Linear fluctuation amplitude on 8 h⁻¹ Mpc scales."),
        ):
            signal = _relative_signal(
                label,
                _finite_scalar(baseline.get(key)),
                _finite_scalar(candidate.get(key)),
                context,
            )
            if signal:
                signals.append(signal)
        pivot = _finite_scalar(candidate_params.get("k_pivot"))
        if pivot is not None:
            signal = _relative_signal(
                "P(kₚ)",
                _sample_positive(
                    baseline_arrays.get("k", []), baseline_arrays.get("P", []), pivot
                ),
                _sample_positive(
                    candidate_arrays.get("k", []), candidate_arrays.get("P", []), pivot
                ),
                "Stored linear matter power at the candidate pivot scale.",
            )
            if signal:
                signals.append(signal)
        selected_mass = 10 ** float(candidate_params.get("selected_mass_exp", 12))
        signal = _relative_signal(
            "σ(selected mass)",
            _sample_positive(
                baseline_arrays.get("M_h", []),
                baseline_arrays.get("sigma", []),
                selected_mass,
            ),
            _sample_positive(
                candidate_arrays.get("M_h", []),
                candidate_arrays.get("sigma", []),
                selected_mass,
            ),
            f"Stored linear σ(M) at {selected_mass:.0e} h⁻¹ M☉.",
        )
        if signal:
            signals.append(signal)

    return {
        "version": FINGERPRINT_VERSION,
        "has_baseline": bool(baseline),
        "parameter_changes": parameter_changes,
        "signals": signals,
        "scope_limit": "This compact fingerprint visualizes stored linear inputs and observables only. It is not a literal simulated universe, an uncertainty estimate, an observational result, or evidence that an empirical HMF is calibrated for the candidate cosmology.",
    }


def fingerprint_markdown(fingerprint: dict) -> str:
    """Render a compact text equivalent for bundles and assistive workflows."""
    lines = ["### Cosmic fingerprint", ""]
    if not fingerprint.get("has_baseline"):
        lines.append(
            "No named baseline is attached, so this standalone run has no controlled-change fingerprint."
        )
    elif fingerprint.get("parameter_changes"):
        lines.extend(
            [
                f"- {row['label']}: {row['baseline']:.6g} → {row['candidate']:.6g}"
                for row in fingerprint["parameter_changes"]
            ]
        )
    else:
        lines.append(
            "No tracked scalar parameter change was found relative to the selected baseline."
        )
    if fingerprint.get("signals"):
        lines.extend(["", "**Stored linear response**"])
        for signal in fingerprint["signals"]:
            change = (
                "undefined (zero baseline)"
                if signal["fractional_change"] is None
                else f"{signal['fractional_change']:+.2%}"
            )
            lines.append(f"- {signal['label']}: {change}. {signal['context']}")
    lines.extend(["", "**Boundary:** " + fingerprint["scope_limit"]])
    return "\n".join(lines) + "\n"
