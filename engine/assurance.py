"""Explicit scientific-assurance states; a calculated curve is not a verdict."""

from __future__ import annotations


def assurance_report(run: dict, hmf_validity: dict | None = None) -> list[dict]:
    """Return distinct, deliberately conservative claims for the current run."""
    arrays = run.get("arrays", {})
    numerical = run.get("numerical_diagnostics", {})
    if not numerical:
        numerical = run.get("sigma_result", {}).get("numerical_diagnostics", {})
    coverage = numerical.get("coverage", {})
    endpoints = [
        numerical.get("high_k_truncation", {}),
        numerical.get("low_k_truncation", {}),
    ]
    endpoint_ok = all(item.get("status") == "low_sensitivity" for item in endpoints)
    range_ok = coverage.get("status") == "range_looks_adequate"
    computed = bool(
        len(arrays.get("k", run.get("power_result", {}).get("k", [])))
        and run.get("class_status", run.get("power_result", {}).get("class_status"))
        == "AXICLASS"
    )
    calibration_ok = bool(
        hmf_validity
        and hmf_validity.get("calibrated_mask") is not None
        and all(hmf_validity["calibrated_mask"])
    )
    params = run.get("params", {})
    empirical = params.get("fitting") not in {
        "Press-Schechter 1974",
        "Sheth-Tormen 2001",
    }
    return [
        {
            "claim": "Computed precisely",
            "state": "pass" if computed else "unknown",
            "detail": "AxiCLASS returned the stored sampled arrays."
            if computed
            else "No successful AxiCLASS array payload is available.",
        },
        {
            "claim": "Numerically converged",
            "state": "review" if endpoint_ok and range_ok else "not established",
            "detail": "Endpoint sensitivity is low, but this is still not a full solver/grid convergence proof."
            if endpoint_ok and range_ok
            else "Sampled-range diagnostics require review or are unavailable.",
        },
        {
            "claim": "Physically appropriate",
            "state": "not established",
            "detail": "This requires a question-specific modelling assessment; HaloForge does not infer it from a smooth curve.",
        },
        {
            "claim": "Calibrated by simulations",
            "state": "pass"
            if calibration_ok and empirical
            else ("not applicable" if not empirical else "not established"),
            "detail": "All plotted HMF points are inside the fit contract."
            if calibration_ok and empirical
            else (
                "The selected analytic model is not a simulation calibration."
                if not empirical
                else "At least some HMF points are outside the declared fit contract or could not be checked."
            ),
        },
        {
            "claim": "Supported for this cosmology",
            "state": "review" if calibration_ok else "not established",
            "detail": "Fit-range checks are necessary but do not independently validate every cosmology family."
            if calibration_ok
            else "No supported-cosmology claim is made for this configuration.",
        },
        {
            "claim": "Suitable for publication",
            "state": "not established",
            "detail": "Requires independent reference comparison, documented convergence, calibration relevance, and scientific review.",
        },
    ]
