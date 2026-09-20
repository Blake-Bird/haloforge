"""Explicit scientific-assurance states; a calculated curve is not a verdict."""

from __future__ import annotations

from engine.contracts import all_fit_points_checked


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
        in {"CLASS", "AXICLASS"}
        and run.get("integrity_status", {}).get("state") != "invalid"
    )
    calibration_ok = all_fit_points_checked(hmf_validity)
    params = run.get("params", {})
    analytic = params.get("fitting") in {"Press-Schechter", "Press-Schechter 1974"}
    sheth_tormen = params.get("fitting") in {"Sheth-Tormen", "Sheth-Tormen 2001"}
    return [
        {
            "claim": "Solver completed",
            "state": "pass" if computed else "unknown",
            "detail": "CLASS/AxiCLASS returned the stored sampled arrays; completion alone does not establish accuracy."
            if computed
            else "No usable completed CLASS/AxiCLASS payload is available.",
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
            "state": "not applicable"
            if analytic
            else "review"
            if calibration_ok
            else "not established",
            "detail": "Press-Schechter is an analytic collapse model, not a simulation fit."
            if analytic
            else "Sheth-Tormen includes simulation-fitted coefficients. Evaluation of its formula does not establish calibration for this mass range or cosmology."
            if sheth_tormen
            else (
                "The evaluated points pass the implemented fit-range checks. Simulation calibration still requires matching the halo definition, cosmology, and published domain."
                if calibration_ok
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
