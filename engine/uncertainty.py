"""Explicit inventory of distinct uncertainty and validity sources."""

from __future__ import annotations

from engine.contracts import all_fit_points_checked


def uncertainty_inventory(
    run: dict, hmf_validity: dict | None = None, benchmark: dict | None = None
) -> list[dict]:
    """Return named evidence states without collapsing them into one number."""
    params = run.get("params", {})
    sigma = run.get("sigma_result", {})
    diagnostics = sigma.get(
        "numerical_diagnostics", run.get("numerical_diagnostics", {})
    )
    endpoint = [
        diagnostics.get("low_k_truncation", {}),
        diagnostics.get("high_k_truncation", {}),
    ]
    endpoints_ok = all(check.get("status") == "low_sensitivity" for check in endpoint)
    benchmark_rows = (benchmark or {}).get("rows", [])
    benchmark_states = {row.get("status") for row in benchmark_rows}
    calibration_ok = all_fit_points_checked(hmf_validity)
    return [
        {
            "source": "Numerical integration",
            "state": "measured"
            if "pass" in benchmark_states
            else ("review" if "review" in benchmark_states else "unquantified"),
            "evidence": "Adaptive σ₈ benchmark is recorded."
            if "pass" in benchmark_states
            else (
                "Adaptive σ₈ benchmark recorded a quadrature warning."
                if "review" in benchmark_states
                else "No saved independent adaptive σ₈ check for this run."
            ),
        },
        {
            "source": "k-range truncation",
            "state": "review" if endpoints_ok else "unquantified",
            "evidence": "Endpoint-removal sensitivity is low on the sampled range."
            if endpoints_ok
            else "Endpoint sensitivity is material, missing, or not a complete convergence proof.",
        },
        {
            "source": "Grid resolution",
            "state": "unquantified",
            "evidence": f"Current run uses {params.get('k_points', 'unknown')} k samples and {params.get('mass_points', 'unknown')} mass samples; a resolution sweep has not been established.",
        },
        {
            "source": "Interpolation behavior",
            "state": "documented",
            "evidence": "Positive P(k) interpolation is logarithmic inside the solved range; out-of-range interpolation is rejected.",
        },
        {
            "source": "CLASS/AxiCLASS settings",
            "state": "recorded",
            "evidence": "Exact submitted solver settings are stored in the run provenance and export bundle.",
        },
        {
            "source": "HMF fit calibration",
            "state": "range-checked" if calibration_ok else "review",
            "evidence": "All evaluated points pass the implemented fit-range checks; this does not establish simulation calibration for the current cosmology."
            if calibration_ok
            else "At least one point is outside the checked fit contract or could not be checked.",
        },
        {
            "source": "Cosmology support",
            "state": "unquantified",
            "evidence": "A fit contract does not independently prove simulation support for every cosmology family, particularly EDE.",
        },
        {
            "source": "Theory/model uncertainty",
            "state": "unquantified",
            "evidence": "This linear calculation does not quantify nonlinear, baryonic, or model-family theory uncertainty.",
        },
        {
            "source": "User assumption sensitivity",
            "state": "available" if params else "unavailable",
            "evidence": "Use controlled saved-run sensitivity analysis to inspect one parameter at a time; it is not a global posterior.",
        },
        {
            "source": "Emulator uncertainty",
            "state": "not applicable",
            "evidence": "HaloForge currently ships no emulator or surrogate prediction layer.",
        },
    ]
