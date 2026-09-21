"""Versioned, transparent numerical benchmark checks for completed runs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from engine.sigma import build_power_interpolator
from engine.windows import window_squared
from engine.saved_run import pipeline_from_saved_run


BENCHMARK_VERSION = "haloforge-internal-sigma8-v1"
CANONICAL_CASES_VERSION = "haloforge-canonical-validation-cases-v2"
DEFAULT_RELATIVE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CanonicalValidationCase:
    """A reproducible coverage target, not a claimed external benchmark."""

    identifier: str
    title: str
    parameter_overrides: dict
    target: str
    expected_evidence: str
    boundary: str


CANONICAL_VALIDATION_CASES = (
    CanonicalValidationCase(
        "lcdm-planck-like",
        "ΛCDM Planck-like baseline",
        {"enable_ede": False, "Omega_k": 0.0, "single_z": 0.0},
        "Baseline linear P(k), σ(M), and analytic HMF pipeline.",
        "Internal σ₈ fixed-grid/adaptive agreement at every requested redshift.",
        "CI compares one fixed flat ΛCDM cosmology with CAMB 1.6.6; this configuration check does not run that comparison.",
    ),
    CanonicalValidationCase(
        "ede-linear",
        "Early dark energy linear evolution",
        {
            "enable_ede": True,
            "f_EDE": 0.10,
            "log10_a_c": -3.5,
            "n_EDE": 3,
            "single_z": 0.0,
        },
        "AxiCLASS linear background and P(k) response for the supported EDE configuration.",
        "Internal numerical checks plus explicit HMF cosmology-support warning.",
        "An EDE HMF curve is not an externally calibrated halo prediction.",
    ),
    CanonicalValidationCase(
        "curved-lcdm",
        "Curved ΛCDM",
        {"enable_ede": False, "Omega_k": 0.01, "single_z": 0.0},
        "Non-flat background settings through the linear pipeline.",
        "Internal σ₈ integration agreement and stored provenance of curvature settings.",
        "No independently reproduced curved reference table is bundled yet.",
    ),
    CanonicalValidationCase(
        "high-redshift",
        "High-redshift linear output",
        {"enable_ede": False, "z_values": [0.0, 2.0, 10.0], "single_z": 10.0},
        "Stored P(k,z) and σ(M,z) at z = 10.",
        "Exact-redshift σ₈ internal comparison, with HMF fit validity displayed separately.",
        "A supported linear solve does not validate every empirical HMF at high redshift.",
    ),
    CanonicalValidationCase(
        "low-mass-coverage",
        "Low-mass numerical coverage",
        {
            "enable_ede": False,
            "mass_min_exp": 8,
            "selected_mass_exp": 8,
            "k_max": 300.0,
        },
        "Sensitivity of σ(M) near the low-mass edge to sampled high-k coverage.",
        "Endpoint-removal diagnostics and a deliberate one-setting convergence follow-up.",
        "Finite-range sensitivity is not full numerical or physical convergence.",
    ),
    CanonicalValidationCase(
        "high-mass-tail",
        "High-mass rare tail",
        {"enable_ede": False, "mass_max_exp": 16, "selected_mass_exp": 14},
        "Large-radius σ(M) and the rare-halo tail within the sampled mass range.",
        "Internal σ₈ check plus HMF calibration mask and finite upper-bound cumulative semantics.",
        "The tail is not observational validation or a universal fit calibration.",
    ),
    CanonicalValidationCase(
        "tinker-redshift-boundary",
        "Intentional empirical-fit boundary",
        {
            "enable_ede": False,
            "fitting": "Tinker 2008",
            "mass_definition": "so_mean",
            "delta_halo": 200.0,
            "z_values": [0.0, 3.0],
            "single_z": 3.0,
        },
        "Visible behavior when a requested HMF redshift exceeds Tinker 2008's published z ≤ 2.5 range.",
        "A dotted/outside-calibration validity state; no hidden pass or silent calibration upgrade.",
        "This is a boundary-signaling test, not a scientifically supported Tinker prediction at z = 3.",
    ),
)


def canonical_case(identifier: str) -> CanonicalValidationCase:
    for case in CANONICAL_VALIDATION_CASES:
        if case.identifier == identifier:
            return case
    raise KeyError(f"Unknown canonical validation case: {identifier}")


def assess_canonical_case(params: dict, case: CanonicalValidationCase) -> dict:
    """Report whether a run configuration covers a case without inventing an expected output."""
    mismatches = []
    for key, expected in case.parameter_overrides.items():
        actual = params.get(key)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and np.isclose(
                float(actual), expected, rtol=1e-12, atol=0.0
            )
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(
                {"parameter": key, "expected": expected, "actual": actual}
            )
    return {
        "case_id": case.identifier,
        "title": case.title,
        "configured": not mismatches,
        "mismatches": mismatches,
        "target": case.target,
        "expected_evidence": case.expected_evidence,
        "boundary": case.boundary,
        "reference_status": (
            "Fixed ΛCDM CAMB reference tested in CI; not evaluated for this run"
            if case.identifier in {"lcdm-planck-like", "high-redshift"}
            else "No external reference evaluated for this case"
        ),
        "suite_version": CANONICAL_CASES_VERSION,
    }


def canonical_case_rows(params: dict) -> list[dict]:
    """Return the whole transparent validation matrix for a run configuration."""
    return [assess_canonical_case(params, case) for case in CANONICAL_VALIDATION_CASES]


def _adaptive_sigma8_reference(
    k: np.ndarray, power: np.ndarray, h: float, *, epsrel: float = 1e-7
) -> tuple[float, str | None]:
    """Independently integrate sampled P(k) in log-k using adaptive quadrature."""
    k = np.asarray(k, dtype=float)
    power = np.asarray(power, dtype=float)
    interpolate = build_power_interpolator(k, power)
    radius = 8.0 / float(h)

    def integrand(log_k: float) -> float:
        kval = float(np.exp(log_k))
        return float(
            kval**3
            * interpolate(kval)
            * window_squared(kval * radius, "Top-hat")
            / (2.0 * np.pi**2)
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        integral, _error = quad(
            integrand,
            float(np.log(k[0])),
            float(np.log(k[-1])),
            epsrel=epsrel,
            limit=300,
        )
    if not np.isfinite(integral) or integral <= 0:
        raise FloatingPointError(
            "Adaptive sigma8 benchmark returned an invalid integral"
        )
    warning = str(caught[0].message) if caught else None
    return float(np.sqrt(integral)), warning


def adaptive_sigma8(
    k: np.ndarray, power: np.ndarray, h: float, *, epsrel: float = 1e-7
) -> float:
    """Adaptive value only; benchmark callers should use its diagnostic sibling."""
    return _adaptive_sigma8_reference(k, power, h, epsrel=epsrel)[0]


def internal_sigma8_benchmark(
    run: dict, *, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
) -> dict:
    """Compare saved fixed-grid σ8 with adaptive quadrature on the same P(k).

    This verifies the integration implementation only. It intentionally cannot
    establish agreement with an external Boltzmann code or physical validity.
    """
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("Benchmark relative tolerance must be finite and positive")
    if "power_result" not in run or "sigma_result" not in run:
        run = pipeline_from_saved_run(run)
    power = run["power_result"]
    sigma = run["sigma_result"]
    k = np.asarray(power["k"], dtype=float)
    p_by_z = np.asarray(power.get("P_by_z", [power["P"]]), dtype=float)
    redshifts = np.asarray(
        power.get("redshifts", sigma.get("redshifts", [0.0])), dtype=float
    )
    stored = np.asarray(sigma.get("sigma8_pipeline_by_z"), dtype=float)
    h = float(power.get("derived", {}).get("h", float(run["params"]["H0"]) / 100.0))
    if (
        p_by_z.ndim != 2
        or redshifts.ndim != 1
        or redshifts.size == 0
        or p_by_z.shape[0] != redshifts.size
        or stored.shape != redshifts.shape
    ):
        raise ValueError(
            "The saved P(k,z) and sigma8 arrays are incompatible for benchmarking"
        )
    if (
        not np.isfinite(stored).all()
        or np.any(stored <= 0)
        or not np.isfinite(redshifts).all()
        or np.any(redshifts < 0)
        or np.any(np.diff(redshifts) <= 0)
        or not np.isfinite(h)
        or h <= 0
    ):
        raise ValueError(
            "Benchmark requires positive finite sigma8 and h, and ordered nonnegative redshifts"
        )
    started = perf_counter()
    rows = []
    for index, redshift in enumerate(redshifts):
        reference, integration_warning = _adaptive_sigma8_reference(k, p_by_z[index], h)
        discrepancy = abs(float(stored[index]) / reference - 1.0)
        rows.append(
            {
                "observable": "sigma8",
                "redshift": float(redshift),
                "stored_fixed_grid": float(stored[index]),
                "adaptive_reference": reference,
                "fractional_discrepancy": discrepancy,
                "status": "fail"
                if discrepancy > relative_tolerance
                else ("review" if integration_warning else "pass"),
                "reference_warning": integration_warning or "",
            }
        )
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "relative_tolerance": float(relative_tolerance),
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
        "scope_limit": "This is an internal numerical cross-check: adaptive quadrature versus HaloForge's fixed sampled-grid integration over the same P(k). It is not an independent CLASS, CAMB, Colossus, hmf, simulation, or physical-model benchmark.",
    }
