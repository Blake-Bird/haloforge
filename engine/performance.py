"""Small, reproducible performance probes for the local scientific core.

These are deliberately separate from scientific benchmarks: timing evidence
does not validate a cosmology, and a fast result is not necessarily correct.
"""

from __future__ import annotations

import statistics
import tracemalloc
from time import perf_counter

from engine.hmf import hmf_z
from engine.sigma import compute_sigma_result


PERFORMANCE_BENCHMARK_VERSION = "haloforge-core-performance-v1"
CORE_REBUILD_BUDGET_SECONDS = 2.0
CORE_PEAK_ALLOCATION_BUDGET_MIB = 128.0


def _state(value: float, budget: float) -> str:
    return "pass" if value <= budget else "review"


def profile_core_pipeline(run: dict, *, repetitions: int = 3) -> dict:
    """Time sigma/HMF recomputation and Python allocation peak for one run.

    It intentionally does not claim browser load, external-solver, or process
    RSS limits: those require an end-to-end environment and are reported as
    unmeasured rather than guessed.
    """
    if not isinstance(repetitions, int) or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be an integer between 1 and 10")
    power, params = run["power_result"], run["params"]
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    timings: list[float] = []
    try:
        for _ in range(repetitions):
            started = perf_counter()
            sigma = compute_sigma_result(power, params)
            if sigma.get("window_type") == "Top-hat":
                try:
                    hmf_z(
                        {
                            "params": params,
                            "power_result": power,
                            "sigma_result": sigma,
                        },
                        float(params.get("single_z", 0.0)),
                        params["fitting"],
                    )
                except (ValueError, FloatingPointError):
                    # An ineligible fit remains a scientific validity issue,
                    # not a reason to lose the sigma-core timing measurement.
                    pass
            timings.append(perf_counter() - started)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    median = float(statistics.median(timings))
    peak_mib = float(peak) / (1024 * 1024)
    return {
        "benchmark_version": PERFORMANCE_BENCHMARK_VERSION,
        "repetitions": repetitions,
        "core_rebuild_median_seconds": median,
        "core_rebuild_budget_seconds": CORE_REBUILD_BUDGET_SECONDS,
        "core_rebuild_state": _state(median, CORE_REBUILD_BUDGET_SECONDS),
        "python_peak_allocation_mib": peak_mib,
        "python_peak_allocation_budget_mib": CORE_PEAK_ALLOCATION_BUDGET_MIB,
        "python_peak_allocation_state": _state(
            peak_mib, CORE_PEAK_ALLOCATION_BUDGET_MIB
        ),
        "unmeasured_budgets": [
            "Browser initial load and interaction latency require browser instrumentation.",
            "External AxiCLASS startup and process RSS require an uncached real-solver run.",
            "Export time requires a representative filesystem and bundle-size scenario.",
        ],
        "scope_limit": "This is a local Python-core timing/allocation probe, not an end-to-end UI, solver, export, or hardware performance certification.",
    }
