"""Deterministic parameter-campaign planning with space-filling designs and resource bounds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from typing import Any

import numpy as np

from config.defaults import DEFAULT_PARAMS
from engine.parameter_validation import solver_parameter_errors


@dataclass(frozen=True)
class CampaignResourceEstimate:
    """Estimated resource requirements for a parameter campaign."""

    total_runs: int
    estimated_cpu_seconds: float
    estimated_wall_seconds: float
    recommended_workers: int
    estimated_memory_mb: float
    estimated_disk_mb: float
    cartesian_warning: str | None = None


def estimate_campaign_resources(
    total_runs: int,
    *,
    worker_count: int | None = None,
    seconds_per_run: float = 1.25,
) -> CampaignResourceEstimate:
    """Estimate CPU, memory, disk, and wall time with core-oversubscription guard."""
    cpu_cores = os.cpu_count() or 4
    recommended_workers = max(1, cpu_cores - 1)
    workers = worker_count or recommended_workers
    workers = max(1, min(workers, cpu_cores))

    total_cpu_seconds = total_runs * seconds_per_run
    wall_seconds = total_cpu_seconds / workers

    # Approx 25MB RAM per active solver worker, ~0.8MB disk per saved run archive
    est_memory_mb = 120.0 + workers * 35.0
    est_disk_mb = total_runs * 0.85

    warning = None
    if total_runs > 64:
        warning = (
            f"Campaign contains {total_runs} runs. A Cartesian design may produce "
            "combinatorial runtime growth. Consider Latin Hypercube or Sobol sampling."
        )

    return CampaignResourceEstimate(
        total_runs=total_runs,
        estimated_cpu_seconds=total_cpu_seconds,
        estimated_wall_seconds=wall_seconds,
        recommended_workers=recommended_workers,
        estimated_memory_mb=est_memory_mb,
        estimated_disk_mb=est_disk_mb,
        cartesian_warning=warning,
    )


def validate_campaign_grid(combinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify physical constraints and reject duplicate or unphysical configurations."""
    if not combinations:
        raise ValueError("Campaign has zero parameter combinations")

    valid_runs = []
    seen = set()
    for idx, combo in enumerate(combinations):
        candidate = {**DEFAULT_PARAMS, **combo}
        issues = solver_parameter_errors(candidate)
        if issues:
            raise ValueError(
                f"Configuration #{idx + 1} violates physical constraints: {'; '.join(issues)}"
            )
        # Deduplication key
        key = tuple(
            sorted(
                (k, float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in combo.items()
            )
        )
        if key in seen:
            raise ValueError(
                f"Duplicate parameter combination found in campaign: {combo}"
            )
        seen.add(key)
        valid_runs.append(combo)

    return valid_runs


def cartesian_campaign(values: dict[str, list], *, max_runs: int = 256) -> list[dict]:
    """Expand explicit parameter values without silently accepting duplicates."""
    if not values or not isinstance(max_runs, int) or max_runs < 1:
        raise ValueError("Campaign needs parameters and a positive run limit")
    names = list(values)
    choices = []
    for name in names:
        options = values[name]
        if not isinstance(name, str) or not isinstance(options, list) or not options:
            raise ValueError(
                "Each campaign parameter needs a nonempty explicit value list"
            )
        if len({repr(value) for value in options}) != len(options):
            raise ValueError(f"Campaign parameter {name!r} contains duplicate values")
        choices.append(options)
    count = 1
    for options in choices:
        count *= len(options)
    if count > max_runs:
        raise ValueError(
            f"Campaign expands to {count} runs, above the {max_runs} run limit"
        )
    raw = [
        dict(zip(names, combination, strict=True)) for combination in product(*choices)
    ]
    return validate_campaign_grid(raw)


def latin_hypercube_campaign(
    param_bounds: dict[str, tuple[float, float]],
    num_samples: int,
    *,
    seed: int = 42,
    log_scale_params: set[str] | None = None,
) -> list[dict[str, float]]:
    """Generate space-filling Latin Hypercube Sampling (LHS) design across parameters."""
    if not param_bounds or num_samples < 2:
        raise ValueError(
            "LHS campaign requires bounded parameters and at least 2 samples"
        )

    rng = np.random.default_rng(seed)
    log_params = log_scale_params or {"A_s", "k_max"}
    names = list(param_bounds)
    dim = len(names)

    # Standard LHS stratification in [0, 1]
    result_matrix = np.zeros((num_samples, dim))
    for j in range(dim):
        perm = rng.permutation(num_samples)
        intervals = (perm + rng.uniform(0.05, 0.95, size=num_samples)) / num_samples
        result_matrix[:, j] = intervals

    combinations = []
    for i in range(num_samples):
        run = {}
        for j, name in enumerate(names):
            lo, hi = param_bounds[name]
            u = float(result_matrix[i, j])
            if name in log_params and lo > 0 and hi > 0:
                val = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                val = float(lo + u * (hi - lo))
            run[name] = val
        combinations.append(run)

    return validate_campaign_grid(combinations)


def sobol_campaign(
    param_bounds: dict[str, tuple[float, float]],
    num_samples: int,
    *,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate low-discrepancy quasi-random Sobol sequence design."""
    from scipy.stats import qmc

    if not param_bounds or num_samples < 2:
        raise ValueError("Sobol campaign requires bounded parameters and >= 2 samples")

    names = list(param_bounds)
    dim = len(names)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    # Sobol works best with powers of 2
    m = int(np.ceil(np.log2(num_samples)))
    sample_points = sampler.random_base2(m=m)[:num_samples]

    lower = np.array([param_bounds[name][0] for name in names])
    upper = np.array([param_bounds[name][1] for name in names])
    scaled = qmc.scale(sample_points, lower, upper)

    combinations = []
    for row in scaled:
        combinations.append(
            {name: float(val) for name, val in zip(names, row, strict=True)}
        )

    return validate_campaign_grid(combinations)
