"""Interpretable experiment plans, deliberately not a black-box inference tool."""

from __future__ import annotations


PLANS = {
    "Tilt and low-mass structure": {
        "question": "How does a modest primordial tilt change propagate into low-mass structure?",
        "candidate_parameters": {"enable_ede": False, "n_s": 0.99},
        "hold_fixed": [
            "A_s",
            "k_pivot",
            "background cosmology",
            "HMF fit",
            "numerical grid",
        ],
        "inspect": [
            "Primordial spectrum around k_p",
            "Matter P(k)",
            "sigma(M) at lower mass",
            "HMF calibration state",
        ],
        "prediction": "Relative high-k power should rise first; lower-mass sigma(M) may respond more than cluster-scale sigma(M).",
        "caveat": "An HMF response remains dependent on the selected fit and calibration domain.",
    },
    "Numerical coverage at low mass": {
        "question": "Does the low-mass conclusion survive a wider sampled k range?",
        "candidate_parameters": {"enable_ede": False, "k_max": 300.0},
        "hold_fixed": ["all physical cosmological parameters", "mass grid", "HMF fit"],
        "inspect": [
            "Selected-mass k contribution band",
            "high-k endpoint sensitivity",
            "sigma(M) finite response",
            "Benchmark Lab",
        ],
        "prediction": "This changes numerical coverage, not the underlying cosmology; the strongest effect should be at smaller smoothing scales.",
        "caveat": "A wider k range alone is not a complete convergence proof.",
    },
    "Early expansion versus baseline": {
        "question": "Can a temporary EDE pulse alter a late-time linear structure prediction?",
        "candidate_parameters": {"enable_ede": True, "f_EDE": 0.12, "log10_a_c": -3.5},
        "hold_fixed": [
            "primordial parameters",
            "late-time background inputs",
            "numerical grid",
            "mass definition",
        ],
        "inspect": [
            "AxiCLASS background change",
            "Matter P(k)",
            "sigma(M)",
            "HMF validity and cosmology-support states",
        ],
        "prediction": "EDE changes early evolution, then scale-dependent linear structure; any empirical HMF interpretation needs separate calibration scrutiny.",
        "caveat": "Fit-range checks do not prove simulation calibration support for EDE.",
    },
}


def design_experiment(goal: str, baseline_params: dict) -> dict:
    """Return a concrete candidate based on a supplied named scientific goal."""
    if goal not in PLANS:
        raise ValueError(f"Unknown experiment-design goal: {goal}")
    plan = PLANS[goal]
    candidate = {**baseline_params, **plan["candidate_parameters"]}
    changed = {
        key: {"baseline": baseline_params.get(key), "candidate": candidate.get(key)}
        for key in plan["candidate_parameters"]
        if baseline_params.get(key) != candidate.get(key)
    }
    return {
        "goal": goal,
        **plan,
        "parameter_diff": changed,
        "scope_limit": "This is a transparent one-change plan, not active learning, an emulator, a parameter posterior, or an automated scientific conclusion.",
    }
