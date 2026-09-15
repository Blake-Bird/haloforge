# Contributing

HaloForge changes must preserve scientific traceability, user-data privacy, and reproducibility.

Before proposing a change:

1. Run `ruff format --check app.py config content engine state tests`, `ruff check app.py config content engine state tests`, and `pytest -q`.
2. Add a regression test for a corrected numerical, storage, or interface contract.
3. State units, source equations, calibration domain, and intended failure behavior for scientific changes.
4. Do not commit `data/`, exports, caches, credentials, or participant data.
5. Do not label a result "validated", "publication-ready", or "supported" without the corresponding independent evidence.

For any new empirical fit, include its mass definition, reference density, halo finder, calibration cosmology, redshift and scale range, citation, and an explicit invalid/extrapolated UI state.
