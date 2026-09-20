# Independent CAMB reference

`camb_lcdm.json` contains linear total-matter power, top-hat σ(M), and σ₈
computed with CAMB 1.6.6. The generator imports no HaloForge calculations.
The JSON records the cosmology, full CAMB settings, versions, units, generation
time, and tolerances.

## Coverage

- Flat ΛCDM with massless neutrinos and no early dark energy.
- Redshifts 0, 2, and 10.
- P(k): 81 logarithmically spaced samples from 10⁻⁴ to 10 Mpc⁻¹, in Mpc³.
- σ(M): 11 masses from 10¹⁰ to 10¹⁵ h⁻¹ M☉, using physical radii in Mpc.
- σ₈: a top-hat radius of 8 h⁻¹ Mpc at each redshift.

The real-solver test compares AxiCLASS output against every tabulated point.
Relative tolerances were declared before the comparison: 1% for P(k), 0.3%
for σ(M) and σ₈. These accommodate differences between independent solvers,
recombination/BBN defaults, and finite-range integration. They are regression
thresholds, not error estimates or guarantees across the parameter space.

This reference does not validate EDE, curvature, massive neutrinos, nonlinear
power, alternative smoothing windows, or empirical halo-mass-function calibration.
The in-app adaptive σ₈ benchmark is a separate check on the same sampled spectrum;
it does not evaluate this CAMB reference for an arbitrary saved run.

## Reproduce

Use a separate Python environment to keep CAMB reference dependencies apart from
the production AxiCLASS environment:

```sh
python3.11 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference.txt
.venv-reference/bin/python scripts/generate_camb_reference.py
```

Review all numerical changes before accepting a regenerated fixture. Do not
increase tolerances to accommodate a failing comparison without a scientific
explanation. In an environment with the pinned AxiCLASS binding, run:

```sh
HALOFORGE_TEST_CLASS=1 python -m pytest -q tests/test_class_integration.py
```

CAMB documents the [power interpolation units](https://camb.readthedocs.io/en/stable/results.html#camb.results.CAMBdata.get_matter_power_interpolator)
and [top-hat variance calculation](https://camb.readthedocs.io/en/stable/results.html#camb.results.CAMBdata.get_sigmaR).
