# ADR 0002: Halo mass definitions and HMF calibration contracts fail closed

**Status:** Accepted  
**Date:** 2026-09-14  
**Decision area:** Halo mass functions, calibration scope, and scientific validity

## Context

Halo-mass-function formulae are not interchangeable curves. Published fits may assume an analytic top-hat smoothing mass, a friends-of-friends catalogue with a stated linking length, or a spherical-overdensity halo relative to mean matter density. Their stated redshift, overdensity, and ln(sigma^-1) domains are also part of the model definition.

A smooth output outside those contracts can look persuasive while failing to support the intended scientific claim. Automatically converting or relabelling mass conventions would obscure that mismatch.

## Decision

1. Maintain one explicit fit-contract registry that records the required halo-mass definition, published ranges, family, and citation for every supported HMF fit.
2. Reject incompatible mass definitions and unsupported overdensity settings before fit evaluation; do not silently convert between FOF, SO, and analytic top-hat masses.
3. Return a pointwise calibration mask for compatible configurations. Plot and export in-domain and outside-domain values distinctly, while preserving numerical values for inspection.
4. Keep Press-Schechter and Sheth-Tormen analytic references explicitly separate from empirical simulation calibration claims.
5. Record contract evidence inside the scientific-validity artifact, but do not allow it alone to establish cosmology support or publication suitability.

## Consequences

- Users receive an actionable error for incompatible configurations instead of an apparently quantitative but undefined conversion.
- Extrapolated values remain visible for numerical exploration but are visibly dotted/review-state data, not hidden agreement.
- EDE and other cosmology-family support remain unresolved unless independent calibration evidence exists, even where a point falls inside a nominal fit range.
- Supporting more fits requires contract research, tests, citations, and explicit convention decisions rather than only adding a formula.

## Evidence and implementation

- Contract registry and validation: `engine/contracts.py`
- HMF evaluation: `engine/hmf.py` and `engine/fitting_functions.py`
- Separate validity claims: `engine/assurance.py`, `engine/validity.py`, and `engine/uncertainty.py`
- Scientific reference notes: `docs/SCIENCE-REFERENCES.md`
- Contract and calibration tests: `tests/test_tinker_overdensity.py` and `tests/test_hmf.py`

## Revisit when

- Adding a new mass-definition conversion, empirical fit, cosmology family, or nonlinear emulator.
- Obtaining independent validation for an EDE or nonstandard-cosmology HMF calibration.
- Changing calibration-domain semantics, pointwise mask behavior, or validity-artifact schema.
