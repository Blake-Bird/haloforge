# Scientific implementation references

## Tinker 2008 coefficient correction

Reference: Tinker et al., 2008, ApJ 688, 709–728, https://arxiv.org/abs/0803.2706.
Independent implementation inspected: https://github.com/halomod/hmf/blob/master/src/hmf/mass_function/fitting_functions.py (Tinker08).

The previous implementation held A, a, b, c at the Δ=200 values while changing only the redshift exponent with Δ. At z=0 this made every overdensity produce the same curve. All four coefficients now depend on spherical overdensity relative to the mean matter density, using the nine tabulated overdensities and coefficient precision recorded by hmf. Intermediate overdensities use a not-a-knot cubic spline in Δ, matching the interpolation convention of hmf's interpolating spline. Values outside 200–3200 are rejected.

This fixes the coefficient evaluation only. It does not establish EDE calibration, validate every selected redshift or sigma, convert M200c into M200m, or certify publication suitability. Those requirements remain open. Empirical error is not the floating-point interpolation tolerance.

## EDE background source

For every completed CLASS/AxiCLASS solve, HaloForge reads the solver's background table and derives Ωm(z)=Ωm,0(1+z)^3/[H(z)/H(0)]² on the requested redshift grid. Watson SO receives that value. When an EDE solve lacks a usable CLASS background table, Watson SO is unavailable rather than using a ΛCDM closure approximation. Other outputs remain available and record the background warning. This establishes internal provenance for the background value; it does not validate the Watson SO calibration for EDE cosmologies.

## Numerical output semantics

Nonpositive or nonfinite sigma is rejected instead of clipped. Differential HMF underflow may produce zero; no artificial positive floor is added. Cumulative abundance integrates to the last sampled mass and has an exact zero endpoint. Ratio comparisons do not extrapolate the baseline and do not divide by zero.
