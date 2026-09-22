# Scientific implementation references

## Initial-condition unit and velocity conventions

CLASS matter spectra enter the IC generator as k in Mpc⁻¹ and P in Mpc³. The periodic box uses coordinates in Mpc/h, so generation converts to k_box=k_CLASS/h and P_box=h³P_CLASS before checking Fourier coverage or drawing modes. The shell check reports power in the same box units. A fixed-amplitude test at h=0.67 converts the measured shell power back to physical units and compares it against the input spectrum.

The generator returns physical peculiar velocities in km/s. The GADGET-4 HDF5 writer stores u=v_pec/√a, consistent with the [official snapshot-format convention](https://wwwmpa.mpa-garching.mpg.de/gadget4/06_snapshotformat/). Nonzero velocity tests cover a=1, 1/4, and 1/50. These are unit and file-format checks; comparison with an established IC generator and validated GADGET-4 execution remain open.

## Tinker 2008 coefficient correction

Reference: Tinker et al., 2008, ApJ 688, 709–728, https://arxiv.org/abs/0803.2706.
Independent implementation inspected: https://github.com/halomod/hmf/blob/master/src/hmf/mass_function/fitting_functions.py (Tinker08).

The previous implementation held A, a, b, c at the Δ=200 values while changing only the redshift exponent with Δ. At z=0 this made every overdensity produce the same curve. All four coefficients now depend on spherical overdensity relative to the mean matter density, using the nine tabulated overdensities and coefficient precision recorded by hmf. Intermediate overdensities use a not-a-knot cubic spline in Δ, matching the interpolation convention of hmf's interpolating spline. Values outside 200–3200 are rejected.

This fixes the coefficient evaluation only. It does not establish EDE calibration, validate every selected redshift or sigma, convert M200c into M200m, or certify publication suitability. Those requirements remain open. Empirical error is not the floating-point interpolation tolerance.

### Fit-domain conventions

Tinker et al. (2008) express their sampled domain as \(\log_{10}(\sigma^{-1})\), not a natural logarithm: the implemented interval is −0.6 to 0.4 at z=0 and −0.2 to 0.4 for z>0. Watson SO (2013) uses \(\ln(\sigma^{-1})\), with the implemented published interval −0.55 to 1.05. These masks only expose the reported fit-domain boundary; they do not establish cosmology or halo-definition calibration.

### Watson SO endpoint coefficient correction

Watson et al. (2013), equation 17 and its coefficient table, define the multiplicity prefactor as \(A[(\beta/\sigma)^\alpha+1]\exp(-\gamma/\sigma^2)\). The z=0 and z≥6 endpoint branches now preserve that alpha/beta mapping. Fixed regression values at Δ=178 protect both endpoints. This correction does not validate Watson SO for EDE cosmologies.

## EDE background source

For every completed CLASS/AxiCLASS solve, HaloForge reads the solver's background table and derives Ωm(z)=Ωm,0(1+z)^3/[H(z)/H(0)]² on the requested redshift grid. Watson SO receives that value. When an EDE solve lacks a usable CLASS background table, Watson SO is unavailable rather than using a ΛCDM closure approximation. Other outputs remain available and record the background warning. This establishes internal provenance for the background value; it does not validate the Watson SO calibration for EDE cosmologies.

## Numerical output semantics

Nonpositive or nonfinite sigma is rejected instead of clipped. Differential HMF underflow may produce zero; no artificial positive floor is added. Cumulative abundance integrates to the last sampled mass and has an exact zero endpoint. Ratio comparisons do not extrapolate the baseline and do not divide by zero.

## Sharp-k quadrature and interpolation

The Fourier top-hat cuts the variance integral off at k=1/R. Applying a sampled Heaviside mask inside Simpson quadrature made the previous result piecewise constant in R between adjacent k samples. The new implementation integrates each log-linear P(k) segment analytically and includes the partial segment at the exact cutoff. Tests compare finite-range power laws, including P∝k⁻³, against their closed forms. Gaussian tests independently check the gamma-function variance integral and mass derivative. The cutoff convention and non-unique mass assignment are discussed by [Schneider, Smith & Reed (2013)](https://academic.oup.com/mnras/article/433/2/1573/1750290) and the [hmf SharpK implementation](https://hmf.readthedocs.io/en/latest/_autosummary/filters/hmf.density_field.filters.SharpK.html).

## Reed 2003 and 2007 corrections

Reed03 multiplies Sheth–Tormen by exp[−0.7/(σ cosh⁵(2σ))]. Previously the code used (σ cosh(2σ))⁵, incorrectly raising σ to the fifth power. The correction now uses a stable log-cosh evaluation and is checked at several σ values.

Reed07 equation (11) depends on n_eff. The HMF now supplies n_eff=−6 dlnσ/dlnM−3 from the calculated variance instead of assuming −2 everywhere. The multiplicity function requires this argument; standalone reference plots explicitly label their chosen n_eff. Coefficients use c=1.08 and ca=0.764, and the implemented ln(σ⁻¹) calibration interval is −0.5 to 1.2. Sources: [Reed et al. (2007)](https://academic.oup.com/mnras/article/374/1/2/959965) and the [independent hmf fitting-function implementation](https://hmf.readthedocs.io/en/latest/_modules/hmf/mass_function/fitting_functions.html). These corrections do not establish EDE calibration.

## Exact limits and comparison semantics

AxiCLASS rejects scalar-field shooting with zero EDE fraction. HaloForge now uses the identical settings as ΛCDM at this exact boundary; a real-solver regression compares the complete spectra. Closed cosmologies may have Ωm(z)>1, so that condition no longer discards a valid CLASS background. Real-solver tests cover both cases.

Positive brackets use log-log interpolation even if a distant rare-tail sample underflows to zero. Brackets containing zero use linear interpolation, and a zero baseline remains an undefined ratio. Residual plot ranges include every finite sample instead of clipping the most extreme percentile. Cumulative fit masks include the calibration state of the complete integrated tail.

### Linear field normalization

The periodic Gaussian field uses NumPy's unnormalized forward FFT and inverse
FFT divided by N³. Unit-variance real-space noise therefore has expected mode
power N³. Multiplying its modes by √[P(k)N³/L³] gives the physical convention
⟨|δₖ|²⟩ = V P(k), after accounting for the discrete Fourier volume factors.
For constant P, tests recover the mean-subtracted noise multiplied by
√(P/V_cell), for both odd and even grids. Multiplying P by four doubles the
field amplitude. Removing the DC mode reduces the ensemble cell variance by
the finite-grid factor (1 − 1/N³).

Field maps use cell-center coordinates and transpose the stored [x,y] slice to
Plotly's [y,x] convention. All panels share the baseline slice RMS
√⟨δ²⟩ (not the slice standard deviation); a single slice need not have zero mean,
even though the full periodic volume does. Color saturation is stated next to
the maps, and both axes retain the simulated box extent.

### Saved-run sensitivity

For an observable y and one changed parameter p, the sensitivity table reports
Δy/y₀ and (Δy/y₀)/Δp. When p₀ is nonzero it also reports Δp/p₀ and the finite
response ratio (Δy/y₀)/(Δp/p₀). These are finite differences between completed
runs, not estimates of a converged infinitesimal derivative. For p₀ = 0,
parameter percentages and their ratio are undefined and remain blank; the
response per absolute parameter unit is still available.

Mass sampling uses bounded log-log interpolation of positive saved σ(M), with
an exact saved redshift. Out-of-range masses are rejected rather than assigned
endpoint values. Invalid candidate arrays or parameter values appear as excluded
comparisons. The saved-counterfactual search ranks actual qualifying experiments
by |Δp| and uses proximity to the requested response magnitude to break ties;
it never estimates a new parameter setting.

### Cumulative halo counts

For sampled differential abundances fᵢ = dn/dln M, positive adjacent samples
are connected by a power law. Its exact interval contribution is
Δln M × (fᵢ₊₁ − fᵢ)/(ln fᵢ₊₁ − ln fᵢ), with the constant-value limit fᵢ Δln M.
The implementation evaluates this logarithmic mean with `expm1` to avoid
cancellation and overflow. Intervals touching an underflow zero use a linear
interpolant in ln M; no positive floor or extrapolated tail is invented.
Reverse accumulation preserves the finite upper-mass bound and small tail sums.

Analytic tests cover constant and sloped power laws on sparse grids, zero tails,
and a 600-decade abundance contrast. For a 20-point grid spanning 10¹⁰–10¹⁵ and
f ∝ M⁻¹·⁵, the former trapezoidal rule overestimated the integral by 6.79%; the
power-law integration recovers the analytic finite-bound result to floating-point
precision. This improvement does not remove uncertainty in the sampled HMF or
validate extrapolation beyond the fit's calibration range.

Redshift lookup is shared by plots, variance diagnostics, HMF evaluation,
sensitivity, and inspection. A request must match exactly one saved redshift
within an absolute rounding tolerance of 10⁻⁹. Missing, unordered, non-finite,
or ambiguous redshift grids raise an error; no nearest-redshift substitution
or redshift interpolation is performed.


## Fit-range evidence and simulation calibration

Scientific validity schema v3 records `all_evaluated_points_pass_fit_checks`. It requires a nonempty, one-dimensional Boolean mask with every entry true. Empty, scalar, numeric, string, and malformed masks do not establish evidence. Passing the implemented range checks does not certify calibration for the current cosmology or halo definition.

Sheth–Tormen is classified as semi-empirical: its ellipsoidal-collapse argument is analytic, while the coefficients are informed by simulation comparisons. HaloForge evaluates it with a real-space top-hat smoothing mass, but does not assign that mass to a specific FOF or spherical-overdensity halo catalogue. Its calibration mask remains unverified until a primary-source audit establishes a matching halo definition, cosmology, and domain. See [Sheth, Mo & Tormen (2001)](https://arxiv.org/abs/astro-ph/9907024). The source comparison describes high-resolution N-body simulations; it does not license an arbitrary halo-definition conversion.


## Inspection selection

Mass inspection selects the nearest saved sample in log mass, within the stored mass interval, and reports both requested and selected masses. It does not interpolate variance or radius. Invalid or out-of-range requests are rejected. Power and variance locate the requested redshift in their own coordinate grids; an array index from one is never reused as the redshift identity of the other. The smoothing convention comes from the saved variance result. Fourier contribution fractions remain finite-grid trapezoidal estimates, with the actual included sample interval reported explicitly.


The central 10–90% contribution band uses a piecewise-linear density in ln(k), consistent with trapezoidal interval integration. Within each interval, its cumulative area is quadratic, so percentile locations are found by solving that quadratic rather than interpolating the cumulative endpoints linearly. Contributions are scaled by their maximum before computing fractions to prevent cumulative overflow. Negative, non-finite, mismatched, or identically zero contributions cannot produce a ranked diagnostic. These changes do not supply information about unresolved modes.
