"""Published halo multiplicity functions f(sigma).

The HMF convention is dn/dlnM=(rho_m/M) f(sigma) |dlnsigma/dlnM|.
Empirical fits are only used inside their published calibration domains when
the UI requests a validity mask; extrapolation remains available but labelled.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from engine.contracts import FIT_ALIASES, FIT_CONTRACTS


# Tinker et al. (2008), SO overdensity relative to mean matter density.
# Coefficient precision follows the public hmf implementation. See
# docs/SCIENCE-REFERENCES.md for source and interpolation convention.
TINKER_OVERDENSITIES = np.array(
    [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200], dtype=float
)
TINKER_COEFFICIENTS = np.array(
    [
        [0.1858659, 1.466904, 2.571104, 1.193958],
        [0.1995973, 1.521782, 2.254217, 1.270316],
        [0.2115659, 1.559186, 2.048674, 1.335191],
        [0.2184113, 1.614585, 1.869559, 1.446266],
        [0.2480968, 1.869936, 1.588649, 1.581345],
        [0.2546053, 2.128056, 1.507134, 1.795050],
        [0.2600000, 2.301275, 1.464374, 1.965613],
        [0.2600000, 2.529241, 1.436827, 2.237466],
        [0.2600000, 2.661983, 1.405210, 2.439729],
    ]
)
_TINKER_INTERPOLATOR = CubicSpline(
    TINKER_OVERDENSITIES, TINKER_COEFFICIENTS, extrapolate=False
)


FITTING_NAMES = [
    "Press-Schechter 1974",
    "Sheth-Tormen 2001",
    "Jenkins 2001",
    "Reed 2003",
    "Warren 2006",
    "Reed 2007",
    "Tinker 2008",
    "Crocce 2010",
    "Courtin 2010",
    "Bhattacharya 2011",
    "Angulo 2012",
    "Watson FOF 2013",
    "Watson SO 2013",
]

FIT_METADATA = {
    name: (
        "analytic"
        if c.family == "analytic"
        else "simulation coefficients; domain unverified"
        if c.family == "semi-empirical"
        else (
            str(c.log_inv_sigma_range)
            if c.log_inv_sigma_range
            else "see published mass range"
        ),
        "all"
        if c.redshift_range is None
        else f"{c.redshift_range[0]:g}–{c.redshift_range[1]:g}",
        f"{c.citation}; {c.mass_definition}",
    )
    for name, c in FIT_CONTRACTS.items()
}


def _s(sigma):
    values = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    return values


def _positive_scalar(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _exp(value):
    # Cutoff underflow is physical zero; infinite intermediate cutoffs are safe
    # in log space, where no polynomial prefactor multiplies zero by infinity.
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(value)


def _log_nu(sigma, delta_c):
    return np.log(_positive_scalar(delta_c, "Collapse threshold")) - np.log(_s(sigma))


def _log_cutoff_form(sigma, A, a, b, c):
    """ln{A[(b/σ)^a+1] exp(−c/σ²)} without overflowing prefactors."""
    logs = np.log(_s(sigma))
    return (
        np.log(A) + np.logaddexp(a * (np.log(b) - logs), 0) - _exp(np.log(c) - 2 * logs)
    )


def _redshift_factor(z):
    z = float(z)
    if not np.isfinite(z) or z < 0:
        raise ValueError("Redshift must be finite and nonnegative")
    return 1 + z


def nu_from_sigma(sigma, delta_c=1.686):
    return _positive_scalar(delta_c, "Collapse threshold") / _s(sigma)


def f_press_schechter(sigma, delta_c=1.686, **_):
    lognu = _log_nu(sigma, delta_c)
    return _exp(0.5 * np.log(2 / np.pi) + lognu - 0.5 * _exp(2 * lognu))


def f_sheth_tormen(sigma, delta_c=1.686, A=0.3222, a=0.707, p=0.3, **_):
    A = _positive_scalar(A, "Normalization")
    a = _positive_scalar(a, "Coefficient a")
    if not np.isfinite(p):
        raise ValueError("Coefficient p must be finite")
    lognu = _log_nu(sigma, delta_c)
    log_a_nu2 = np.log(a) + 2 * lognu
    return _exp(
        np.log(A)
        + 0.5 * np.log(2 * a / np.pi)
        + np.logaddexp(0, -p * log_a_nu2)
        + lognu
        - 0.5 * _exp(log_a_nu2)
    )


def f_jenkins(sigma, **_):
    return 0.315 * _exp(-(np.abs(-np.log(_s(sigma)) + 0.61) ** 3.8))


def f_reed03(sigma, delta_c=1.686, **_):
    s = _s(sigma)
    # Only cosh(2σ), not σ, is raised to the fifth power in Reed03.
    with np.errstate(over="ignore"):
        log_cosh = np.logaddexp(2 * s, -2 * s) - np.log(2)
        correction = _exp(-0.7 * _exp(-np.log(s) - 5 * log_cosh))
    return f_sheth_tormen(s, delta_c) * correction


def f_warren(sigma, **_):
    logs = np.log(_s(sigma))
    return _exp(
        np.log(0.7234)
        + np.logaddexp(-1.625 * logs, np.log(0.2538))
        - _exp(np.log(1.1982) - 2 * logs)
    )


def f_reed07(sigma, delta_c=1.686, neff=None, **_):
    s = _s(sigma)
    if neff is None:
        raise ValueError("Reed 2007 requires the effective spectral slope n_eff")
    neff = np.asarray(neff, dtype=float)
    if np.any(~np.isfinite(neff)) or np.any(neff <= -3):
        raise ValueError("Reed 2007 requires finite n_eff > -3")
    if neff.ndim and neff.shape != s.shape:
        raise ValueError("n_eff must be scalar or match the sigma grid")
    lognu = _log_nu(s, delta_c)
    a, p, ca, A = 0.764 / 1.08, 0.3, 1.08, 0.3222
    lninv = -np.log(s)
    g1 = _exp(-((lninv - 0.4) ** 2) / 0.72)
    g2 = _exp(-((lninv - 0.75) ** 2) / 0.08)
    log_bracket = np.logaddexp(
        np.log1p(0.6 * g1 + 0.4 * g2), -p * (np.log(a) + 2 * lognu)
    )
    exponent = -0.5 * _exp(np.log(ca * a) + 2 * lognu) - _exp(
        np.log(0.03) - 2 * np.log(neff + 3) + 0.6 * lognu
    )
    return _exp(
        np.log(A) + 0.5 * np.log(2 * a / np.pi) + log_bracket + lognu + exponent
    )


def f_tinker08(sigma, z=0.0, delta_halo=200.0, **_):
    """Tinker08 multiplicity at bounded overdensity relative to mean density.

    Evaluation outside the redshift calibration remains exploratory.
    """
    s, zp1, delta = _s(sigma), _redshift_factor(z), float(delta_halo)
    if not np.isfinite(delta) or not 200 <= delta <= 3200:
        raise ValueError("Tinker 2008 requires 200 ≤ Δmean ≤ 3200")
    A0, a0, b0, c = _TINKER_INTERPOLATOR(delta)
    alpha = 10 ** (-((0.75 / np.log10(delta / 75)) ** 1.2))
    A, a, b = A0 * zp1**-0.14, a0 * zp1**-0.06, b0 * zp1**-alpha
    return _exp(_log_cutoff_form(s, A, a, b, c))


def f_crocce10(sigma, z=0.0, **_):
    logs, zp1 = np.log(_s(sigma)), _redshift_factor(z)
    A, a, b, c = (
        0.58 * zp1**-0.13,
        1.37 * zp1**-0.15,
        0.30 * zp1**-0.084,
        1.036 * zp1**-0.024,
    )
    return _exp(
        np.log(A) + np.logaddexp(-a * logs, np.log(b)) - _exp(np.log(c) - 2 * logs)
    )


def f_courtin10(sigma, delta_c=1.686, **_):
    return f_sheth_tormen(sigma, delta_c, A=0.348, a=0.695, p=0.1)


def f_bhattacharya11(sigma, z=0.0, delta_c=1.686, **_):
    lognu, zp1 = _log_nu(sigma, delta_c), _redshift_factor(z)
    A, a, p, q = 0.333 * zp1**-0.11, 0.788 * zp1**-0.01, 0.807, 1.795
    log_a_nu2 = np.log(a) + 2 * lognu
    return _exp(
        np.log(A)
        + 0.5 * np.log(2 / np.pi)
        - 0.5 * _exp(log_a_nu2)
        + np.logaddexp(0, -p * log_a_nu2)
        + 0.5 * q * log_a_nu2
    )


def f_angulo12(sigma, subhalos=False, **_):
    A, a, b, c = (0.265, 1.9, 1.675, 1.4) if subhalos else (0.201, 1.7, 2.08, 1.172)
    return _exp(_log_cutoff_form(sigma, A, a, b, c))


def f_watson_fof13(sigma, **_):
    return _exp(_log_cutoff_form(sigma, 0.282, 1.406, 2.163, 1.210))


def f_watson_so13(sigma, z=0.0, omega_m_z=0.3, delta_halo=200.0, **_):
    s, zp1 = _s(sigma), _redshift_factor(z)
    om = _positive_scalar(omega_m_z, "Matter fraction")
    delta = _positive_scalar(delta_halo, "Halo overdensity")
    if z == 0:
        # Watson et al. (2013), eq. 17: alpha is the exponent and beta
        # is the scale in [(beta/sigma)^alpha + 1].
        A, a, b, c = 0.194, 1.805, 2.267, 1.287
    elif z >= 6:
        A, a, b, c = 0.563, 3.810, 0.874, 1.453
    else:
        A = om * (1.907 * zp1**-3.216 + 0.074)
        a = om * (3.136 * zp1**-3.058 + 2.349)
        b = om * (5.907 * zp1**-3.599 + 2.344)
        c = 1.318
    log_base = _log_cutoff_form(s, A, a, b, c)
    log_C = 0.023 * (delta / 178 - 1)
    d = -0.456 * om - 0.139
    correction = 0.072 * (1 - delta / 178)
    # Δ<178 diverges at sufficiently small σ outside the empirical domain.
    # Reject such extrapolations instead of returning NaN or a fabricated zero.
    with np.errstate(invalid="ignore"):
        tail = (
            np.sign(correction) * _exp(np.log(abs(correction)) - 2.130 * np.log(s))
            if correction
            else np.zeros_like(s)
        )
        result = _exp(log_base + log_C + d * np.log(delta / 178) + tail)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Watson SO extrapolation exceeds floating-point range")
    return result


_DISPATCH = {
    "Press-Schechter 1974": f_press_schechter,
    "Sheth-Tormen 2001": f_sheth_tormen,
    "Jenkins 2001": f_jenkins,
    "Reed 2003": f_reed03,
    "Warren 2006": f_warren,
    "Reed 2007": f_reed07,
    "Tinker 2008": f_tinker08,
    "Crocce 2010": f_crocce10,
    "Courtin 2010": f_courtin10,
    "Bhattacharya 2011": f_bhattacharya11,
    "Angulo 2012": f_angulo12,
    "Watson FOF 2013": f_watson_fof13,
    "Watson SO 2013": f_watson_so13,
}


def canonical_name(name: str) -> str:
    return FIT_ALIASES.get(name, name)


def fitting_values(sigma, delta_c, fitting, z=0.0, **kwargs):
    name = canonical_name(fitting)
    if name not in _DISPATCH:
        raise ValueError(f"Unknown fitting function: {fitting}")
    return _DISPATCH[name](sigma, delta_c=delta_c, z=z, **kwargs)
