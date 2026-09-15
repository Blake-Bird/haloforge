"""Published halo multiplicity functions f(sigma).

The HMF convention is dn/dlnM=(rho_m/M) f(sigma) |dlnsigma/dlnM|.
Empirical fits are only used inside their published calibration domains when
the UI requests a validity mask; extrapolation remains available but labelled.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from engine.contracts import FIT_CONTRACTS


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


def nu_from_sigma(sigma, delta_c=1.686):
    return float(delta_c) / _s(sigma)


def f_press_schechter(sigma, delta_c=1.686, **_):
    nu = nu_from_sigma(sigma, delta_c)
    return np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)


def f_sheth_tormen(sigma, delta_c=1.686, A=0.3222, a=0.707, p=0.3, **_):
    nu = nu_from_sigma(sigma, delta_c)
    return (
        A
        * np.sqrt(2.0 * a / np.pi)
        * (1.0 + (1.0 / (a * nu**2)) ** p)
        * nu
        * np.exp(-0.5 * a * nu**2)
    )


def f_jenkins(sigma, **_):
    s = _s(sigma)
    return 0.315 * np.exp(-(np.abs(np.log(1.0 / s) + 0.61) ** 3.8))


def f_reed03(sigma, delta_c=1.686, **_):
    s = _s(sigma)
    return f_sheth_tormen(s, delta_c) * np.exp(-0.7 / (s * np.cosh(2.0 * s)) ** 5)


def f_warren(sigma, **_):
    s = _s(sigma)
    return 0.7234 * (s**-1.625 + 0.2538) * np.exp(-1.1982 / s**2)


def f_reed07(sigma, delta_c=1.686, neff=-2.0, **_):
    s = _s(sigma)
    dc = float(delta_c)
    a, p, ca, A = 0.707, 0.3, 1.08, 0.3222
    lninv = np.log(1.0 / s)
    g1 = np.exp(-((lninv - 0.4) ** 2) / 0.72)
    g2 = np.exp(-((lninv - 0.75) ** 2) / 0.08)
    prefactor = A * np.sqrt(2.0 * a / np.pi)
    bracket = 1.0 + (s**2 / (a * dc**2)) ** p + 0.6 * g1 + 0.4 * g2
    exponent = (
        -ca * a * dc**2 / (2.0 * s**2)
        - 0.03 / (float(neff) + 3.0) ** 2 * (dc / s) ** 0.6
    )
    return prefactor * bracket * (dc / s) * np.exp(exponent)


def f_tinker08(sigma, z=0.0, delta_halo=200.0, **_):
    """Tinker08 multiplicity; delta_halo is relative to mean matter density.

    Numerical evaluation outside the redshift calibration is exploratory;
    this function does not certify a cosmology or sigma range as calibrated.
    Overdensity extrapolation is never performed.
    """
    s, zp1, delta = _s(sigma), 1.0 + float(z), float(delta_halo)
    if not np.isfinite(delta) or not 200 <= delta <= 3200:
        raise ValueError("Tinker 2008 requires 200 ≤ Δmean ≤ 3200")
    if not np.isfinite(zp1) or zp1 < 1:
        raise ValueError("Redshift must be finite and nonnegative")
    A0, a0, b0, c = _TINKER_INTERPOLATOR(delta)
    alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
    A = A0 * zp1**-0.14
    a = a0 * zp1**-0.06
    b = b0 * zp1**-alpha
    return A * ((b / s) ** a + 1.0) * np.exp(-c / s**2)


def f_crocce10(sigma, z=0.0, **_):
    s, zp1 = _s(sigma), 1.0 + float(z)
    A, a, b, c = (
        0.58 * zp1**-0.13,
        1.37 * zp1**-0.15,
        0.30 * zp1**-0.084,
        1.036 * zp1**-0.024,
    )
    return A * (s**-a + b) * np.exp(-c / s**2)


def f_courtin10(sigma, delta_c=1.686, **_):
    return f_sheth_tormen(sigma, delta_c, A=0.348, a=0.695, p=0.1)


def f_bhattacharya11(sigma, z=0.0, delta_c=1.686, **_):
    s, dc, zp1 = _s(sigma), float(delta_c), 1.0 + float(z)
    A, a, p, q = 0.333 * zp1**-0.11, 0.788 * zp1**-0.01, 0.807, 1.795
    return (
        A
        * np.sqrt(2.0 / np.pi)
        * np.exp(-a * dc**2 / (2.0 * s**2))
        * (1.0 + (a * dc**2 / s**2) ** -p)
        * (dc * np.sqrt(a) / s) ** q
    )


def f_angulo12(sigma, subhalos=False, **_):
    s = _s(sigma)
    A, a, b, c = (0.265, 1.9, 1.675, 1.4) if subhalos else (0.201, 1.7, 2.08, 1.172)
    return A * ((b / s) ** a + 1.0) * np.exp(-c / s**2)


def f_watson_fof13(sigma, **_):
    s = _s(sigma)
    A, a, b, c = 0.282, 1.406, 2.163, 1.210
    return A * ((b / s) ** a + 1.0) * np.exp(-c / s**2)


def f_watson_so13(sigma, z=0.0, omega_m_z=0.3, delta_halo=200.0, **_):
    s, z, om, delta = _s(sigma), float(z), float(omega_m_z), float(delta_halo)
    if z == 0.0:
        A, a, b, c = 0.194, 2.267, 1.805, 1.287
    elif z >= 6.0:
        A, a, b, c = 0.563, 0.874, 3.810, 1.453
    else:
        A = om * (1.907 * (1.0 + z) ** -3.216 + 0.074)
        a = om * (3.136 * (1.0 + z) ** -3.058 + 2.349)
        b = om * (5.907 * (1.0 + z) ** -3.599 + 2.344)
        c = 1.318
    base = A * ((b / s) ** a + 1.0) * np.exp(-c / s**2)
    C = np.exp(0.023 * (delta / 178.0 - 1.0))
    d = -0.456 * om - 0.139
    gamma = C * (delta / 178.0) ** d * np.exp(0.072 * (1.0 - delta / 178.0) / s**2.130)
    return gamma * base


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
    aliases = {
        "Press-Schechter": "Press-Schechter 1974",
        "Sheth-Tormen": "Sheth-Tormen 2001",
    }
    return aliases.get(name, name)


def fitting_values(sigma, delta_c, fitting, z=0.0, **kwargs):
    name = canonical_name(fitting)
    if name not in _DISPATCH:
        raise ValueError(f"Unknown fitting function: {fitting}")
    return _DISPATCH[name](sigma, delta_c=delta_c, z=z, **kwargs)
