"""Published halo multiplicity functions f(sigma).

The HMF convention is dn/dlnM=(rho_m/M) f(sigma) |dlnsigma/dlnM|.
Empirical fits are only used inside their published calibration domains when
the UI requests a validity mask; extrapolation remains available but labelled.
"""

from __future__ import annotations

import numpy as np


FITTING_NAMES = [
    "Press-Schechter 1974", "Sheth-Tormen 2001", "Jenkins 2001",
    "Reed 2003", "Warren 2006", "Reed 2007", "Tinker 2008",
    "Crocce 2010", "Courtin 2010", "Bhattacharya 2011",
    "Angulo 2012", "Watson FOF 2013", "Watson SO 2013",
]

FIT_METADATA = {
    "Press-Schechter 1974": ("analytic", "all", "spherical collapse"),
    "Sheth-Tormen 2001": ("analytic", "all", "ellipsoidal / Einstein-de Sitter"),
    "Jenkins 2001": ("-1.2 < ln sigma^-1 < 1.05", "0-5", "tauCDM, LCDM"),
    "Reed 2003": ("-1.7 < ln sigma^-1 < 0.9", "0-15", "Omega_m=0.3, Omega_L=0.7"),
    "Warren 2006": ("10^10 < M/Msun < 10^15", "0", "LCDM / WMAP1"),
    "Reed 2007": ("-1.7 < ln sigma^-1 < 0.9", "0-30", "LCDM / WMAP1"),
    "Tinker 2008": ("-0.6 < ln sigma^-1 < 0.4", "0-2.5", "LCDM / WMAP1,3+"),
    "Crocce 2010": ("10^10.5 < M/Msun < 10^15.5", "0-2", "LCDM suite"),
    "Courtin 2010": ("-0.8 < ln sigma^-1 < 0.7", "0", "LCDM / WMAP5"),
    "Bhattacharya 2011": ("10^11.8 < M/Msun < 10^15.5", "0-2", "wCDM+"),
    "Angulo 2012": ("10^8 < M/Msun < 10^16", "0", "LCDM / WMAP1"),
    "Watson FOF 2013": ("-0.55 < ln sigma^-1 < 1.31", "0-30", "LCDM / WMAP5"),
    "Watson SO 2013": ("redshift-dependent peak-height range", "0-30", "LCDM / WMAP5"),
}


def _s(sigma):
    return np.maximum(np.asarray(sigma, dtype=float), 1e-30)


def nu_from_sigma(sigma, delta_c=1.686):
    return float(delta_c) / _s(sigma)


def f_press_schechter(sigma, delta_c=1.686, **_):
    nu = nu_from_sigma(sigma, delta_c)
    return np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)


def f_sheth_tormen(sigma, delta_c=1.686, A=0.3222, a=0.707, p=0.3, **_):
    nu = nu_from_sigma(sigma, delta_c)
    return A * np.sqrt(2.0 * a / np.pi) * (1.0 + (1.0 / (a * nu**2)) ** p) * nu * np.exp(-0.5 * a * nu**2)


def f_jenkins(sigma, **_):
    s = _s(sigma)
    return 0.315 * np.exp(-np.abs(np.log(1.0 / s) + 0.61) ** 3.8)


def f_reed03(sigma, delta_c=1.686, **_):
    s = _s(sigma)
    return f_sheth_tormen(s, delta_c) * np.exp(-0.7 / (s * np.cosh(2.0 * s)) ** 5)


def f_warren(sigma, **_):
    s = _s(sigma)
    return 0.7234 * (s ** -1.625 + 0.2538) * np.exp(-1.1982 / s**2)


def f_reed07(sigma, delta_c=1.686, neff=-2.0, **_):
    s = _s(sigma)
    dc = float(delta_c)
    a, p, ca, A = 0.707, 0.3, 1.08, 0.3222
    lninv = np.log(1.0 / s)
    g1 = np.exp(-((lninv - 0.4) ** 2) / 0.72)
    g2 = np.exp(-((lninv - 0.75) ** 2) / 0.08)
    prefactor = A * np.sqrt(2.0 * a / np.pi)
    bracket = 1.0 + (s**2 / (a * dc**2)) ** p + 0.6 * g1 + 0.4 * g2
    exponent = -ca * a * dc**2 / (2.0 * s**2) - 0.03 / (float(neff) + 3.0) ** 2 * (dc / s) ** 0.6
    return prefactor * bracket * (dc / s) * np.exp(exponent)


def f_tinker08(sigma, z=0.0, delta_halo=200.0, **_):
    s, zp1, delta = _s(sigma), 1.0 + float(z), float(delta_halo)
    alpha = 10.0 ** (-(0.75 / np.log10(delta / 75.0)) ** 1.2)
    A = 0.186 * zp1 ** -0.14
    a = 1.47 * zp1 ** -0.06
    b = 2.57 * zp1 ** -alpha
    c = 1.19
    return A * ((b / s) ** a + 1.0) * np.exp(-c / s**2)


def f_crocce10(sigma, z=0.0, **_):
    s, zp1 = _s(sigma), 1.0 + float(z)
    A, a, b, c = 0.58 * zp1**-0.13, 1.37 * zp1**-0.15, 0.30 * zp1**-0.084, 1.036 * zp1**-0.024
    return A * (s**-a + b) * np.exp(-c / s**2)


def f_courtin10(sigma, delta_c=1.686, **_):
    return f_sheth_tormen(sigma, delta_c, A=0.348, a=0.695, p=0.1)


def f_bhattacharya11(sigma, z=0.0, delta_c=1.686, **_):
    s, dc, zp1 = _s(sigma), float(delta_c), 1.0 + float(z)
    A, a, p, q = 0.333 * zp1**-0.11, 0.788 * zp1**-0.01, 0.807, 1.795
    return A * np.sqrt(2.0 / np.pi) * np.exp(-a * dc**2 / (2.0 * s**2)) * (1.0 + (a * dc**2 / s**2) ** -p) * (dc * np.sqrt(a) / s) ** q


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
    aliases = {"Press-Schechter": "Press-Schechter 1974", "Sheth-Tormen": "Sheth-Tormen 2001"}
    return aliases.get(name, name)


def fitting_values(sigma, delta_c, fitting, z=0.0, **kwargs):
    name = canonical_name(fitting)
    if name not in _DISPATCH:
        raise ValueError(f"Unknown fitting function: {fitting}")
    return _DISPATCH[name](sigma, delta_c=delta_c, z=z, **kwargs)
