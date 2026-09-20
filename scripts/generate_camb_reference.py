"""Generate the independent ΛCDM fixture; never import HaloForge calculations.

Requires the pinned packages in requirements-reference.txt. Review numerical
changes before replacing the committed reference. Tolerances are declared here
before comparison with HaloForge, rather than fitted to observed discrepancies.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import camb
import numpy as np


PARAMETERS = {
    "H0": 67.81,
    "Omega_m": 0.3098830430481205,
    "Omega_b": 0.0493,
    "Omega_k": 0.0,
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "k_pivot": 0.05,
    "N_eff": 3.046,
    "Tcmb": 2.7255,
    "tau_reio": 0.06,
    "enable_ede": False,
}
TOLERANCES = {"power_relative": 0.01, "sigma_relative": 0.003}


def main():
    if camb.__version__ != "1.6.6":
        raise RuntimeError("Reference generation requires CAMB 1.6.6")
    h = PARAMETERS["H0"] / 100
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=PARAMETERS["H0"],
        ombh2=PARAMETERS["Omega_b"] * h**2,
        omch2=(PARAMETERS["Omega_m"] - PARAMETERS["Omega_b"]) * h**2,
        omk=0.0,
        mnu=0.0,
        num_massive_neutrinos=0,
        nnu=PARAMETERS["N_eff"],
        TCMB=PARAMETERS["Tcmb"],
        tau=PARAMETERS["tau_reio"],
    )
    params.InitPower.set_params(
        As=PARAMETERS["A_s"], ns=PARAMETERS["n_s"], pivot_scalar=PARAMETERS["k_pivot"]
    )
    params.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0)
    params.WantCls = False
    params.set_matter_power(
        redshifts=[10.0, 2.0, 0.0],
        kmax=50.0,
        k_per_logint=30,
        nonlinear=False,
        accurate_massive_neutrino_transfers=True,
    )
    results = camb.get_results(params)
    interpolate = results.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=False,
        k_hunit=False,
        var1="delta_tot",
        var2="delta_tot",
    )
    redshifts = np.array([0.0, 2.0, 10.0])
    k = np.geomspace(1e-4, 10.0, 81)
    masses_h = np.geomspace(1e10, 1e15, 11)
    density = PARAMETERS["Omega_m"] * 2.775e11 * h**2
    radii = (3 * (masses_h / h) / (4 * np.pi * density)) ** (1 / 3)
    # CAMB stores the requested transfer redshifts in descending order.
    sigma8 = results.get_sigma8()[::-1]
    sigma_m = results.get_sigmaR(
        radii, hubble_units=False, var1="delta_tot", var2="delta_tot"
    )[::-1]
    reference = {
        "schema": "haloforge-independent-camb-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_camb_reference.py",
        "camb_version": camb.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "parameters": PARAMETERS,
        "camb_settings": str(params),
        "tolerances": TOLERANCES,
        "units": {
            "k": "Mpc^-1",
            "power": "Mpc^3",
            "mass": "h^-1 Msun",
            "sigma": "dimensionless",
        },
        "scope": "Linear total-matter power and top-hat variance in one flat, massless-neutrino LCDM cosmology. Does not validate EDE, nonlinear evolution or empirical HMF calibration.",
        "redshifts": redshifts.tolist(),
        "k": k.tolist(),
        "power_by_z": interpolate.P(redshifts, k).tolist(),
        "mass_h": masses_h.tolist(),
        "sigma_by_z": sigma_m.tolist(),
        "sigma8_by_z": sigma8.tolist(),
    }
    target = Path(__file__).resolve().parents[1] / "tests/reference/camb_lcdm.json"
    target.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {target}; sigma8(z=0)={sigma8[0]:.9f}")


if __name__ == "__main__":
    main()
