"""Default parameter values for the HaloForge control system."""

from __future__ import annotations

DEFAULT_PARAMS = {
    "mode": "Learning",
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "k_pivot": 0.05,
    "H0": 67.81,
    "Omega_m": 0.3098830430481205,
    "Omega_b": 0.0493,
    "Omega_k": 0.0,
    "N_eff": 3.046,
    "Tcmb": 2.7255,
    "tau_reio": 0.06,
    "enable_ede": True,
    "f_EDE": 0.10,
    "log10_a_c": -3.5,
    "n_EDE": 3,
    "scf_parameters": "2.806,0.0",
    "k_min": 1e-4,
    "k_max": 200.0,
    "k_points": 1200,
    "quad_limit": 200,
    "window_type": "Top-hat",
    "fitting": "Sheth-Tormen 2001",
    "delta_halo": 200.0,
    "delta_c": 1.686,
    "mass_min_exp": 8,
    "mass_max_exp": 15,
    "mass_points": 80,
    "selected_mass_exp": 12,
    "single_z": 0.0,
    "z_presets_selected": [0, 0.5, 1, 2, 5, 10, 100],
    "custom_z_list": "",
    "z_values": [0, 0.5, 1, 2, 5, 10, 100],
}

PAGE_TITLE = "HaloForge"
