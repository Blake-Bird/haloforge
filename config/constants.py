"""Constants used by the Chunk 2 parameter and derived-quantity system."""

from __future__ import annotations

RHO_CRIT_COEFF = 2.775e11
T_CMB_DEFAULT = 2.7255
OMEGA_GAMMA_H2_DEFAULT = 2.469e-5
NEUTRINO_RADIATION_FACTOR = 0.22710731766
DELTA_C_DEFAULT = 1.686
Z_PRESETS = [0, 0.5, 1, 2, 5, 10, 100]
WINDOW_TYPES = ["Top-hat", "Gaussian", "Sharp-k"]
from engine.fitting_functions import FITTING_NAMES

FITTING_FUNCTIONS = FITTING_NAMES
N_EDE_OPTIONS = [2, 3, 4, 5, 6]
