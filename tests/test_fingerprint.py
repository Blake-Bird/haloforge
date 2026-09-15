import numpy as np

from engine.fingerprint import (
    FINGERPRINT_VERSION,
    cosmic_fingerprint,
    fingerprint_markdown,
)


def _run(amplitude: float, sigma8: float):
    return {
        "params": {
            "A_s": amplitude,
            "n_s": 0.965,
            "k_pivot": 0.05,
            "selected_mass_exp": 12,
        },
        "sigma8": sigma8,
        "arrays": {
            "k": np.array([1e-3, 0.05, 1.0]),
            "P": np.array([1.0, amplitude / 1e-9, 0.2]),
            "M_h": np.array([1e10, 1e12, 1e14]),
            "sigma": np.array([2.0, amplitude / 1e-9, 0.4]),
        },
    }


def test_cosmic_fingerprint_uses_stored_matched_deltas_and_clear_scope_boundary():
    fingerprint = cosmic_fingerprint(_run(2.4e-9, 0.9), _run(2.0e-9, 0.8))
    assert fingerprint["version"] == FINGERPRINT_VERSION
    assert fingerprint["has_baseline"]
    assert fingerprint["parameter_changes"][0]["parameter"] == "A_s"
    assert (
        next(signal for signal in fingerprint["signals"] if signal["label"] == "σ₈")[
            "fractional_change"
        ]
        == 0.125
    )
    assert "not a literal simulated universe" in fingerprint["scope_limit"]


def test_standalone_fingerprint_does_not_invent_a_comparison():
    fingerprint = cosmic_fingerprint(_run(2.1e-9, 0.81))
    assert not fingerprint["has_baseline"]
    assert not fingerprint["signals"]
    assert "No named baseline" in fingerprint_markdown(fingerprint)
