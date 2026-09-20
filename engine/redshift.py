"""Exact sampled-redshift selection for calculations and plots."""

import numpy as np


def redshift_index(redshifts, requested: float) -> int:
    """Find one matching saved sample without substituting a nearby redshift."""
    redshifts = np.asarray(redshifts, dtype=float)
    requested = float(requested)
    if (
        redshifts.ndim != 1
        or redshifts.size == 0
        or not np.isfinite(redshifts).all()
        or np.any(redshifts < 0)
        or np.any(np.diff(redshifts) <= 0)
        or not np.isfinite(requested)
        or requested < 0
    ):
        raise ValueError(
            "Saved and requested redshifts must be finite, nonnegative, and ordered"
        )
    indices = np.flatnonzero(np.isclose(redshifts, requested, rtol=0, atol=1e-9))
    if indices.size != 1:
        raise ValueError(
            "This run does not include one unambiguous sample at the requested redshift"
        )
    return int(indices[0])
