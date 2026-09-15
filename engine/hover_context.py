"""Scientific context embedded in interactive figure hovers."""

from __future__ import annotations

from html import escape


HOVER_CONTEXT = {
    "Primordial curvature spectrum": "This is the starting fluctuation power at this comoving scale; compare its location with kₚ to read a tilt change.",
    "Linear matter power": "This is linear matter clustering at this scale and redshift, not a nonlinear field or a galaxy-survey measurement.",
    "Dimensionless power per ln k": "This is the unsmoothed variance contribution per logarithmic scale interval; it does not itself prove convergence.",
    "Scale-dependent processing shape": "This normalized shape proxy highlights transfer processing after dividing out the primordial tilt; it is not an independently fitted transfer function.",
    "Squared smoothing response": "This is how strongly this dimensionless mode contributes after smoothing; σ² uses W², so sign information from W is absent here.",
    "Smoothing-window response": "This is a mathematical Fourier weighting response, not a literal edge around an individual halo.",
    "Small-kR top-hat validation": "This is agreement between two numerical evaluations near small kR, not validation of the physical model.",
    "Mass variance": "This is the linear RMS density contrast after smoothing on the mass scale; it is not a measured halo catalogue.",
    "Logarithmic variance slope": "This local slope is one factor in the differential HMF and can be sensitive to mass sampling at the endpoints.",
    "Mass to top-hat radius": "This is the mean-density top-hat radius assigned to the displayed mass convention.",
    "Growth consistency check": "This compares two internal linear-growth indicators; agreement is an implementation check, not an independent solver validation.",
    "Halo multiplicity functions": "This maps fluctuation rarity into a collapse prescription or empirical fit; its calibration domain is fit-specific.",
    "Differential halo mass function": "This is predicted abundance per logarithmic mass interval under the selected fit; dotted values are outside the stated calibration checks.",
    "Cumulative halo abundance": "This is the finite sampled abundance above this mass, integrated only to the largest stored mass rather than infinity.",
    "Controlled finite sensitivity": "This point comes from an already saved one-parameter comparison; it is not a posterior distribution or an emulator prediction.",
}

_EXTRA = "<extra></extra>"


def hover_context(title: str) -> str:
    """Return a specific explanation when known, otherwise an honest fallback."""
    return HOVER_CONTEXT.get(
        str(title),
        "This cursor reports a stored plotted sample. Interpret it with the chart caption, stated assumptions, and numerical/calibration limits.",
    )


def with_hover_context(template: str | None, context: str) -> str:
    """Add one escaped scientific explanation without losing a trace's values."""
    text = str(template or "%{x}<br>%{y}")
    if "Scientific meaning:" in text:
        return text
    text = text.replace(_EXTRA, "")
    return (
        text.rstrip("<br>")
        + "<br><br><b>Scientific meaning:</b> "
        + escape(str(context))
        + _EXTRA
    )
