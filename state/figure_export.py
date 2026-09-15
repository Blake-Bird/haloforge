"""Static publication figures derived solely from a saved HaloForge run."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


class FigureExportError(RuntimeError):
    """Raised when static figure export is unavailable or invalid."""


PAPER_COLORS = ("#17627f", "#8a5a14", "#804d87")
GRAYSCALE_COLORS = ("#111111", "#5f5f5f", "#969696")
GRAYSCALE_DASHES = ("solid", "dash", "dot")


def _paper_layout(fig: go.Figure, title: str, caption: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        width=960,
        height=650,
        title=dict(
            text=title, font=dict(family="Arial", size=22, color="#102a3b"), x=0.02
        ),
        font=dict(family="Arial", size=14, color="#17252c"),
        margin=dict(l=90, r=36, t=90, b=150),
        legend=dict(orientation="h", y=1.02, x=0),
        annotations=[
            dict(
                text=caption,
                x=0,
                xref="paper",
                y=-0.22,
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#40535d"),
            )
        ],
    )
    return fig


def _static_formats(figures: list[tuple[str, go.Figure]]) -> dict[str, bytes]:
    """Batch render all figures so Kaleido starts once per output format."""
    output: dict[str, bytes] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="haloforge-figures-") as folder:
            root = Path(folder)
            for image_format in ("pdf", "svg", "png"):
                paths = [root / f"{stem}.{image_format}" for stem, _ in figures]
                pio.write_images(
                    [figure for _, figure in figures],
                    paths,
                    format=image_format,
                    width=960,
                    height=650,
                    scale=2 if image_format == "png" else 1,
                )
                output.update({path.name: path.read_bytes() for path in paths})
    except Exception as exc:
        raise FigureExportError(f"Static figure export failed: {exc}") from exc
    return output


def _grayscale_variant(fig: go.Figure, title: str, caption: str) -> go.Figure:
    """Make a print-safe figure whose series remain distinct without color."""
    variant = go.Figure(fig)
    for index, trace in enumerate(variant.data):
        trace.line.color = GRAYSCALE_COLORS[index % len(GRAYSCALE_COLORS)]
        trace.line.dash = GRAYSCALE_DASHES[index % len(GRAYSCALE_DASHES)]
        trace.line.width = 3.2 if index == 0 else 2.8
    return _paper_layout(
        variant,
        f"{title} - grayscale-safe",
        f"{caption}<br>This version uses grayscale contrast and line patterns for print-safe interpretation.",
    )


def _with_grayscale_variants(
    figures: list[tuple[str, go.Figure, str, str]],
) -> list[tuple[str, go.Figure]]:
    paired: list[tuple[str, go.Figure]] = []
    for stem, figure, title, caption in figures:
        paired.append((stem, figure))
        paired.append((f"{stem}_grayscale", _grayscale_variant(figure, title, caption)))
    return paired


def build_figure_exports(run: dict) -> dict[str, bytes]:
    """Build matching PDF, SVG, and PNG figures with captioned scientific scope."""
    arrays = run.get("arrays", {})
    k, power = (
        np.asarray(arrays.get("k", []), dtype=float),
        np.asarray(arrays.get("P", []), dtype=float),
    )
    if (
        k.ndim != 1
        or power.shape != k.shape
        or k.size < 2
        or np.any(k <= 0)
        or np.any(power <= 0)
    ):
        raise FigureExportError(
            "A positive stored P(k) grid is required for static figure export."
        )
    figures: list[tuple[str, go.Figure, str, str]] = []
    power_fig = go.Figure(
        go.Scatter(
            x=k,
            y=power,
            mode="lines",
            line=dict(color=PAPER_COLORS[0], width=3),
            name="stored linear P(k)",
            hovertemplate="k=%{x:.3e} Mpc⁻¹<br>P=%{y:.3e} Mpc³<extra></extra>",
        )
    )
    power_fig.update_xaxes(type="log", title="k [Mpc⁻¹]")
    power_fig.update_yaxes(type="log", title="Linear P(k) [Mpc³]")
    power_title = "Linear matter power spectrum"
    power_caption = "Stored AxiCLASS/CLASS linear spectrum at the focused redshift. Exact source and solver metadata: citation_metadata.json, class_settings.json, and provenance.json."
    figures.append(
        (
            "matter_power",
            _paper_layout(power_fig, power_title, power_caption),
            power_title,
            power_caption,
        )
    )

    masses, sigma = (
        np.asarray(arrays.get("M_h", []), dtype=float),
        np.asarray(arrays.get("sigma", []), dtype=float),
    )
    if (
        masses.ndim == 1
        and sigma.shape == masses.shape
        and masses.size >= 2
        and np.all(masses > 0)
        and np.all(sigma > 0)
    ):
        sigma_fig = go.Figure(
            go.Scatter(
                x=masses,
                y=sigma,
                mode="lines",
                line=dict(color=PAPER_COLORS[1], width=3),
                name="stored sigma(M)",
            )
        )
        sigma_fig.update_xaxes(type="log", title="M [h⁻¹ M☉]")
        sigma_fig.update_yaxes(type="log", title="σ(M)")
        sigma_title = "Mass variance"
        sigma_caption = "Stored fixed-grid smoothing result. Sources: citation_metadata.json. Sampled-range checks do not establish complete solver convergence; see scientific_validity.json."
        figures.append(
            (
                "mass_variance",
                _paper_layout(sigma_fig, sigma_title, sigma_caption),
                sigma_title,
                sigma_caption,
            )
        )

    ps, st = (
        np.asarray(arrays.get("hmf_press_schechter_z0", []), dtype=float),
        np.asarray(arrays.get("hmf_sheth_tormen_z0", []), dtype=float),
    )
    if (
        masses.ndim == 1
        and ps.shape == masses.shape
        and st.shape == masses.shape
        and masses.size >= 2
        and np.all(ps > 0)
        and np.all(st > 0)
    ):
        hmf_fig = go.Figure()
        hmf_fig.add_trace(
            go.Scatter(
                x=masses,
                y=ps,
                mode="lines",
                line=dict(color=PAPER_COLORS[0], width=2.8),
                name="Press-Schechter 1974",
            )
        )
        hmf_fig.add_trace(
            go.Scatter(
                x=masses,
                y=st,
                mode="lines",
                line=dict(color=PAPER_COLORS[2], width=2.8),
                name="Sheth-Tormen 2001",
            )
        )
        hmf_fig.update_xaxes(type="log", title="M [h⁻¹ M☉]")
        hmf_fig.update_yaxes(type="log", title="dn/dln M [h³ Mpc⁻³]")
        hmf_title = "Analytic halo mass-function references"
        hmf_caption = "Analytic top-hat reference curves at z=0. Sources: citation_metadata.json. They are not a universal empirical calibration; inspect scientific_validity.json before publication use."
        figures.append(
            (
                "analytic_hmf_reference",
                _paper_layout(hmf_fig, hmf_title, hmf_caption),
                hmf_title,
                hmf_caption,
            )
        )
    return _static_formats(_with_grayscale_variants(figures))


def build_figure_pdfs(run: dict) -> dict[str, bytes]:
    """Backward-compatible PDF-only view of :func:`build_figure_exports`."""
    return {
        filename: content
        for filename, content in build_figure_exports(run).items()
        if filename.endswith(".pdf")
    }
