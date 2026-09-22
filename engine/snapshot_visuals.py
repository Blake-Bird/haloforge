"""Views and frame data derived from recorded GADGET particle snapshots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from engine.gadget_snapshot import GadgetSnapshotParticles


def particle_density_grid(snapshot: GadgetSnapshotParticles, cells: int = 32) -> np.ndarray:
    """Cloud-in-cell-free count density on the snapshot's periodic box.

    The grid is a visualization product. Each equal-mass particle contributes
    to exactly one cell, so the mean and total particle count are preserved.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or not 8 <= cells <= 128:
        raise ValueError("Density grid cells must be an integer from 8 to 128")
    positions = np.asarray(snapshot.positions_mpc_h, dtype=float)
    box = float(snapshot.box_size_mpc_h)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(box) or box <= 0:
        raise ValueError("Snapshot geometry is invalid")
    if not np.all(np.isfinite(positions)) or np.any(positions < 0) or np.any(positions >= box):
        raise ValueError("Snapshot particles fall outside the periodic box")
    edges = np.linspace(0.0, box, cells + 1)
    counts, _ = np.histogramdd(positions, bins=(edges, edges, edges))
    return counts.astype(np.int64)


def projected_overdensity(snapshot: GadgetSnapshotParticles, cells: int = 64, axis: int = 2) -> np.ndarray:
    """Return a dimensionless full-box surface-density contrast."""
    if axis not in (0, 1, 2):
        raise ValueError("Projection axis must be 0, 1, or 2")
    counts = particle_density_grid(snapshot, cells).sum(axis=axis)
    mean = float(counts.mean())
    if mean <= 0:
        raise ValueError("A density view needs at least one particle")
    return counts / mean - 1.0


def density_slice_figure(snapshot: GadgetSnapshotParticles, *, cells: int = 64, axis: int = 2) -> go.Figure:
    """Render a measured particle projection with physical box coordinates."""
    contrast = projected_overdensity(snapshot, cells, axis)
    box = float(snapshot.box_size_mpc_h)
    figure = go.Figure(go.Heatmap(
        z=contrast.T,
        x=(np.arange(cells) + 0.5) * box / cells,
        y=(np.arange(cells) + 0.5) * box / cells,
        colorscale="Viridis",
        colorbar={"title": "Σ/Σ̄ − 1"},
        hovertemplate="x=%{x:.2f}, y=%{y:.2f} h⁻¹ Mpc<br>Σ/Σ̄ − 1=%{z:.3f}<extra></extra>",
    ))
    figure.update_layout(
        title=f"Recorded particle surface density · z={snapshot.redshift:g}",
        xaxis_title="x [h⁻¹ Mpc]",
        yaxis_title="y [h⁻¹ Mpc]",
        meta={
            "caption": f"Projection of {snapshot.particle_ids.size:,} recorded particles through the full {box:g} h⁻¹ Mpc box. Each particle enters one grid cell.",
            "mislead": "This grid has finite cell resolution and is not a halo finder. Colors encode a projected count density, not a 3D mass profile.",
        },
    )
    figure.update_xaxes(range=[0, box], constrain="domain")
    figure.update_yaxes(range=[0, box], scaleanchor="x", scaleratio=1)
    return figure


def particle_cube_figure(snapshot: GadgetSnapshotParticles, *, max_points: int = 20_000) -> go.Figure:
    """Render a deterministic subset of actual evolved particle coordinates."""
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    count = int(snapshot.particle_ids.size)
    indices = np.linspace(0, count - 1, min(count, max_points), dtype=int)
    positions = snapshot.positions_mpc_h[indices]
    box = float(snapshot.box_size_mpc_h)
    figure = go.Figure(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers",
        marker={"size": 1.5, "color": "#5dcfc5", "opacity": 0.38},
        hovertemplate="x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f} h⁻¹ Mpc<extra></extra>",
        name="Recorded DM particles",
    ))
    figure.update_layout(
        title=f"Recorded DM particle cube · z={snapshot.redshift:g}",
        scene={
            "aspectmode": "cube",
            "xaxis": {"title": "x [h⁻¹ Mpc]", "range": [0, box]},
            "yaxis": {"title": "y [h⁻¹ Mpc]", "range": [0, box]},
            "zaxis": {"title": "z [h⁻¹ Mpc]", "range": [0, box]},
        },
        meta={
            "caption": f"Displays {indices.size:,} deterministic samples from {count:,} recorded dark-matter particles in a {box:g} h⁻¹ Mpc box.",
            "mislead": "This is a particle view. Visible clumps are not counted halos until a finder measures a catalogue.",
        },
    )
    return figure
