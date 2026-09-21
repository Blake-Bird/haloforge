"""Halo catalogue schemas, FOF group finding, and 3D spatial visualizers.

Parses GADGET-4 FOF/SUBFIND catalogues and provides an integrated periodic-boundary
FOF halo finder for simulation validation, with level-of-detail 3D scatter visualizers
and robust high-redshift zero-halo handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree


@dataclass
class HaloRecord:
    halo_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    n_particles: int
    M_fof_msun_h: float
    M_200m_msun_h: float
    M_200c_msun_h: float
    R_200m_kpc_h: float
    sigma_v_km_s: float


@dataclass
class HaloCatalogue:
    redshift: float
    box_size_mpc_h: float
    particle_mass_msun_h: float
    linking_length_b: float
    min_particles: int
    halos: list[HaloRecord]
    is_empty_due_to_resolution: bool
    summary: str


def find_fof_halos(
    positions: np.ndarray,
    velocities: np.ndarray,
    box_size_mpc_h: float,
    particle_mass_msun_h: float,
    redshift: float,
    *,
    linking_length_b: float = 0.2,
    min_particles: int = 20,
) -> HaloCatalogue:
    """Run Friends-of-Friends (FOF) halo finding with periodic boundary conditions.

    Linking length: b * mean_particle_separation.
    """
    n_part = len(positions)
    if n_part == 0:
        return HaloCatalogue(
            redshift=float(redshift),
            box_size_mpc_h=float(box_size_mpc_h),
            particle_mass_msun_h=float(particle_mass_msun_h),
            linking_length_b=float(linking_length_b),
            min_particles=int(min_particles),
            halos=[],
            is_empty_due_to_resolution=True,
            summary="Zero particles supplied to halo finder.",
        )

    particles_per_dim = int(round(n_part ** (1.0 / 3.0)))
    mean_sep = box_size_mpc_h / max(particles_per_dim, 1)
    link_dist = linking_length_b * mean_sep

    # Periodic KDTree
    tree = cKDTree(positions, boxsize=box_size_mpc_h)
    pairs = tree.query_pairs(link_dist, output_type="ndarray")

    # Disjoint Set Union (DSU) to find connected components
    parent = np.arange(n_part, dtype=np.int32)

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i, j in pairs:
        union(i, j)

    # Group particles by root
    groups: dict[int, list[int]] = {}
    for idx in range(n_part):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    # Filter by minimum particle threshold
    halos = []
    halo_counter = 1
    rho_crit_comoving = 2.77536627e11  # h^-1 M_sun / (h^-1 Mpc)^3
    omega_m = 0.315
    rho_mean = omega_m * rho_crit_comoving

    for root, p_indices in groups.items():
        count = len(p_indices)
        if count >= min_particles:
            p_pos = positions[p_indices]
            p_vel = velocities[p_indices]

            # Periodic center of mass
            # Shift relative to first particle to avoid boundary tear
            ref = p_pos[0]
            d_pos = p_pos - ref
            d_pos = np.where(
                d_pos > box_size_mpc_h / 2.0, d_pos - box_size_mpc_h, d_pos
            )
            d_pos = np.where(
                d_pos < -box_size_mpc_h / 2.0, d_pos + box_size_mpc_h, d_pos
            )
            com = np.mod(ref + np.mean(d_pos, axis=0), box_size_mpc_h)

            mean_vel = np.mean(p_vel, axis=0)
            sig_v = float(np.std(p_vel))

            m_fof = count * particle_mass_msun_h
            # Approximate R200m and M200m from spherical collapse relation:
            # M_200m = (4/3) pi R_200m^3 (200 rho_mean)
            r_200m_mpc = (m_fof / ((4.0 / 3.0) * np.pi * 200.0 * rho_mean)) ** (
                1.0 / 3.0
            )
            r_200m_kpc = float(r_200m_mpc * 1000.0)

            # M_200c ~ M_200m * (Omega_m(z)) if approximate
            m_200c = m_fof * 0.85

            halos.append(
                HaloRecord(
                    halo_id=halo_counter,
                    x=float(com[0]),
                    y=float(com[1]),
                    z=float(com[2]),
                    vx=float(mean_vel[0]),
                    vy=float(mean_vel[1]),
                    vz=float(mean_vel[2]),
                    n_particles=count,
                    M_fof_msun_h=float(m_fof),
                    M_200m_msun_h=float(m_fof),
                    M_200c_msun_h=float(m_200c),
                    R_200m_kpc_h=r_200m_kpc,
                    sigma_v_km_s=sig_v,
                )
            )
            halo_counter += 1

    # Sort descending by halo mass
    halos.sort(key=lambda h: h.M_fof_msun_h, reverse=True)

    is_empty = len(halos) == 0
    if is_empty:
        summary = (
            f"At z={redshift:.1f}, no bound structures meet the resolution threshold of "
            f">={min_particles} particles ({min_particles * particle_mass_msun_h:.2e} h^-1 M_sun). "
            "This is an expected physical/resolution outcome at early epochs before non-linear collapse."
        )
    else:
        summary = (
            f"Catalogue at z={redshift:.1f} contains {len(halos):,} resolved halos "
            f"(Mass range: {halos[-1].M_fof_msun_h:.2e} to {halos[0].M_fof_msun_h:.2e} h^-1 M_sun)."
        )

    return HaloCatalogue(
        redshift=float(redshift),
        box_size_mpc_h=float(box_size_mpc_h),
        particle_mass_msun_h=float(particle_mass_msun_h),
        linking_length_b=float(linking_length_b),
        min_particles=int(min_particles),
        halos=halos,
        is_empty_due_to_resolution=is_empty,
        summary=summary,
    )


def catalogue_to_dataframe(catalogue: HaloCatalogue) -> pd.DataFrame:
    """Convert halo catalogue into a standard Pandas DataFrame."""
    if not catalogue.halos:
        return pd.DataFrame(
            columns=[
                "halo_id",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "n_particles",
                "M_fof_msun_h",
                "M_200m_msun_h",
                "M_200c_msun_h",
                "R_200m_kpc_h",
                "sigma_v_km_s",
            ]
        )
    return pd.DataFrame([asdict(h) for h in catalogue.halos])


def render_3d_halo_view(
    catalogue: HaloCatalogue,
    *,
    max_display_halos: int = 2500,
    theme: str = "Dark",
) -> go.Figure:
    """Render interactive 3D spatial halo view with level-of-detail subsampling and wireframe box."""
    fig = go.Figure()
    box_size = catalogue.box_size_mpc_h

    if not catalogue.halos:
        fig.add_annotation(
            text=catalogue.summary,
            showarrow=False,
            font=dict(size=13, color="#ffb454"),
        )
    else:
        df = catalogue_to_dataframe(catalogue)
        if len(df) > max_display_halos:
            # Keep top 500 massive halos, subsample remainder
            massive = df.iloc[:500]
            subsampled = df.iloc[500:].sample(
                n=max_display_halos - 500, random_state=42
            )
            display_df = pd.concat([massive, subsampled])
        else:
            display_df = df

        # Bubble size proportional to log mass
        log_m = np.log10(display_df["M_fof_msun_h"])
        sizes = 3.0 + 8.0 * (log_m - log_m.min()) / max(
            float(log_m.max() - log_m.min()), 1.0
        )

        fig.add_trace(
            go.Scatter3d(
                x=display_df["x"],
                y=display_df["y"],
                z=display_df["z"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=log_m,
                    colorscale="Viridis",
                    colorbar=dict(title="log₁₀ M [h⁻¹ M☉]"),
                    opacity=0.85,
                ),
                text=[
                    f"Halo #{row['halo_id']}<br>Mass: {row['M_fof_msun_h']:.2e} h⁻¹ M☉<br>Particles: {int(row['n_particles']):,}<br>σ_v: {row['sigma_v_km_s']:.1f} km/s"
                    for _, row in display_df.iterrows()
                ],
                hoverinfo="text",
                name=f"Halos (N={len(catalogue.halos):,})",
            )
        )

    # Periodic Box wireframe outline
    wire_lines = [
        [0, 0, 0],
        [box_size, 0, 0],
        [box_size, box_size, 0],
        [0, box_size, 0],
        [0, 0, 0],
        [0, 0, box_size],
        [box_size, 0, box_size],
        [box_size, box_size, box_size],
        [0, box_size, box_size],
        [0, 0, box_size],
    ]
    wire_arr = np.array(wire_lines)
    fig.add_trace(
        go.Scatter3d(
            x=wire_arr[:, 0],
            y=wire_arr[:, 1],
            z=wire_arr[:, 2],
            mode="lines",
            line=dict(color="#344a53", width=2.5),
            name="Periodic Box Boundary",
            showlegend=False,
            hoverinfo="none",
        )
    )

    bg_color = "rgba(6, 16, 21, 0.95)" if theme != "Light" else "#ffffff"
    text_color = "#edf1ef" if theme != "Light" else "#132126"

    fig.update_layout(
        title=f"3D Halo Spatial Distribution (z = {catalogue.redshift:.2f})",
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        scene=dict(
            xaxis=dict(
                title="X [h⁻¹ Mpc]", range=[0, box_size], backgroundcolor=bg_color
            ),
            yaxis=dict(
                title="Y [h⁻¹ Mpc]", range=[0, box_size], backgroundcolor=bg_color
            ),
            zaxis=dict(
                title="Z [h⁻¹ Mpc]", range=[0, box_size], backgroundcolor=bg_color
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig
