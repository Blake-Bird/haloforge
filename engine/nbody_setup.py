"""N-body simulation box setup, particle resolution contracts, and physical validation.

Computes exact particle mass, mean separation, Nyquist frequency, softening policy,
halo-mass particle count thresholds (20, 50, 100, 300, 1000 particles), memory footprint,
and snapshot sizes for cosmological dark matter and hydrodynamic SPH boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


from config.defaults import DEFAULT_PARAMS


@dataclass(frozen=True)
class BoxResolutionContracts:
    """Exact spatial, frequency, and mass resolution metrics for a periodic box."""

    box_size_mpc_h: float
    particles_per_dimension: int
    total_particles: int
    particle_mass_msun_h: float
    mean_separation_mpc_h: float
    k_fundamental_h_mpc: float
    k_nyquist_h_mpc: float
    recommended_softening_kpc_h: float
    min_halo_mass_20p_msun_h: float
    reliable_abundance_mass_100p_msun_h: float
    well_resolved_mass_300p_msun_h: float
    profile_resolved_mass_1000p_msun_h: float
    estimated_memory_gb: float
    estimated_snapshot_gb: float
    warnings: list[str]


def compute_box_resolution(
    box_size_mpc_h: float,
    particles_per_dim: int,
    params: dict[str, Any] | None = None,
    *,
    is_hydro: bool = False,
    user_softening_kpc_h: float | None = None,
) -> BoxResolutionContracts:
    """Compute physical resolution contracts for an N-body periodic box.

    Units:
      - Box size: h^-1 Mpc (comoving)
      - Particle mass: h^-1 M_sun
      - Softening: h^-1 kpc (Plummer equivalent)
    """
    if box_size_mpc_h <= 0 or not math.isfinite(box_size_mpc_h):
        raise ValueError("Box size must be positive and finite")
    if particles_per_dim < 16 or not isinstance(particles_per_dim, int):
        raise ValueError("Particles per dimension must be an integer >= 16")

    p = params or DEFAULT_PARAMS
    omega_m = float(p.get("Omega_m", 0.315))
    omega_b = float(p.get("Omega_b", 0.049))

    # Critical density at z=0: rho_crit = 3 H0^2 / (8 pi G)
    # in units of (h^-1 M_sun) / (h^-1 Mpc)^3:
    # 3 * (100 km/s/Mpc)^2 / (8 pi G) = 2.77536627e11 h^2 M_sun / Mpc^3
    # In comoving (h^-1 Mpc)^3 and h^-1 M_sun, rho_crit_comoving = 2.77536627e11
    rho_crit_comoving = 2.77536627e11  # h^-1 M_sun / (h^-1 Mpc)^3

    dm_density = (omega_m - omega_b if is_hydro else omega_m) * rho_crit_comoving
    total_particles = int(particles_per_dim**3)
    box_volume = float(box_size_mpc_h**3)
    particle_mass = (dm_density * box_volume) / total_particles

    mean_sep = float(box_size_mpc_h / particles_per_dim)
    k_fund = float(2.0 * math.pi / box_size_mpc_h)
    k_nyq = float(math.pi * particles_per_dim / box_size_mpc_h)

    # Standard softening policy: eps = mean_sep / 30 to / 25 (Power et al. 2003)
    default_softening = float(mean_sep * 1000.0 / 30.0)  # in h^-1 kpc
    softening = (
        user_softening_kpc_h if user_softening_kpc_h is not None else default_softening
    )

    # Halo mass particle thresholds
    m_20 = particle_mass * 20.0
    m_100 = particle_mass * 100.0
    m_300 = particle_mass * 300.0
    m_1000 = particle_mass * 1000.0

    # Memory and snapshot size estimations:
    # GADGET-4 PartType1: pos(3x8) + vel(3x4) + id(8) = 44 bytes bare, ~68 bytes with trees & domain
    bytes_per_particle = 120.0 if is_hydro else 68.0
    est_memory_gb = float(total_particles * bytes_per_particle * 2.2 / (1024**3))
    est_snapshot_gb = float(total_particles * 36.0 / (1024**3))

    warnings = []
    if box_size_mpc_h < 10.0:
        warnings.append(
            f"Box length L={box_size_mpc_h} h^-1 Mpc suffers from significant finite-volume effects "
            "and cannot resolve cluster-scale modes (k_fund is too large)."
        )
    if mean_sep > 2.0:
        warnings.append(
            f"Mean particle separation {mean_sep:.2f} h^-1 Mpc is very coarse. Halos below 10^13 h^-1 M_sun "
            "will be completely unresolved."
        )
    if softening < mean_sep * 1000.0 / 50.0:
        warnings.append(
            f"Softening {softening:.2f} h^-1 kpc is smaller than d/50, risking artificial two-body relaxation."
        )
    elif softening > mean_sep * 1000.0 / 10.0:
        warnings.append(
            f"Softening {softening:.2f} h^-1 kpc is larger than d/10, suppressing gravitational clustering."
        )

    return BoxResolutionContracts(
        box_size_mpc_h=float(box_size_mpc_h),
        particles_per_dimension=particles_per_dim,
        total_particles=total_particles,
        particle_mass_msun_h=float(particle_mass),
        mean_separation_mpc_h=float(mean_sep),
        k_fundamental_h_mpc=float(k_fund),
        k_nyquist_h_mpc=float(k_nyq),
        recommended_softening_kpc_h=float(softening),
        min_halo_mass_20p_msun_h=float(m_20),
        reliable_abundance_mass_100p_msun_h=float(m_100),
        well_resolved_mass_300p_msun_h=float(m_300),
        profile_resolved_mass_1000p_msun_h=float(m_1000),
        estimated_memory_gb=float(est_memory_gb),
        estimated_snapshot_gb=float(est_snapshot_gb),
        warnings=warnings,
    )


def validate_hmf_mass_range_for_box(
    res: BoxResolutionContracts,
    target_mass_msun_h: float,
) -> dict[str, Any]:
    """Check whether a requested halo mass is trustworthy at the box's particle resolution."""
    target = float(target_mass_msun_h)
    particles_in_halo = target / res.particle_mass_msun_h

    if particles_in_halo < 20:
        state = "unresolved"
        message = (
            f"Halos at {target:.2e} h^-1 M_sun contain only {particles_in_halo:.1f} particles (< 20). "
            "Halo finders cannot identify groups at this scale; abundance comparisons are meaningless."
        )
    elif particles_in_halo < 100:
        state = "completeness_warning"
        message = (
            f"Halos at {target:.2e} h^-1 M_sun contain {particles_in_halo:.0f} particles (20 - 100). "
            "Catalogue counts suffer from systematic resolution incompleteness."
        )
    elif particles_in_halo < 300:
        state = "statistically_usable"
        message = (
            f"Halos at {target:.2e} h^-1 M_sun contain {particles_in_halo:.0f} particles. "
            "Suitable for halo abundance comparisons; internal profiles remain approximate."
        )
    else:
        state = "well_resolved"
        message = (
            f"Halos at {target:.2e} h^-1 M_sun contain {particles_in_halo:.0f} particles (>= 300). "
            "High numerical precision for both mass and structural properties."
        )

    return {
        "target_mass_msun_h": target,
        "particle_count": float(particles_in_halo),
        "resolution_state": state,
        "assessment": message,
        "particle_mass_msun_h": res.particle_mass_msun_h,
    }
