import numpy as np
from engine.halo_catalogue import HaloCatalogue, HaloRecord
from engine.hmf_nbody_comparison import (
    compatible_mass_column,
    compare_catalogue_to_analytic_hmf,
    render_hmf_comparison_plot,
)


def test_compatible_mass_column():
    assert compatible_mass_column("Tinker (2008)") == "M_200m_msun_h"
    assert compatible_mass_column("Sheth-Tormen (1999)") == "M_fof_msun_h"
    assert compatible_mass_column("Watson SO (2013)") == "M_200m_msun_h"
    assert compatible_mass_column("Press-Schechter (1974)") == "M_fof_msun_h"


def test_compare_catalogue_to_analytic_hmf_handles_empty_catalogue():
    cat = HaloCatalogue(
        redshift=0.0,
        box_size_mpc_h=100.0,
        particle_mass_msun_h=1e10,
        linking_length_b=0.2,
        min_particles=20,
        halos=[],
        is_empty_due_to_resolution=True,
        summary="Empty",
    )
    mass_grid = np.logspace(11, 15, 50)
    sigma_grid = 2.0 / (mass_grid / 1e11) ** 0.2
    report = compare_catalogue_to_analytic_hmf(
        cat,
        mass_grid_h=mass_grid,
        sigma_grid=sigma_grid,
        rho0=2.775e11 * 0.315,
        h=0.6736,
        fitting="Sheth-Tormen (1999)",
    )
    assert report.total_halos == 0
    assert report.complete_halos == 0
    assert len(report.bins) == 0


def test_compare_catalogue_to_analytic_hmf_with_synthetic_catalogue():
    # Construct a synthetic catalogue with 500 halos distributed logarithmically in mass
    rng = np.random.default_rng(42)
    halo_masses = 10.0 ** rng.uniform(11.5, 14.5, size=500)
    halos = [
        HaloRecord(
            halo_id=i + 1,
            x=rng.uniform(0, 100),
            y=rng.uniform(0, 100),
            z=rng.uniform(0, 100),
            vx=0.0,
            vy=0.0,
            vz=0.0,
            n_particles=int(m / 1e10),
            M_fof_msun_h=float(m),
            M_200m_msun_h=float(m),
            M_200c_msun_h=float(m * 0.85),
            R_200m_kpc_h=500.0,
            sigma_v_km_s=250.0,
        )
        for i, m in enumerate(halo_masses)
    ]
    cat = HaloCatalogue(
        redshift=0.0,
        box_size_mpc_h=100.0,
        particle_mass_msun_h=1e10,
        linking_length_b=0.2,
        min_particles=20,
        halos=halos,
        is_empty_due_to_resolution=False,
        summary="Synthetic test catalogue",
    )

    mass_grid = np.logspace(11, 15, 50)
    sigma_grid = 2.0 / (mass_grid / 1e11) ** 0.2
    report = compare_catalogue_to_analytic_hmf(
        cat,
        mass_grid_h=mass_grid,
        sigma_grid=sigma_grid,
        rho0=2.775e11 * 0.315,
        h=0.6736,
        fitting="Sheth-Tormen (1999)",
        num_mass_bins=8,
    )

    assert report.total_halos == 500
    assert len(report.bins) == 8
    assert report.completeness_mass_msun_h == 1e12  # 100 * 1e10
    fig = render_hmf_comparison_plot(report)
    assert fig is not None
