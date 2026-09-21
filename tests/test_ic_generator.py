import numpy as np
from config.defaults import DEFAULT_PARAMS
from engine.ic_generator import generate_2lpt_particles, generate_gaussian_random_field


def test_gaussian_random_field_dimensions_and_variance():
    grid_size = 16
    box_size = 50.0
    k_eval = np.logspace(-2, 1, 100)
    # Simple power-law test power spectrum
    p_eval = 1000.0 * (k_eval / 0.1) ** (-1.5)

    delta_k = generate_gaussian_random_field(
        grid_size, box_size, k_eval, p_eval, seed=123
    )
    # Fourier half-complex layout: (N, N, N // 2 + 1)
    assert delta_k.shape == (grid_size, grid_size, grid_size // 2 + 1)
    assert delta_k[0, 0, 0] == 0.0


def test_generate_2lpt_particles_preserves_periodicity_and_momentum():
    grid_size = 16
    box_size = 50.0
    k_eval = np.logspace(-2, 1, 100)
    p_eval = 2000.0 * (k_eval / 0.1) ** (-1.2)

    positions, velocities, ids, report = generate_2lpt_particles(
        box_size_mpc_h=box_size,
        particles_per_dim=grid_size,
        k_power=k_eval,
        p_power=p_eval,
        start_redshift=49.0,
        params=DEFAULT_PARAMS,
        seed=42,
    )

    assert len(positions) == grid_size**3
    assert len(velocities) == grid_size**3
    assert len(ids) == grid_size**3
    assert report.periodicity_passed
    assert (
        report.center_of_mass_velocity_km_s < 1.0
    )  # Center of mass momentum conserved
    assert report.checks_passed
    assert np.all((positions >= 0.0) & (positions < box_size))
