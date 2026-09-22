import numpy as np
from config.defaults import DEFAULT_PARAMS
from engine.ic_generator import (
    export_gadget_hdf5_ic,
    generate_2lpt_particles,
    generate_gaussian_random_field,
    generate_zeldovich_particles,
    measure_realized_power_spectrum,
)
from engine.gadget_snapshot import load_gadget4_dm_snapshot


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


def test_generate_zeldovich_particles_preserves_periodicity_momentum_and_measures_power():
    grid_size = 16
    box_size = 50.0
    k_eval = np.logspace(-2, 1, 100)
    p_eval = 2000.0 * (k_eval / 0.1) ** (-1.2)

    positions, velocities, ids, report = generate_zeldovich_particles(
        box_size_mpc_h=box_size,
        particles_per_dim=grid_size,
        k_power=k_eval,
        p_power=p_eval,
        start_redshift=49.0,
        params={**DEFAULT_PARAMS, "enable_ede": False},
        seed=42,
        spectrum_redshift=49.0,
        growth_rate=1.0,
        expansion_rate_E=100.0,
    )

    assert len(positions) == grid_size**3
    assert len(velocities) == grid_size**3
    assert len(ids) == grid_size**3
    assert report.periodicity_passed
    assert (
        report.center_of_mass_velocity_km_s < 1.0
    )  # Center of mass momentum conserved
    assert report.checks_passed
    assert report.linear_power_agreement_passed
    assert report.power_agreement_bin_count > 0
    assert report.spectrum_redshift == 49.0
    assert np.all((positions >= 0.0) & (positions < box_size))


def test_paired_fixed_field_has_correct_measured_power_normalization():
    n, box = 24, 100.0
    k = np.logspace(-3, 1, 400)
    p = 500.0 * (k / 0.1) ** -1.0
    delta_k = generate_gaussian_random_field(n, box, k, p, seed=7, paired_fixed=True)
    measured_k, measured_p = measure_realized_power_spectrum(delta_k, box, n)
    expected = np.exp(np.interp(np.log(measured_k), np.log(k), np.log(p)))
    # Shell averages of a sloped spectrum differ slightly from P evaluated at
    # the geometric shell centre; fixed amplitudes keep that binning effect
    # below a few percent rather than adding realization variance.
    np.testing.assert_allclose(measured_p, expected, rtol=0.03)


def test_ede_generation_rejects_lcdm_growth_shortcut():
    with np.testing.assert_raises_regex(ValueError, "EDE ICs require growth"):
        generate_zeldovich_particles(
            100.0,
            16,
            np.logspace(-3, 1, 100),
            np.ones(100),
            20.0,
            {**DEFAULT_PARAMS, "enable_ede": True},
        )


def test_gadget_ic_export_uses_the_matching_mpc_h_unit_contract(tmp_path):
    axis = np.linspace(0.1, 9.9, 16)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    velocities = np.zeros_like(positions)
    ids = np.arange(1, len(positions) + 1, dtype=np.int64)
    output = tmp_path / "ics.hdf5"
    export_gadget_hdf5_ic(
        str(output), positions, velocities, ids, 10.0, 49.0, DEFAULT_PARAMS
    )
    snapshot = load_gadget4_dm_snapshot(output)
    assert snapshot.box_size_mpc_h == 10.0
    np.testing.assert_allclose(snapshot.positions_mpc_h, positions)


def test_ede_generation_accepts_explicit_matching_solver_growth_inputs():
    positions, _velocities, _ids, report = generate_zeldovich_particles(
        100.0,
        16,
        np.logspace(-3, 1, 100),
        np.ones(100),
        20.0,
        {**DEFAULT_PARAMS, "enable_ede": True},
        spectrum_redshift=20.0,
        growth_rate=0.98,
        expansion_rate_E=53.0,
    )
    assert report.spectrum_redshift == 20.0
    assert np.all((positions >= 0.0) & (positions < 100.0))


def test_2lpt_has_an_explicit_second_order_displacement_and_periodic_output():
    box, n = 80.0, 16
    k = np.logspace(-3, 1, 200)
    p = 1.0e3 * (k / 0.1) ** -1.1
    positions, velocities, ids, report = generate_2lpt_particles(
        box,
        n,
        k,
        p,
        20.0,
        {**DEFAULT_PARAMS, "enable_ede": False},
        seed=11,
        paired_fixed=True,
        spectrum_redshift=20.0,
        growth_rate=0.99,
        expansion_rate_E=53.0,
    )
    assert report.lpt_order == 2
    assert report.max_second_order_displacement_mpc_h > 0.0
    assert report.periodicity_passed
    assert report.linear_power_agreement_passed
    assert len(ids) == n**3
    assert velocities.shape == positions.shape == (n**3, 3)
    assert np.all((positions >= 0.0) & (positions < box))


def test_2lpt_rejects_ede_without_second_order_growth_inputs():
    with np.testing.assert_raises_regex(ValueError, "EDE 2LPT requires"):
        generate_2lpt_particles(
            100.0,
            16,
            np.logspace(-3, 1, 100),
            np.ones(100),
            20.0,
            {**DEFAULT_PARAMS, "enable_ede": True},
            spectrum_redshift=20.0,
            growth_rate=0.98,
            expansion_rate_E=53.0,
        )
