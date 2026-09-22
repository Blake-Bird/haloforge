from config.defaults import DEFAULT_PARAMS
from engine.nbody_setup import compute_box_resolution, validate_hmf_mass_range_for_box
import pytest


def test_box_resolution_computes_exact_properties():
    res = compute_box_resolution(
        box_size_mpc_h=100.0, particles_per_dim=64, params=DEFAULT_PARAMS
    )
    assert res.box_size_mpc_h == 100.0
    assert res.particles_per_dimension == 64
    assert res.total_particles == 64**3
    # Mean separation = 100 / 64 = 1.5625 h^-1 Mpc
    assert abs(res.mean_separation_mpc_h - 1.5625) < 1e-4
    # Particle mass should be order of 10^11 h^-1 M_sun
    assert 1e11 < res.particle_mass_msun_h < 1e12
    # Thresholds are 20, 100, 300, 1000 times particle mass
    assert abs(res.min_halo_mass_20p_msun_h - 20.0 * res.particle_mass_msun_h) < 1e-3
    assert (
        abs(res.reliable_abundance_mass_100p_msun_h - 100.0 * res.particle_mass_msun_h)
        < 1e-3
    )
    assert (
        abs(res.well_resolved_mass_300p_msun_h - 300.0 * res.particle_mass_msun_h)
        < 1e-3
    )
    assert (
        abs(res.profile_resolved_mass_1000p_msun_h - 1000.0 * res.particle_mass_msun_h)
        < 1e-3
    )


def test_box_resolution_validates_inputs():
    with pytest.raises(ValueError, match="Box size must be positive"):
        compute_box_resolution(box_size_mpc_h=-10.0, particles_per_dim=64)
    with pytest.raises(ValueError, match="Particles per dimension"):
        compute_box_resolution(box_size_mpc_h=100.0, particles_per_dim=8)
    with pytest.raises(ValueError, match="collisionless DM-only"):
        compute_box_resolution(
            box_size_mpc_h=100.0, particles_per_dim=64, is_hydro=True
        )


def test_validate_hmf_mass_range_for_box_categorizes_correctly():
    res = compute_box_resolution(box_size_mpc_h=50.0, particles_per_dim=64)
    # Mass with < 20 particles
    under = validate_hmf_mass_range_for_box(
        res, target_mass_msun_h=res.particle_mass_msun_h * 10
    )
    assert under["resolution_state"] == "unresolved"

    # Mass with 20-100 particles
    low = validate_hmf_mass_range_for_box(
        res, target_mass_msun_h=res.particle_mass_msun_h * 50
    )
    assert low["resolution_state"] == "completeness_warning"

    # Mass with 100-300 particles
    med = validate_hmf_mass_range_for_box(
        res, target_mass_msun_h=res.particle_mass_msun_h * 200
    )
    assert med["resolution_state"] == "statistically_usable"

    # Mass with >= 300 particles
    high = validate_hmf_mass_range_for_box(
        res, target_mass_msun_h=res.particle_mass_msun_h * 500
    )
    assert high["resolution_state"] == "well_resolved"
