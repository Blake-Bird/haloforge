import numpy as np
from engine.halo_catalogue import (
    catalogue_to_dataframe,
    find_fof_halos,
    render_3d_halo_view,
)


def test_find_fof_halos_empty_catalogue():
    pos = np.empty((0, 3), dtype=float)
    vel = np.empty((0, 3), dtype=float)
    cat = find_fof_halos(
        pos, vel, box_size_mpc_h=50.0, particle_mass_msun_h=1e10, redshift=20.0
    )
    assert cat.is_empty_due_to_resolution
    assert len(cat.halos) == 0
    df = catalogue_to_dataframe(cat)
    assert len(df) == 0
    fig = render_3d_halo_view(cat)
    assert fig is not None


def test_find_fof_halos_identifies_clusters_across_periodic_boundary():
    # Construct two tight clumps of 25 particles each, one centered at (25, 25, 25)
    # and one wrapping across the boundary at x=0 and x=50
    rng = np.random.default_rng(42)
    clump1 = 25.0 + rng.normal(0, 0.05, size=(25, 3))

    # Clump 2 wrapping: 13 particles at x=0.02, 12 particles at x=49.98
    clump2_left = np.column_stack(
        [
            rng.uniform(0.01, 0.04, size=13),
            rng.uniform(10.0, 10.05, size=13),
            rng.uniform(10.0, 10.05, size=13),
        ]
    )
    clump2_right = np.column_stack(
        [
            rng.uniform(49.96, 49.99, size=12),
            rng.uniform(10.0, 10.05, size=12),
            rng.uniform(10.0, 10.05, size=12),
        ]
    )
    clump2 = np.vstack([clump2_left, clump2_right])

    positions = np.vstack([clump1, clump2])
    velocities = rng.normal(0, 50, size=positions.shape)

    cat = find_fof_halos(
        positions,
        velocities,
        box_size_mpc_h=50.0,
        particle_mass_msun_h=1e10,
        redshift=0.0,
        linking_length_b=0.2,
        min_particles=20,
    )

    assert not cat.is_empty_due_to_resolution
    assert len(cat.halos) == 2
    assert cat.halos[0].n_particles == 25
    assert cat.halos[1].n_particles == 25

    df = catalogue_to_dataframe(cat)
    assert len(df) == 2
    assert "M_fof_msun_h" in df.columns
    assert "M_200m_msun_h" in df.columns

    fig = render_3d_halo_view(cat)
    assert fig is not None
