from config.defaults import DEFAULT_PARAMS
from engine.gadget4_adapter import (
    GADGET4_PINNED_COMMIT,
    generate_config_sh,
    generate_gadget4_parameter_file,
    generate_tabulated_expansion_history,
    run_installation_doctor,
)


def test_installation_doctor_returns_valid_report():
    report = run_installation_doctor()
    assert report.cpu_cores >= 1
    assert report.ram_gb > 0
    assert report.free_disk_gb > 0
    assert report.recommended_runtime
    assert isinstance(report.ready_for_simulation, bool)
    assert isinstance(report.recommendations, list)


def test_generate_config_sh_includes_pinned_revision_and_flags():
    config_dm = generate_config_sh(
        is_hydro=False, enable_2lpt=True, enable_fof=True, enable_subfind=True
    )
    assert GADGET4_PINNED_COMMIT in config_dm
    assert "PERIODIC" in config_dm
    assert "SECOND_ORDER_LPT_ICS" in config_dm
    assert "FOF" in config_dm
    assert "SUBFIND" in config_dm
    assert "COOLING" not in config_dm

    config_hydro = generate_config_sh(is_hydro=True, enable_2lpt=True)
    assert "PRESSURE_ENTROPY_SPH" in config_hydro
    assert "COOLING" in config_hydro
    assert "STARFORMATION" in config_hydro


def test_generate_gadget4_parameter_file():
    param_txt = generate_gadget4_parameter_file(
        box_size_mpc_h=100.0,
        particles_per_dim=64,
        output_dir="output",
        output_redshifts=[10.0, 5.0, 2.0, 1.0, 0.0],
        params=DEFAULT_PARAMS,
        start_redshift=49.0,
    )
    assert "BoxSize                   100.0" in param_txt
    assert "TimeBegin                 0.020000" in param_txt  # 1 / (1 + 49) = 0.02
    assert "TimeMax                   1.000000" in param_txt
    assert "HubbleParam" in param_txt
    assert "GroupLinkLength           0.2" in param_txt


def test_generate_tabulated_expansion_history_ede():
    params_ede = {
        **DEFAULT_PARAMS,
        "enable_ede": True,
        "f_EDE": 0.12,
        "log10_a_c": -3.5,
    }
    table = generate_tabulated_expansion_history(params_ede, num_points=20)
    lines = [
        line.strip() for line in table.strip().split("\n") if not line.startswith("#")
    ]
    assert len(lines) == 20
    first_a, first_h = map(float, lines[0].split())
    last_a, last_h = map(float, lines[-1].split())
    assert first_a < 1e-3
    assert abs(last_a - 1.0) < 1e-5
    # H(a=1)/H0 should be ~1.0 for flat universe
    assert abs(last_h - 1.0) < 0.05
