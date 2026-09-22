import json

import numpy as np
import pytest

from config.defaults import DEFAULT_PARAMS
from engine.gadget4_adapter import (
    GADGET4_PINNED_COMMIT,
    generate_config_sh,
    generate_gadget4_parameter_file,
    generate_output_times_file,
    generate_tabulated_expansion_history,
    run_installation_doctor,
    validate_acceptance_manifest,
)


def test_installation_doctor_returns_valid_report():
    report = run_installation_doctor()
    assert report.cpu_cores >= 1
    assert report.ram_gb > 0
    assert report.free_disk_gb > 0
    assert report.recommended_runtime
    assert isinstance(report.ready_for_simulation, bool)
    assert isinstance(report.recommendations, list)
    assert isinstance(report.acceptance_evidence_present, bool)


def test_acceptance_manifest_requires_matching_runtime_and_complete_evidence(tmp_path):
    image = "example.invalid/gadget@sha256:abc123"
    manifest = tmp_path / "acceptance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "haloforge-gadget4-acceptance-v1",
                "status": "passed",
                "gadget4_pinned_commit": GADGET4_PINNED_COMMIT,
                "image_digest": image,
                "fixture_id": "official-lcdm-smoke",
                "snapshot_sha256": "a" * 64,
                "completed_at": "2026-09-21T00:00:00Z",
            }
        )
    )
    assert validate_acceptance_manifest(manifest, image) == (
        True,
        "Pinned-image GADGET-4 acceptance evidence is recorded.",
    )
    assert validate_acceptance_manifest(manifest, image + "different")[0] is False


def test_generate_config_sh_includes_pinned_revision_and_flags():
    config_dm = generate_config_sh(
        is_hydro=False, enable_2lpt=True, enable_fof=True, enable_subfind=True
    )
    assert GADGET4_PINNED_COMMIT in config_dm
    assert "PERIODIC" in config_dm
    assert "NSOFTCLASSES=1" in config_dm
    assert "NTYPES=2" in config_dm
    assert "POWERSPEC_ON_OUTPUT" in config_dm
    assert "POWERSPEC_ON_THE_FLY" not in config_dm
    assert "SECOND_ORDER_LPT_ICS" in config_dm
    assert "FOF" in config_dm
    assert "FOF_PRIMARY_LINK_TYPES=2" in config_dm
    assert "FOF_SECONDARY_LINK_TYPES=0" in config_dm
    assert "FOF_GROUP_MIN_LEN=20" in config_dm
    assert "FOF_LINKLENGTH=0.2" in config_dm
    assert "SUBFIND" in config_dm
    assert "SUBFIND_STORE_PARTICLE_PROPERTIES" not in config_dm
    assert "COOLING" not in config_dm

    with pytest.raises(ValueError, match="collisionless DM-only"):
        generate_config_sh(is_hydro=True, enable_2lpt=True)


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
    assert "ICFormat                  3" in param_txt
    assert "MaxMemSize                1024" in param_txt
    assert "DesLinkNgb                 20" in param_txt
    assert "UnitLength_in_cm         3.085678e24" in param_txt
    assert "UnitMass_in_g            1.989e43" in param_txt
    assert "UnitVelocity_in_cm_per_s 1.0e5" in param_txt
    assert "SofteningComovingClass0" in param_txt
    assert "SofteningMaxPhysClass0" in param_txt
    assert "SofteningComovingType" not in param_txt
    assert "GroupLinkLength" not in param_txt


def test_hdf5_initial_conditions_use_gadget_filename_stem():
    param_txt = generate_gadget4_parameter_file(
        10.0, 16, "output", [0.0], DEFAULT_PARAMS, ic_filename="ics.hdf5"
    )
    assert "InitCondFile              ics\n" in param_txt


def test_output_times_are_plain_scale_factors_and_validate_start():
    output = generate_output_times_file([10.0, 2.0, 0.0], start_redshift=49.0)
    assert output.splitlines() == ["0.0909090909091", "0.333333333333", "1"]
    with np.testing.assert_raises_regex(ValueError, "precedes"):
        generate_output_times_file([50.0], start_redshift=49.0)


def test_generate_tabulated_expansion_history_ede():
    params_ede = {
        **DEFAULT_PARAMS,
        "enable_ede": True,
        "f_EDE": 0.12,
        "log10_a_c": -3.5,
    }
    with np.testing.assert_raises_regex(RuntimeError, "were removed"):
        generate_tabulated_expansion_history(params_ede, num_points=20)


def test_gadget_parameter_file_rejects_invented_ede_background_setting():
    with np.testing.assert_raises_regex(ValueError, "no validated generic EDE"):
        generate_gadget4_parameter_file(
            100.0,
            64,
            "output",
            [0.0],
            DEFAULT_PARAMS,
            tabulated_expansion_file="ExpansionHistory.txt",
        )
