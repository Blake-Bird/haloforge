from engine.uncertainty import uncertainty_inventory


def test_uncertainty_inventory_keeps_sources_separate():
    run = {
        "params": {"k_points": 100, "mass_points": 50},
        "sigma_result": {
            "numerical_diagnostics": {
                "low_k_truncation": {"status": "low_sensitivity"},
                "high_k_truncation": {"status": "low_sensitivity"},
            }
        },
    }
    rows = uncertainty_inventory(
        run, {"calibrated_mask": [True, True]}, {"rows": [{"status": "pass"}]}
    )
    by_source = {row["source"]: row for row in rows}
    assert by_source["Numerical integration"]["state"] == "measured"
    assert by_source["Theory/model uncertainty"]["state"] == "unquantified"
    assert by_source["Emulator uncertainty"]["state"] == "not applicable"
