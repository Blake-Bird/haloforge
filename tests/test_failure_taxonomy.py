from engine.failure_taxonomy import classify_failure


def test_solver_timeout_and_busy_failures_are_specific_and_preserve_detail():
    timeout = classify_failure(
        RuntimeError("AxiCLASS exceeded the 30-second safety timeout and was stopped")
    )
    assert timeout.code == "solver_timeout"
    assert "prior completed run" in timeout.summary
    assert "30-second" in timeout.technical_detail
    busy = classify_failure(RuntimeError("Another AxiCLASS solve is already running."))
    assert busy.code == "solver_busy"


def test_invalid_configuration_and_import_failure_have_actionable_categories():
    invalid = classify_failure(ValueError("Omega_b must be smaller than Omega_m"))
    assert invalid.code == "invalid_configuration"
    unsafe = classify_failure(ValueError("Unsafe archive path: '../run.json'"))
    assert unsafe.code == "unsafe_import"
    assert "before HaloForge wrote" in unsafe.summary


def test_unknown_failure_is_not_hidden_behind_a_false_specific_category():
    unknown = classify_failure(RuntimeError("unexpected device fault"))
    assert unknown.code == "calculation_interrupted"
    assert unknown.technical_detail == "unexpected device fault"
