from copy import deepcopy

from config.defaults import DEFAULT_PARAMS
from state.provenance import (
    APP_VERSION,
    reproducibility_hash,
    run_provenance,
    software_provenance,
)


def test_software_provenance_has_required_release_fields():
    value = software_provenance()
    assert value["app_version"] == APP_VERSION
    assert value["schema_version"]
    assert value["requirements_sha256"]
    assert {"streamlit", "numpy", "scipy"} <= value["package_versions"].keys()


def test_reproducibility_hash_is_stable_and_input_sensitive():
    provenance = {
        "schema_version": "v",
        "app_version": "a",
        "git_revision": "g",
        "axiclass_commit": "c",
    }
    params = deepcopy(DEFAULT_PARAMS)
    first = reproducibility_hash(params, {"output": "mPk"}, provenance)
    assert first == reproducibility_hash(
        deepcopy(params), {"output": "mPk"}, provenance
    )
    params["A_s"] *= 1.01
    assert first != reproducibility_hash(params, {"output": "mPk"}, provenance)


def test_run_provenance_has_hash_that_is_distinct_from_timestamp():
    value = run_provenance(DEFAULT_PARAMS, {"output": "mPk"})
    assert value["created_at"]
    assert len(value["reproducibility_hash"]) == 64
