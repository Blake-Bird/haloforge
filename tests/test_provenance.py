from copy import deepcopy

from config.defaults import DEFAULT_PARAMS
from state.provenance import (
    APP_VERSION,
    reproducibility_hash,
    run_provenance,
    software_provenance,
    source_tree_sha256,
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
    assert "solver_binding" in value
    assert "worker_python_sha256" in value["solver_binding"]
    assert value["created_at"]
    assert len(value["reproducibility_hash"]) == 64


def test_source_identity_tracks_uncommitted_code_but_excludes_user_data(tmp_path):
    source = tmp_path / "engine"
    source.mkdir()
    module = source / "sigma.py"
    module.write_text("value = 1\n")
    first = source_tree_sha256(tmp_path)
    private = tmp_path / "data"
    private.mkdir()
    (private / "notes.json").write_text('{"private": "experiment"}')
    assert source_tree_sha256(tmp_path) == first
    module.write_text("value = 2\n")
    assert source_tree_sha256(tmp_path) != first


def test_v3_identity_includes_actual_source_and_environment():
    provenance = software_provenance()
    first = reproducibility_hash(DEFAULT_PARAMS, {}, provenance)
    for key, changed in {
        "source_tree_sha256": "different source",
        "requirements_sha256": "different dependencies",
        "package_versions": {"numpy": "different version"},
        "python": "different interpreter",
    }.items():
        assert (
            reproducibility_hash(DEFAULT_PARAMS, {}, {**provenance, key: changed})
            != first
        )
    assert (
        reproducibility_hash(DEFAULT_PARAMS, {}, {**provenance, "created_at": "later"})
        == first
    )


def test_v2_identity_remains_compatible_with_its_original_fields():
    provenance = dict(
        schema_version="haloforge-run-v2",
        app_version="a",
        git_revision="g",
        axiclass_commit="c",
    )
    first = reproducibility_hash(DEFAULT_PARAMS, {}, provenance)
    assert (
        reproducibility_hash(
            DEFAULT_PARAMS,
            {},
            {**provenance, "package_versions": {"numpy": "ignored by v2"}},
        )
        == first
    )
