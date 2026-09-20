import json

import numpy as np
import pytest

from engine.class_runner import ClassRuntimeError, _read_worker_result
from engine.power_contract import validate_power_arrays
from state import cache
from config.defaults import DEFAULT_PARAMS


def valid_result():
    return {
        "k": np.array([0.01, 0.1]),
        "P": np.array([10.0, 2.0]),
        "P_by_z": np.array([[10.0, 2.0], [2.5, 0.5]]),
        "redshifts": np.array([0.0, 1.0]),
        "growth_class": np.array([1.0, 0.5]),
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("P_by_z", [[10.0, 2.0]]),
        ("P_by_z", [[10.0, 2.0], [np.nan, 1.0]]),
        ("P", [11.0, 2.0]),
        ("redshifts", [0.0, 0.0]),
        ("redshifts", [1.0, 2.0]),
        ("growth_class", [1.0]),
        ("growth_class", [1.0, np.inf]),
        ("growth_class", [2.0, 1.0]),
        ("background_omega_m_by_z", [0.3]),
        ("background_omega_m_by_z", [0.3, -1.0]),
    ],
)
def test_worker_rejects_inconsistent_scientific_arrays(tmp_path, key, value):
    result = valid_result()
    result[key] = np.asarray(value)
    path = tmp_path / "worker.npz"
    np.savez(path, **result, metadata=json.dumps({}))
    with pytest.raises(ClassRuntimeError):
        _read_worker_result(path)


def test_contract_preserves_closed_background_and_unavailable_background():
    result = valid_result()
    validate_power_arrays(result)
    result["background_omega_m_by_z"] = np.array([0.3, 1.05])
    validate_power_arrays(result)


def test_cache_rejects_invalid_slices_even_with_matching_checksum(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    params = dict(
        DEFAULT_PARAMS,
        k_min=0.01,
        k_max=0.1,
        k_points=2,
        z_values=[0.0, 1.0],
        single_z=0.0,
    )
    path = cache.save_cached_power(params, valid_result())
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"].item()))
    arrays["P_by_z"][1, 0] = np.nan
    envelope = metadata.pop("cache_integrity")
    envelope["content_sha256"] = cache._integrity_digest(arrays, metadata)
    metadata["cache_integrity"] = envelope
    np.savez(path, **arrays, metadata=json.dumps(metadata))
    assert cache.load_cached_power(params) is None
    assert not path.exists()


def test_contract_rejects_output_from_different_requested_grid():
    params = dict(
        DEFAULT_PARAMS,
        k_min=0.01,
        k_max=0.1,
        k_points=2,
        z_values=[0.0, 1.0],
        single_z=0.0,
    )
    validate_power_arrays(valid_result(), params)
    for changes in ({"z_values": [0.0, 2.0]}, {"k_max": 1.0}, {"k_points": 3}):
        with pytest.raises(ValueError, match="requested calculation"):
            validate_power_arrays(valid_result(), dict(params, **changes))


@pytest.mark.parametrize("dtype", [complex, str, bool])
def test_contract_rejects_arrays_that_require_lossy_type_coercion(dtype):
    result = valid_result()
    result["P_by_z"] = result["P_by_z"].astype(dtype)
    with pytest.raises(ValueError):
        validate_power_arrays(result)
