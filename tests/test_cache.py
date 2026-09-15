import numpy as np

from config.defaults import DEFAULT_PARAMS
from state import cache


def test_cache_key_stable():
    assert cache.cache_key(DEFAULT_PARAMS) == cache.cache_key(dict(DEFAULT_PARAMS))


def test_cache_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    result = {
        "k": np.array([0.01, 0.1]),
        "P": np.array([10.0, 2.0]),
        "derived": {"h": 0.6781, "Omega_m": 0.31, "sigma8": None},
        "class_status": "AXICLASS",
        "warning": "test",
        "class_error": "test",
    }
    cache.save_cached_power(DEFAULT_PARAMS, result)
    loaded = cache.load_cached_power(DEFAULT_PARAMS)
    assert loaded is not None
    assert loaded["from_cache"] is True
    assert np.allclose(loaded["k"], result["k"])
    assert np.allclose(loaded["P"], result["P"])
    assert loaded["class_status"] == "AXICLASS"


def test_cache_checksum_mismatch_is_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    result = {
        "k": np.array([0.01, 0.1]),
        "P": np.array([10.0, 2.0]),
        "class_status": "AXICLASS",
        "derived": {"h": 0.6781},
    }
    path = cache.save_cached_power(DEFAULT_PARAMS, result)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = data["metadata"]
    arrays["P"] = np.array([11.0, 2.0])
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays, metadata=metadata)
    assert cache.load_cached_power(DEFAULT_PARAMS) is None
    assert not path.exists()


def test_cache_refuses_nonphysical_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    with np.testing.assert_raises_regex(ValueError, "invalid matter power"):
        cache.save_cached_power(
            DEFAULT_PARAMS, {"k": np.array([0.1, 0.01]), "P": np.array([1.0, -1.0])}
        )
