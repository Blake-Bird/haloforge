from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from state import storage_policy


ROOT = Path(__file__).resolve().parents[1]


def test_default_data_root_is_not_the_repository(monkeypatch):
    monkeypatch.delenv("HALOFORGE_DATA_DIR", raising=False)
    root = storage_policy.default_data_root()
    assert root.name.lower() == "haloforge"
    assert root != Path("data").resolve()


def test_explicit_data_root_is_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("HALOFORGE_DATA_DIR", str(tmp_path / "private-vault"))
    assert storage_policy.default_data_root() == (tmp_path / "private-vault").resolve()


def test_hosted_filesystem_persistence_fails_closed(monkeypatch):
    monkeypatch.setenv("HALOFORGE_DEPLOYMENT", "hosted")
    with pytest.raises(storage_policy.UnsafeHostedStorage):
        storage_policy.require_safe_persistent_storage()


def test_hosted_application_renders_a_fail_closed_boundary(monkeypatch, tmp_path):
    """Verify the UI itself cannot proceed into a shared filesystem vault."""
    monkeypatch.setenv("HALOFORGE_DEPLOYMENT", "hosted")
    monkeypatch.setenv("HALOFORGE_DATA_DIR", str(tmp_path / "server-vault"))

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=15)

    assert not app.exception
    assert len(app.error) == 1
    assert "Unauthenticated hosted persistence is disabled" in app.error[0].value
