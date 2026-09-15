from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from state.bundle import (
    BundleValidationError,
    WORKSPACE_MANIFEST,
    import_workspace,
    plan_workspace_import,
    sha256_bytes,
)


def _zip(entries: dict[str, bytes]) -> bytes:
    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return out.getvalue()


def test_workspace_import_rejects_traversal_before_writing(tmp_path):
    payload = _zip({"saved_runs/ok.json": b"ok", "../escape": b"no"})
    with pytest.raises(BundleValidationError, match="Unsafe archive path"):
        plan_workspace_import(payload, tmp_path)
    assert not (tmp_path / "saved_runs" / "ok.json").exists()


def test_workspace_import_requires_explicit_overwrite(tmp_path):
    target = tmp_path / "saved_runs" / "run.json"
    target.parent.mkdir()
    target.write_bytes(b"old")
    payload = _zip({"saved_runs/run.json": b"new", "state/draft_params.json": b"{}"})
    plan = plan_workspace_import(payload, tmp_path)
    assert plan.collisions == ("saved_runs/run.json",)
    with pytest.raises(BundleValidationError, match="overwrite"):
        import_workspace(payload, tmp_path)
    assert target.read_bytes() == b"old"
    result = import_workspace(payload, tmp_path, overwrite=True)
    assert len(result.files) == 2
    assert target.read_bytes() == b"new"


def test_workspace_manifest_is_verified_before_import(tmp_path):
    body = b"trusted"
    payload = _zip(
        {
            "saved_runs/run.json": body,
            WORKSPACE_MANIFEST: (
                '{"files":{"saved_runs/run.json":"' + sha256_bytes(body) + '"}}'
            ).encode(),
        }
    )
    assert plan_workspace_import(payload, tmp_path).integrity == "verified"
    tampered = _zip(
        {
            "saved_runs/run.json": b"tampered",
            WORKSPACE_MANIFEST: (
                '{"files":{"saved_runs/run.json":"' + sha256_bytes(body) + '"}}'
            ).encode(),
        }
    )
    with pytest.raises(BundleValidationError, match="Integrity verification failed"):
        plan_workspace_import(tampered, tmp_path)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.json",
        "/absolute.json",
        "saved_runs/../state/escape.json",
        "saved_runs\\windows-separator.json",
        "state/line\nbreak.json",
        "unknown_root/run.json",
    ],
)
def test_workspace_import_fuzz_rejects_unsafe_member_names_before_any_write(
    tmp_path, unsafe_name
):
    payload = _zip({"saved_runs/valid.json": b"safe", unsafe_name: b"unsafe"})
    with pytest.raises(BundleValidationError, match="Unsafe archive path"):
        import_workspace(payload, tmp_path, overwrite=True)
    assert not (tmp_path / "saved_runs" / "valid.json").exists()
