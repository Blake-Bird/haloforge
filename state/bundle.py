"""Safe, local-only import planning for HaloForge workspace archives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


MAX_BUNDLE_FILES = 2_000
MAX_BUNDLE_BYTES = 1_000_000_000
ALLOWED_ROOTS = frozenset({"saved_runs", "exports", "state"})
WORKSPACE_MANIFEST = "workspace-manifest.json"


class BundleValidationError(ValueError):
    """An archive is unsafe, malformed, or needs an explicit conflict choice."""


@dataclass(frozen=True)
class WorkspaceImportPlan:
    files: tuple[str, ...]
    collisions: tuple[str, ...]
    total_bytes: int
    integrity: str


def _safe_relative_path(name: str) -> PurePosixPath:
    if name == WORKSPACE_MANIFEST:
        return PurePosixPath(name)
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in ALLOWED_ROOTS
        or any(part in {"", "."} for part in path.parts)
    ):
        raise BundleValidationError(f"Unsafe archive path: {name!r}")
    return path


def plan_workspace_import(payload: bytes, data_root: Path) -> WorkspaceImportPlan:
    """Validate every member before a workspace archive is allowed to write."""
    try:
        with ZipFile(BytesIO(payload)) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            if len(infos) > MAX_BUNDLE_FILES:
                raise BundleValidationError(
                    f"Archive has more than {MAX_BUNDLE_FILES:,} files."
                )
            total_bytes = sum(item.file_size for item in infos)
            if total_bytes > MAX_BUNDLE_BYTES:
                raise BundleValidationError(
                    "Archive exceeds the 1 GB local import safety limit."
                )
            paths = [_safe_relative_path(item.filename) for item in infos]
            names = [path.as_posix() for path in paths]
            if len(set(names)) != len(names):
                raise BundleValidationError("Archive contains duplicate paths.")
            workspace_files = [name for name in names if name != WORKSPACE_MANIFEST]
            integrity = "legacy-unverified"
            if WORKSPACE_MANIFEST in names:
                try:
                    manifest = json.loads(archive.read(WORKSPACE_MANIFEST))
                    expected = manifest["files"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise BundleValidationError(
                        "Workspace integrity manifest is malformed."
                    ) from exc
                if set(expected) != set(workspace_files):
                    raise BundleValidationError(
                        "Workspace integrity manifest does not match the archive contents."
                    )
                for name in workspace_files:
                    if expected[name] != sha256_bytes(archive.read(name)):
                        raise BundleValidationError(
                            f"Integrity verification failed for {name}."
                        )
                integrity = "verified"
    except BadZipFile as exc:
        raise BundleValidationError(
            "The selected file is not a readable ZIP archive."
        ) from exc
    collisions = tuple(
        name for name in workspace_files if (data_root / Path(name)).exists()
    )
    return WorkspaceImportPlan(
        tuple(workspace_files), collisions, total_bytes, integrity
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_workspace(
    payload: bytes, data_root: Path, *, overwrite: bool = False
) -> WorkspaceImportPlan:
    """Import only a fully preflighted archive; never overwrite by accident."""
    plan = plan_workspace_import(payload, data_root)
    if plan.collisions and not overwrite:
        examples = ", ".join(plan.collisions[:3])
        raise BundleValidationError(
            f"Import would overwrite {len(plan.collisions)} existing file(s), including {examples}. "
            "Choose explicit replacement or import into an empty local vault."
        )
    with ZipFile(BytesIO(payload)) as archive:
        for name in plan.files:
            _atomic_write_bytes(data_root / Path(name), archive.read(name))
    return plan


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
