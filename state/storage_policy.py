"""Storage boundary policy.

HaloForge currently supports a *single-user local* run vault.  A Streamlit
process filesystem is shared by all visitors, so treating it as a browser or
account store would leak experiments.  Public hosting is deliberately refused
until an authenticated, per-user storage adapter is supplied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class UnsafeHostedStorage(RuntimeError):
    """Raised before a shared server can write user research to one vault."""


def deployment_mode() -> str:
    """Return the explicit deployment mode, rejecting unknown values."""
    mode = os.environ.get("HALOFORGE_DEPLOYMENT", "local").strip().lower()
    if mode not in {"local", "hosted"}:
        raise ValueError("HALOFORGE_DEPLOYMENT must be either 'local' or 'hosted'.")
    return mode


def require_safe_persistent_storage() -> None:
    """Fail closed for public hosting until a per-user storage adapter exists."""
    if deployment_mode() == "hosted":
        raise UnsafeHostedStorage(
            "HaloForge's filesystem vault is single-user local storage. "
            "Unauthenticated hosted persistence is disabled because it would share runs between visitors. "
            "Deploy only after configuring an authenticated per-user storage adapter."
        )


def default_data_root() -> Path:
    """Choose an OS user-data directory, never the cloned source repository."""
    configured = os.environ.get("HALOFORGE_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "HaloForge"
    elif os.name == "nt":
        root = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
            / "HaloForge"
        )
    else:
        root = (
            Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
            / "haloforge"
        )
    return root.resolve()


def privacy_statement() -> str:
    return "This run stays in this local app-data vault unless you explicitly download or import a bundle."
