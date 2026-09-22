"""Native GADGET-4 FoF/Subfind HDF5 catalogue reader for DM-only runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


MPC_IN_CM = 3.085678e24
TEN_BILLION_SOLAR_MASS_IN_G = 1.989e43


@dataclass(frozen=True)
class GadgetGroupCatalogue:
    source_paths: tuple[str, ...]
    sha256_by_file: tuple[str, ...]
    group_fields: dict[str, np.ndarray]
    subhalo_fields: dict[str, np.ndarray]
    box_size_mpc_h: float
    redshift: float
    scale_factor: float
    omega_m: float
    h: float
    gadget_commit: str

    @property
    def group_mass_msun_h(self) -> np.ndarray:
        """FoF linking-length masses; never relabel these as SO masses."""
        return self.group_fields["GroupMass"] * 1.0e10


def _files_for_catalogue(path: Path) -> tuple[Path, ...]:
    match = re.fullmatch(r"(.+)\.(\d+)\.hdf5", path.name)
    if not match:
        if not path.is_file():
            raise ValueError("GADGET group catalogue does not exist")
        with h5py.File(path) as handle:
            count = int(handle["Header"].attrs.get("NumFiles", 1))
        if count != 1:
            raise ValueError(
                "Multi-file group catalogue requires a numbered shard path"
            )
        return (path,)
    base = path.with_name(match.group(1))
    first = base.with_name(base.name + ".0.hdf5")
    if not first.is_file():
        raise ValueError("Group catalogue is missing shard 0")
    with h5py.File(first) as handle:
        count = int(handle["Header"].attrs.get("NumFiles", 0))
    if count < 2 or count > 4096 or int(match.group(2)) >= count:
        raise ValueError("Group catalogue has an invalid shard count")
    files = tuple(base.with_name(f"{base.name}.{i}.hdf5") for i in range(count))
    if any(not item.is_file() for item in files):
        raise ValueError("Group catalogue is missing one or more shards")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gadget4_group_catalogue(path: str | Path) -> GadgetGroupCatalogue:
    """Read host groups and subhalos separately, checking all native shards."""
    files = _files_for_catalogue(Path(path))
    group_parts: dict[str, list[np.ndarray]] = {}
    sub_parts: dict[str, list[np.ndarray]] = {}
    reference: tuple | None = None
    totals: tuple[int, int] | None = None
    seen_groups = seen_subhalos = 0
    for item in files:
        with h5py.File(item) as handle:
            if (
                "Header" not in handle
                or "Group" not in handle
                or "Parameters" not in handle
            ):
                raise ValueError(
                    "Native group catalogue needs Header, Group, and Parameters"
                )
            hdr, params = handle["Header"].attrs, handle["Parameters"].attrs
            for key in ("GroupMass", "GroupPos", "GroupLen"):
                if key not in handle["Group"]:
                    raise ValueError(f"Group catalogue is missing {key}")
            if not np.isclose(float(params["UnitLength_in_cm"]), MPC_IN_CM, rtol=1e-6):
                raise ValueError("Group catalogue length unit is not Mpc/h")
            if not np.isclose(
                float(params["UnitMass_in_g"]), TEN_BILLION_SOLAR_MASS_IN_G, rtol=1e-6
            ):
                raise ValueError("Group catalogue mass unit is not 1e10 Msun/h")
            commit = hdr.get("Git_commit", b"")
            if isinstance(commit, bytes):
                commit = commit.decode("ascii", errors="replace")
            facts = (
                float(hdr["BoxSize"]),
                float(hdr["Redshift"]),
                float(hdr["Time"]),
                float(params["Omega0"]),
                float(params["HubbleParam"]),
                str(commit),
                int(hdr["NumFiles"]),
            )
            if reference is None:
                reference = facts
                totals = (int(hdr["Ngroups_Total"]), int(hdr["Nsubhalos_Total"]))
            elif facts != reference or totals != (
                int(hdr["Ngroups_Total"]),
                int(hdr["Nsubhalos_Total"]),
            ):
                raise ValueError(
                    "Group catalogue shards disagree on run metadata or totals"
                )
            n_groups = int(hdr["Ngroups_ThisFile"])
            n_subhalos = int(hdr["Nsubhalos_ThisFile"])
            seen_groups += n_groups
            seen_subhalos += n_subhalos
            for name, dataset in handle["Group"].items():
                if dataset.shape[0] != n_groups:
                    raise ValueError(f"Group field {name} length disagrees with Header")
                group_parts.setdefault(name, []).append(np.asarray(dataset))
            if "Subhalo" in handle:
                for name, dataset in handle["Subhalo"].items():
                    if dataset.shape[0] != n_subhalos:
                        raise ValueError(
                            f"Subhalo field {name} length disagrees with Header"
                        )
                    sub_parts.setdefault(name, []).append(np.asarray(dataset))
            elif n_subhalos:
                raise ValueError("Catalogue declares subhalos but has no Subhalo group")
    assert reference is not None and totals is not None
    if (seen_groups, seen_subhalos) != totals:
        raise ValueError("Group catalogue shard counts disagree with totals")
    groups = {key: np.concatenate(parts) for key, parts in group_parts.items()}
    subhalos = {key: np.concatenate(parts) for key, parts in sub_parts.items()}
    if groups["GroupPos"].shape != (seen_groups, 3):
        raise ValueError("GroupPos must contain three coordinates per host")
    if np.any(~np.isfinite(groups["GroupMass"])) or np.any(groups["GroupMass"] <= 0):
        raise ValueError("GroupMass contains invalid values")
    if np.any(groups["GroupLen"] <= 0):
        raise ValueError("GroupLen contains invalid values")
    box, redshift, scale_factor, omega_m, h, commit, _ = reference
    if not np.isclose(redshift, 1.0 / scale_factor - 1.0, atol=1e-5):
        raise ValueError("Group catalogue Time and Redshift disagree")
    if (
        np.any(~np.isfinite(groups["GroupPos"]))
        or np.any(groups["GroupPos"] < 0)
        or np.any(groups["GroupPos"] >= box)
    ):
        raise ValueError("Group positions fall outside the periodic box")
    return GadgetGroupCatalogue(
        source_paths=tuple(str(item.resolve()) for item in files),
        sha256_by_file=tuple(_sha256(item) for item in files),
        group_fields=groups,
        subhalo_fields=subhalos,
        box_size_mpc_h=box,
        redshift=redshift,
        scale_factor=scale_factor,
        omega_m=omega_m,
        h=h,
        gadget_commit=commit,
    )
