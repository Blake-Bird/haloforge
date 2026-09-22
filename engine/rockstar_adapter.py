"""Rockstar HDF5 adapter for validated single-file GADGET-4 DM snapshots."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from engine.gadget_snapshot import GadgetSnapshotParticles


ROCKSTAR_PINNED_COMMIT = "99d56672092e88dbed446f87f6eed87c48ff0e77"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RockstarInput:
    path: str
    source_sha256: str
    converted_sha256: str
    scale_factor: float
    box_size_mpc_h: float
    particle_mass_msun_h: float
    omega_m: float
    h: float


@dataclass(frozen=True)
class RockstarCatalogue:
    path: str
    sha256: str
    config_sha256: str
    columns: dict[str, np.ndarray]
    scale_factor: float
    box_size_mpc_h: float
    omega_m: float
    h: float
    strict_so_masses: bool
    periodic: bool

    @property
    def count(self) -> int:
        return len(self.columns["id"])


def prepare_rockstar_hdf5_snapshot(
    snapshot: GadgetSnapshotParticles, target: str | Path
) -> RockstarInput:
    """Write a six-type AREPO-reader copy with cosmology restored from GADGET.

    Rockstar's HDF5 reader requires six header slots and cosmological Header
    attributes. GADGET-4 with NTYPES=2 writes two slots and puts cosmology in
    Parameters. Particle arrays are copied without unit changes; the Rockstar
    config must use length factor 1 and mass factor 1e10.
    """
    if len(snapshot.source_paths) != 1:
        raise ValueError(
            "Rockstar single-worker adapter needs a complete single-file snapshot"
        )
    source = Path(snapshot.source_path)
    target = Path(target)
    if target.exists():
        raise ValueError("Rockstar input target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with h5py.File(source) as original, h5py.File(temporary, "w") as converted:
            if (
                "Header" not in original
                or "Parameters" not in original
                or "PartType1" not in original
            ):
                raise ValueError(
                    "GADGET snapshot lacks Header, Parameters, or PartType1"
                )
            hdr, params = original["Header"].attrs, original["Parameters"].attrs
            for name, expected in (
                ("Omega0", snapshot.omega_m),
                ("HubbleParam", snapshot.h),
                ("BoxSize", snapshot.box_size_mpc_h),
            ):
                actual = float(params[name] if name != "BoxSize" else hdr[name])
                if not np.isclose(actual, expected, rtol=0, atol=1e-5):
                    raise ValueError(f"GADGET {name} disagrees with the bound snapshot")
            omega_l = float(params["OmegaLambda"])
            if not np.isclose(omega_l, 1.0 - snapshot.omega_m, atol=1e-5):
                raise ValueError("Rockstar adapter supports flat LCDM only")
            original.copy("PartType1", converted)
            output_header = converted.create_group("Header")
            count_fields = {
                "MassTable",
                "NumPart_ThisFile",
                "NumPart_Total",
                "NumPart_Total_HighWord",
            }
            for name, value in hdr.items():
                if name not in count_fields:
                    output_header.attrs[name] = value
            for name in count_fields:
                source_array = (
                    np.asarray(hdr[name])
                    if name in hdr
                    else np.zeros(2, dtype=np.uint32)
                )
                if source_array.ndim != 1 or source_array.size not in (2, 6):
                    raise ValueError(
                        f"GADGET {name} has unsupported particle-type slots"
                    )
                padded = np.zeros(6, dtype=source_array.dtype)
                padded[: source_array.size] = source_array
                output_header.attrs[name] = padded
            output_header.attrs["Omega0"] = snapshot.omega_m
            output_header.attrs["OmegaLambda"] = omega_l
            output_header.attrs["HubbleParam"] = snapshot.h
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return RockstarInput(
        path=str(target.resolve()),
        source_sha256=_sha256(source),
        converted_sha256=_sha256(target),
        scale_factor=snapshot.scale_factor,
        box_size_mpc_h=snapshot.box_size_mpc_h,
        particle_mass_msun_h=snapshot.particle_mass_msun_h,
        omega_m=snapshot.omega_m,
        h=snapshot.h,
    )


def run_rockstar_single_snapshot(
    converted: RockstarInput, output_dir: str | Path, *, force_res_mpc_h: float
) -> Path:
    """Run the HDF5 AREPO reader; require a nonempty native ASCII output.

    This is the documented single-worker path. Its boundary treatment must be
    reviewed before a periodic FoF-versus-Rockstar scientific comparison.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if not np.isfinite(force_res_mpc_h) or force_res_mpc_h <= 0:
        raise ValueError("Rockstar force resolution must be positive")
    config = output / "input.cfg"
    config.write_text(
        "\n".join(
            (
                'FILE_FORMAT = "AREPO"',
                "AREPO_LENGTH_CONVERSION = 1",
                "AREPO_MASS_CONVERSION = 1e10",
                "AREPO_DM_PARTTYPE = 1",
                f"FORCE_RES = {force_res_mpc_h:.12g}",
                "MIN_HALO_OUTPUT_SIZE = 20",
                "STRICT_SO_MASSES = 1",
                f'OUTBASE = "{output.resolve()}"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with (output / "rockstar.log").open("w") as log:
        result = subprocess.run(
            ["/usr/local/bin/rockstar", "-c", str(config), converted.path],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    catalogue = output / "halos_0.0.ascii"
    if result.returncode or not catalogue.is_file() or catalogue.stat().st_size == 0:
        raise RuntimeError(
            "Rockstar failed or produced no ASCII catalogue; see rockstar.log"
        )
    return catalogue


def load_rockstar_ascii_catalogue(
    path: str | Path, *, expected: RockstarInput
) -> RockstarCatalogue:
    """Parse declared columns and metadata; keep single-worker periodicity visible."""
    source = Path(path)
    config = source.parent / "rockstar.cfg"
    if not source.is_file() or not config.is_file():
        raise ValueError("Rockstar catalogue or effective config is missing")
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("#id "):
        raise ValueError("Rockstar ASCII catalogue has no declared columns")
    names = lines[0][1:].split()
    required = {"id", "num_p", "mvir", "m200c", "x", "y", "z"}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ValueError("Rockstar columns are missing or duplicated")
    metadata = {}
    rows = []
    for line in lines[1:]:
        if line.startswith("#"):
            if line.startswith("#a = "):
                metadata["a"] = float(line.partition("=")[2])
            elif line.startswith("#Om = "):
                for part in line[1:].split(";"):
                    key, _, value = part.strip().partition("=")
                    metadata[key.strip()] = float(value)
            elif line.startswith("#Box size: "):
                metadata["box"] = float(line.split()[2])
            elif line.startswith("#Particle mass: "):
                metadata["particle_mass"] = float(line.split()[2])
            elif line.startswith("#Units: Masses in "):
                metadata["mass_unit"] = line
            elif line.startswith("#Units: Positions in "):
                metadata["position_unit"] = line
            continue
        values = line.split()
        if len(values) != len(names):
            raise ValueError("Rockstar row length disagrees with declared columns")
        rows.append([float(value) for value in values])
    if not all(
        key in metadata
        for key in (
            "a",
            "Om",
            "Ol",
            "h",
            "box",
            "particle_mass",
            "mass_unit",
            "position_unit",
        )
    ):
        raise ValueError(
            "Rockstar catalogue is missing required cosmology or unit metadata"
        )
    if (
        "Msun / h" not in metadata["mass_unit"]
        or "Mpc / h" not in metadata["position_unit"]
    ):
        raise ValueError("Rockstar catalogue units do not match Mpc/h and Msun/h")
    if not np.isclose(metadata["a"], expected.scale_factor, atol=1e-5):
        raise ValueError("Rockstar catalogue scale factor does not match the input")
    if not np.isclose(metadata["box"], expected.box_size_mpc_h, atol=1e-5):
        raise ValueError("Rockstar catalogue box does not match the input")
    if not np.isclose(metadata["Om"], expected.omega_m, atol=1e-5) or not np.isclose(
        metadata["h"], expected.h, atol=1e-5
    ):
        raise ValueError("Rockstar catalogue cosmology does not match the input")
    if not np.isclose(
        metadata["particle_mass"], expected.particle_mass_msun_h, rtol=1e-5
    ):
        raise ValueError("Rockstar catalogue particle mass does not match the input")
    settings = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip().strip('"')
    if (
        settings.get("FILE_FORMAT") != "AREPO"
        or settings.get("AREPO_LENGTH_CONVERSION") != "1"
    ):
        raise ValueError("Rockstar effective HDF5 reader or length conversion changed")
    if float(settings.get("AREPO_MASS_CONVERSION", "nan")) != 1e10:
        raise ValueError("Rockstar effective mass conversion changed")
    matrix = np.asarray(rows, dtype=float).reshape((len(rows), len(names)))
    columns = {name: matrix[:, i] for i, name in enumerate(names)}
    if np.any(~np.isfinite(matrix)) or np.any(columns["num_p"] < 0):
        raise ValueError("Rockstar catalogue contains invalid values")
    return RockstarCatalogue(
        path=str(source.resolve()),
        sha256=_sha256(source),
        config_sha256=_sha256(config),
        columns=columns,
        scale_factor=float(metadata["a"]),
        box_size_mpc_h=float(metadata["box"]),
        omega_m=float(metadata["Om"]),
        h=float(metadata["h"]),
        strict_so_masses=settings.get("STRICT_SO_MASSES") == "1",
        periodic=settings.get("PERIODIC") == "1",
    )
