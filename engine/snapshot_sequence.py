"""Validated recorded-particle sequences and fixed-scale structure movies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import plotly.io as pio

from engine.gadget_snapshot import GadgetSnapshotParticles, load_gadget4_dm_snapshot
from engine.snapshot_visuals import density_slice_figure, projected_overdensity


def load_snapshot_sequence(
    paths: list[str | Path], *, omega_m: float | None = None, h: float | None = None
) -> list[GadgetSnapshotParticles]:
    """Load two to twelve snapshots of one equal-mass periodic particle set."""
    if not 2 <= len(paths) <= 12:
        raise ValueError("A structure movie needs 2–12 recorded snapshots")
    snapshots = [load_gadget4_dm_snapshot(path, omega_m=omega_m, h=h) for path in paths]
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        for key in ("box_size_mpc_h", "particle_mass_msun_h", "omega_m", "h"):
            if not np.isclose(
                getattr(snapshot, key), getattr(reference, key), rtol=0, atol=1e-9
            ):
                raise ValueError(f"Snapshot sequence disagrees on {key}")
        if not np.array_equal(
            np.sort(snapshot.particle_ids), np.sort(reference.particle_ids)
        ):
            raise ValueError("Snapshot sequence particle IDs differ")
    if any(a.redshift <= b.redshift for a, b in zip(snapshots, snapshots[1:])):
        raise ValueError("Snapshot paths must be ordered from high to low redshift")
    return snapshots


def snapshot_sequence_manifest(snapshots: list[GadgetSnapshotParticles]) -> dict:
    """Record exact frame files, geometry, units, and provenance."""
    if not snapshots:
        raise ValueError("Sequence is empty")
    entries = []
    for snapshot in snapshots:
        source = Path(snapshot.source_path)
        files = []
        for name in snapshot.source_paths or (snapshot.source_path,):
            path = Path(name)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"file_name": path.name, "sha256": digest.hexdigest()})
        combined_hash = (
            files[0]["sha256"]
            if len(files) == 1
            else hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        entries.append(
            {
                "file_name": source.name,
                "sha256": combined_hash,
                "files": files,
                "redshift": float(snapshot.redshift),
                "scale_factor": float(snapshot.scale_factor),
                "particle_count": int(snapshot.particle_ids.size),
            }
        )
    return {
        "schema_version": "haloforge-snapshot-movie-v1",
        "source": "recorded PartType1 GADGET HDF5 snapshots",
        "box_size_mpc_h": float(snapshots[0].box_size_mpc_h),
        "particle_mass_msun_h": float(snapshots[0].particle_mass_msun_h),
        "omega_m": float(snapshots[0].omega_m),
        "h": float(snapshots[0].h),
        "position_unit": "comoving h^-1 Mpc",
        "rendering": "full-box particle count projection; fixed color range across frames",
        "frames": entries,
    }


def render_snapshot_movie(
    snapshots: list[GadgetSnapshotParticles],
    *,
    video_format: str = "mp4",
    fps: float = 2.0,
    cells: int = 64,
    width: int = 960,
    height: int = 720,
) -> bytes:
    """Render only recorded snapshots, with a shared density color scale."""
    fmt = video_format.lower().lstrip(".")
    if fmt not in {"mp4", "webm", "gif"}:
        raise ValueError("Movie format must be mp4, webm, or gif")
    if not 2 <= len(snapshots) <= 12:
        raise ValueError("A structure movie needs 2–12 recorded snapshots")
    if not np.isfinite(fps) or not 0.1 <= fps <= 30:
        raise ValueError("Movie fps must be between 0.1 and 30")
    contrasts = [projected_overdensity(snapshot, cells) for snapshot in snapshots]
    zmin = min(float(np.min(value)) for value in contrasts)
    zmax = max(float(np.max(value)) for value in contrasts)
    pixels = []
    for snapshot in snapshots:
        figure = density_slice_figure(snapshot, cells=cells)
        figure.data[0].update(zmin=zmin, zmax=zmax)
        png = pio.to_image(figure, format="png", width=width, height=height, scale=1)
        pixels.append(iio.imread(png, extension=".png"))
    options = {"extension": f".{fmt}"}
    if fmt == "mp4":
        options.update(fps=float(fps), codec="libx264", pixelformat="yuv420p")
    elif fmt == "webm":
        options.update(fps=float(fps), codec="libvpx-vp9")
    else:
        options.update(duration=int(round(1000.0 / fps)), loop=0)
    return bytes(iio.imwrite("<bytes>", np.stack(pixels), **options))
