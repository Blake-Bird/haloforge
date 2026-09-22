"""GADGET-4 integration architecture, configuration builder, and installation doctor.

Provides reproducible cross-platform execution prerequisites, system installation
diagnostics, and GADGET-4 configuration planning.  It deliberately does not
invent a generic EDE background-file interface for upstream GADGET-4.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import platform
import shutil
import subprocess
import json
from pathlib import Path
from typing import Any

import numpy as np


GADGET4_VERSION = "GADGET-4 (pinned upstream commit)"
GADGET4_PINNED_COMMIT = "6fb393b5e25907f2ff06211f67d341c5e40e90d1"
# A production image must be supplied as an immutable digest, e.g.
# ``registry.example/haloforge-gadget4@sha256:<digest>``.  A Docker daemon or
# an image tag alone is not reproducibility evidence.
GADGET4_CONTAINER_IMAGE = os.environ.get("HALOFORGE_GADGET4_IMAGE", "unconfigured")
GADGET4_CITATION = (
    "Springel, V., Pakmor, R., Zier, O., & Reinecke, M. (2021). "
    "Simulating cosmic structure formation with the GADGET-4 code. "
    "Monthly Notices of the Royal Astronomical Society, 506(2), 2871-2949. arXiv:2010.03567."
)
GADGET4_LICENSE = "GNU General Public License v3.0 (GPL-3.0)"
GADGET4_ACCEPTANCE_SCHEMA = "haloforge-gadget4-acceptance-v1"
GADGET4_OUTPUT_LIST_REFERENCE = (
    "https://wwwmpa.mpa-garching.mpg.de/gadget4/05_parameterfile/"
)


def validate_bundled_acceptance() -> tuple[bool, str]:
    """Verify image-owned executable hashes and real smoke-run artifacts."""
    evidence = Path("/opt/HALOFORGE_NBODY_ACCEPTANCE.json")
    gadget = Path("/usr/local/bin/Gadget4")
    rockstar = Path("/usr/local/bin/rockstar")
    if not all(path.is_file() for path in (evidence, gadget, rockstar)):
        return False, "Bundled executables or acceptance evidence are missing."

    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    try:
        report = json.loads(evidence.read_text())
        if (
            report.get("gadget_commit") != GADGET4_PINNED_COMMIT
            or report.get("gadget_binary_sha256") != digest(gadget)
            or report.get("rockstar_binary_sha256") != digest(rockstar)
            or report.get("fixture_kind") != "synthetic-power-software-smoke-not-science-validation"
        ):
            raise ValueError("Bundled binary identity or fixture scope differs")
        root = Path(report["run"]).resolve()
        if not root.is_relative_to(Path("/opt/haloforge-acceptance").resolve()):
            raise ValueError("Acceptance run is outside its image-owned directory")
        run = json.loads((root / "run.json").read_text())
        if run.get("state") != "completed" or len(run.get("snapshots", [])) != 3:
            raise ValueError("Acceptance simulation is incomplete")
        for item in run["snapshots"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Acceptance snapshot bytes changed")
            if digest(Path(item["catalogue_path"])) != item["catalogue_sha256"]:
                raise ValueError("Acceptance FoF bytes changed")
        if digest(root / "catalogues/rockstar_final/halos_0.0.ascii") != report["rockstar_catalogue_sha256"]:
            raise ValueError("Acceptance Rockstar bytes changed")
        if digest(root / "exports/structure.gif") != report["movie_sha256"]:
            raise ValueError("Acceptance movie bytes changed")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return False, f"Bundled N-body acceptance could not be verified: {exc}"
    return True, "Bundled GADGET-4, FoF/Subfind, Rockstar, and movie fixture verified."


@dataclass(frozen=True)
class InstallationDoctorReport:
    """Diagnostic scorecard assessing the local host environment for GADGET-4 execution."""

    docker_available: bool
    docker_daemon_available: bool
    docker_version: str
    docker_arch: str
    immutable_image_configured: bool
    image_available: bool
    acceptance_manifest_configured: bool
    acceptance_evidence_present: bool
    acceptance_evidence_error: str
    wsl2_detected: bool
    native_mpi_available: bool
    mpi_version: str
    cpu_cores: int
    ram_gb: float
    free_disk_gb: float
    recommended_runtime: str
    ready_for_simulation: bool
    recommendations: list[str]


def validate_acceptance_manifest(
    manifest_path: str | Path | None, image_digest: str
) -> tuple[bool, str]:
    """Validate the minimal durable evidence required before runtime readiness.

    A manifest is evidence *about* a separately executed official acceptance
    case, not a replacement for it.  Matching the immutable image and pinned
    source revision prevents a previous build's report from enabling another.
    """
    if not manifest_path:
        return False, "No GADGET-4 acceptance manifest is configured."
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return False, "Configured GADGET-4 acceptance manifest does not exist."
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Could not read the GADGET-4 acceptance manifest: {exc}"
    if not isinstance(value, dict):
        return False, "GADGET-4 acceptance manifest must be a JSON object."
    required = {
        "schema_version": GADGET4_ACCEPTANCE_SCHEMA,
        "status": "passed",
        "gadget4_pinned_commit": GADGET4_PINNED_COMMIT,
        "image_digest": image_digest,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            return False, f"Acceptance manifest {key!r} does not match this runtime."
    for key in ("fixture_id", "snapshot_sha256", "completed_at"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            return False, f"Acceptance manifest is missing required {key!r} evidence."
    return True, "Pinned-image GADGET-4 acceptance evidence is recorded."


def run_installation_doctor() -> InstallationDoctorReport:
    """Inspect local container engines, MPI, CPU, and disk to determine simulation readiness."""
    # Check Docker
    docker_ok = False
    docker_ver = "Not found"
    docker_arch = platform.machine()
    docker_daemon_available = False
    docker_cmd = shutil.which("docker")
    if docker_cmd:
        try:
            res = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0:
                docker_ok = True
                docker_ver = res.stdout.strip()
                daemon = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                docker_daemon_available = daemon.returncode == 0
        except Exception:
            pass

    immutable_image_configured = "@sha256:" in GADGET4_CONTAINER_IMAGE
    image_available = False
    if docker_daemon_available and immutable_image_configured:
        try:
            inspected = subprocess.run(
                ["docker", "image", "inspect", GADGET4_CONTAINER_IMAGE],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            image_available = inspected.returncode == 0
        except Exception:
            pass

    acceptance_path = os.environ.get(
        "HALOFORGE_GADGET4_ACCEPTANCE_MANIFEST", ""
    ).strip()
    bundled = Path("/usr/local/bin/Gadget4").is_file()
    acceptance_evidence, acceptance_error = (
        validate_bundled_acceptance()
        if bundled
        else validate_acceptance_manifest(
            acceptance_path or None, GADGET4_CONTAINER_IMAGE
        )
    )

    # Check WSL2
    wsl2 = False
    if platform.system() == "Linux" and "microsoft" in platform.uname().release.lower():
        wsl2 = True

    # Check native MPI
    mpi_ok = False
    mpi_ver = "Not found"
    for mpi_bin in ("mpirun", "mpicxx", "srun"):
        bin_path = shutil.which(mpi_bin)
        if bin_path:
            try:
                res = subprocess.run(
                    [bin_path, "--version"], capture_output=True, text=True, timeout=3
                )
                if res.returncode == 0:
                    mpi_ok = True
                    mpi_ver = f"{mpi_bin}: " + res.stdout.splitlines()[0]
                    break
            except Exception:
                pass

    cpu_cores = os.cpu_count() or 1
    # Check RAM & Disk
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except (ValueError, OSError, AttributeError):
        ram_gb = 0.0
    data_dir = Path(os.environ.get("HALOFORGE_DATA_DIR", "/var/lib/haloforge"))
    free_disk_gb = shutil.disk_usage(data_dir if data_dir.exists() else ".").free / (1024**3)
    writable_data_dir = data_dir.is_dir() and os.access(data_dir, os.W_OK)

    # Determine recommended runtime
    recs = []
    if bundled and acceptance_evidence:
        runtime = "Bundled GADGET-4 and Rockstar smoke fixture verified"
    elif bundled:
        runtime = "Bundled GADGET-4 is present, but its acceptance fixture failed"
    elif docker_daemon_available and image_available:
        runtime = "Docker image present; GADGET-4 acceptance test still required"
    elif docker_daemon_available:
        runtime = "Docker available; immutable GADGET-4 image is not configured locally"
    elif docker_ok:
        runtime = "Docker command found, but its daemon is not running"
        recs.append(
            "Start the local Docker daemon before checking GADGET-4 images or executing a simulation."
        )
    elif wsl2 and mpi_ok:
        runtime = "WSL2 MPI available; a verified GADGET-4 build is still required"
    elif mpi_ok:
        runtime = "Native MPI available; a verified GADGET-4 build is still required"
    else:
        runtime = "No verified GADGET-4 execution environment detected"
        recs.append(
            "Configure an immutable, locally available GADGET-4 Docker image before simulation execution can be enabled."
        )
    if not bundled and docker_ok and not immutable_image_configured:
        recs.append(
            "Set HALOFORGE_GADGET4_IMAGE to an immutable image reference containing @sha256:…; tags are not accepted as reproducible simulation evidence."
        )
    if not bundled and immutable_image_configured and not image_available:
        recs.append(
            "The configured immutable GADGET-4 image is not available locally. Pull/build it explicitly and run the official acceptance case before enabling simulation execution."
        )
    if not acceptance_evidence:
        recs.append(acceptance_error)

    if ram_gb < 8.0:
        recs.append(
            f"System has only {ram_gb:.1f} GB RAM. Keep particle counts below 128^3."
        )
    if free_disk_gb < 20.0:
        recs.append(f"Only {free_disk_gb:.1f} GB disk free. Limit snapshot frequency.")
    if bundled and not writable_data_dir:
        recs.append("The local simulation data directory is not writable.")

    # An immutable image is only an execution prerequisite. The separate
    # manifest binds documented acceptance evidence to that exact build.
    ready = acceptance_evidence and (
        (bundled and writable_data_dir and free_disk_gb >= 1)
        or (docker_daemon_available and immutable_image_configured and image_available)
    )

    return InstallationDoctorReport(
        docker_available=docker_ok,
        docker_daemon_available=docker_daemon_available,
        docker_version=docker_ver,
        docker_arch=docker_arch,
        immutable_image_configured=immutable_image_configured,
        image_available=image_available,
        acceptance_manifest_configured=bool(acceptance_path) or bundled,
        acceptance_evidence_present=acceptance_evidence,
        acceptance_evidence_error=acceptance_error,
        wsl2_detected=wsl2,
        native_mpi_available=mpi_ok,
        mpi_version=mpi_ver,
        cpu_cores=cpu_cores,
        ram_gb=float(round(ram_gb, 1)),
        free_disk_gb=float(round(free_disk_gb, 1)),
        recommended_runtime=runtime,
        ready_for_simulation=ready,
        recommendations=recs,
    )


def generate_config_sh(
    *,
    is_hydro: bool = False,
    enable_2lpt: bool = False,
    enable_fof: bool = True,
    enable_subfind: bool = True,
    pm_mesh: int = 32,
) -> str:
    """Generate a validated GADGET-4 Config.sh compile-time header."""
    if is_hydro:
        raise ValueError(
            "Hydrodynamic/SPH GADGET-4 configurations are not supported by "
            "HaloForge. The available workflow is collisionless DM-only; do not "
            "generate an incomplete gas-physics configuration."
        )
    lines = [
        "# GADGET-4 compile-time configuration generated by HaloForge",
        f"# Pinned revision: {GADGET4_PINNED_COMMIT}",
        "PERIODIC",
        "SELFGRAVITY",
        "TREEPM_NOTIMESPLIT",
        f"PMGRID={pm_mesh}",
        "NSOFTCLASSES=1",
        "NTYPES=2",
        "DOUBLEPRECISION=1",
        "DOUBLEPRECISION_FFTW",
        "POWERSPEC_ON_OUTPUT",
    ]

    if enable_2lpt:
        raise ValueError(
            "SECOND_ORDER_LPT_ICS requires specially constructed Jenkins ICs; "
            "HaloForge currently writes 1LPT ICs and cannot enable this option."
        )

    if enable_fof:
        # These are compile-time options in GADGET-4, not parameter-file
        # entries.  Keep the primary link set explicitly DM-only: adding
        # particle type 0 would be incorrect for this collisionless plan.
        lines.extend(
            [
                "FOF",
                "FOF_PRIMARY_LINK_TYPES=2",
                "FOF_SECONDARY_LINK_TYPES=0",
                "FOF_GROUP_MIN_LEN=20",
                "FOF_LINKLENGTH=0.2",
            ]
        )
        if enable_subfind:
            lines.append("SUBFIND")

    lines.append("# DM-only collisionless physics (no SPH/cooling)")

    return "\n".join(lines) + "\n"


def generate_gadget4_parameter_file(
    box_size_mpc_h: float,
    particles_per_dim: int,
    output_dir: str,
    output_redshifts: list[float],
    params: dict[str, Any],
    *,
    start_redshift: float = 49.0,
    softening_kpc_h: float | None = None,
    ic_filename: str = "ics.hdf5",
    tabulated_expansion_file: str | None = None,
) -> str:
    """Generate a validated GADGET-4 param.txt runtime parameter file."""
    # The corresponding output_times.txt is emitted by
    # ``generate_output_times_file``. Validate its inputs here so a parameter
    # file never silently references an impossible output list.
    generate_output_times_file(output_redshifts, start_redshift=start_redshift)
    h = float(params.get("H0", 67.36)) / 100.0
    omega_m = float(params.get("Omega_m", 0.315))
    omega_b = float(params.get("Omega_b", 0.049))
    omega_l = 1.0 - omega_m - float(params.get("Omega_k", 0.0))

    mean_sep_kpc_h = (box_size_mpc_h * 1000.0) / particles_per_dim
    softening = softening_kpc_h or (mean_sep_kpc_h / 30.0)

    time_begin = 1.0 / (1.0 + start_redshift)
    time_max = 1.0

    # GADGET-4's HDF5 reader appends ``.hdf5`` to InitCondFile.  Accept the
    # intuitive exported filename but emit its required stem in param.txt.
    ic_path = Path(ic_filename)
    ic_stem = str(ic_path.with_suffix("")) if ic_path.suffix == ".hdf5" else ic_filename

    # Sort output times descending in redshift (ascending in scale factor)
    lines = [
        "% GADGET-4 Runtime Parameter File generated by HaloForge",
        f"% Box size: {box_size_mpc_h} h^-1 Mpc, N_part = {particles_per_dim}^3",
        "",
        f"InitCondFile              {ic_stem}",
        f"OutputDir                 {output_dir}",
        "SnapshotFileBase          snap",
        "ICFormat                  3",
        "SnapFormat                3",
        "TimeLimitCPU              86400",
        "MaxMemSize                1024",
        f"TimeBegin                 {time_begin:.6f}",
        f"TimeMax                   {time_max:.6f}",
        f"BoxSize                   {box_size_mpc_h}",
        "",
        "% Cosmological parameters",
        "ComovingIntegrationOn     1",
        f"HubbleParam               {h:.6f}",
        f"Omega0                    {omega_m:.6f}",
        f"OmegaLambda               {omega_l:.6f}",
        f"OmegaBaryon               {omega_b:.6f}",
        "",
        "% GADGET internal units: Mpc/h, 1e10 Msun/h, km/s",
        "UnitLength_in_cm         3.085678e24",
        "UnitMass_in_g            1.989e43",
        "UnitVelocity_in_cm_per_s 1.0e5",
        "GravityConstantInternal  0",
        "Hubble                    100.0",
        "",
        "% Gravitational force softening (comoving Mpc/h)",
        "SofteningClassOfPartType0 0",
        "SofteningClassOfPartType1 0",
        f"SofteningComovingClass0   {softening / 1000.0:.8f}",
        f"SofteningMaxPhysClass0    {softening / 1000.0:.8f}",
        "",
        "% Timestepping and accuracy",
        "MaxSizeTimestep           0.025",
        "MinSizeTimestep           0.0",
        "ErrTolIntAccuracy         0.025",
        "ErrTolTheta               0.5",
        "",
        "% Output cadence",
        "OutputListOn              1",
        "OutputListFilename        output_times.txt",
        "TimeBetSnapshot           0.0",
        "TimeOfFirstSnapshot       0.0",
        "TimeBetStatistics         0.01",
        "NumFilesPerSnapshot       1",
        "MaxFilesWithConcurrentIO  1",
        "CpuTimeBetRestartFile     3600.0",
        "",
        "% FoF settings are compile-time Config.sh options.",
        "CourantFac                 0.3",
        "ErrTolThetaMax             1.0",
        "ErrTolForceAcc             0.002",
        "TypeOfOpeningCriterion     1",
        "TopNodeFactor              3.0",
        "ActivePartFracForNewDomainDecomp 0.01",
        "ActivePartFracForPMinsteadOfEwald 0.05",
        "DesNumNgb                  64",
        "MaxNumNgbDeviation         1",
        "DesLinkNgb                 20",
        "ArtBulkViscConst           1.0",
        "MinEgySpec                 0",
        "InitGasTemp                0",
    ]

    if tabulated_expansion_file:
        raise ValueError(
            "HaloForge cannot emit an ExpansionHistoryFile setting: upstream GADGET-4 "
            "has no validated generic EDE background-file runtime contract in this project. "
            "Do not represent a phenomenological H(a) table as an EDE GADGET-4 run."
        )

    return "\n".join(lines) + "\n"


def generate_output_times_file(
    output_redshifts: list[float], *, start_redshift: float
) -> str:
    """Return the official GADGET-4 plain-ASCII output-time list.

    For comoving integration GADGET-4 time is the scale factor, so each
    requested redshift is converted to ``a = 1 / (1 + z)``. The upstream
    manual specifies one floating-point desired output time per line and a
    default maximum of 1100 values; no comments are placed in this file.
    """
    if not isinstance(output_redshifts, list) or not output_redshifts:
        raise ValueError("At least one GADGET-4 output redshift is required")
    start = float(start_redshift)
    if not np.isfinite(start) or start < 0:
        raise ValueError("GADGET-4 start_redshift must be finite and nonnegative")
    if len(output_redshifts) > 1100:
        raise ValueError("GADGET-4 output list exceeds the default 1100-entry limit")
    scale_factors = []
    for redshift in output_redshifts:
        z = float(redshift)
        if not np.isfinite(z) or z < 0:
            raise ValueError("GADGET-4 output redshifts must be finite and nonnegative")
        if z > start + 1e-12:
            raise ValueError(
                "GADGET-4 output redshift precedes the requested simulation start"
            )
        scale_factors.append(1.0 / (1.0 + z))
    if len({round(value, 14) for value in scale_factors}) != len(scale_factors):
        raise ValueError("GADGET-4 output redshifts contain duplicate scale factors")
    return "".join(f"{value:.12g}\n" for value in sorted(scale_factors))


def generate_tabulated_expansion_history(
    params: dict[str, Any],
    *,
    num_points: int = 1000,
    a_min: float = 1e-4,
    a_max: float = 1.0,
) -> str:
    """Refuse the retired phenomenological EDE table interface.

    The arguments remain temporarily for API compatibility, but this function
    cannot manufacture valid EDE dynamics from cosmological parameters alone.
    A project-specific, externally validated GADGET extension would need an
    explicit contract and acceptance data before an EDE table can be emitted.
    """
    _ = (params, num_points, a_min, a_max)
    raise RuntimeError(
        "Phenomenological EDE expansion tables were removed. HaloForge does not "
        "claim a validated EDE-to-GADGET-4 dynamics interface."
    )
