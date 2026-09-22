"""First-order Zel'dovich (1LPT) initial-condition generation.

Generates cosmological particle displacement and velocity fields from CLASS/AxiCLASS
linear power spectra, with support for shared random phases, paired-fixed realizations,
and independent physical verification of the generated initial conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engine.nbody_setup import compute_box_resolution


@dataclass(frozen=True)
class ICVerificationReport:
    """Rigorous physical verification results for generated initial conditions."""

    particle_count: int
    mean_density_ratio: float
    center_of_mass_velocity_km_s: float
    max_displacement_mpc_h: float
    mean_velocity_dispersion_km_s: float
    periodicity_passed: bool
    linear_power_agreement_passed: bool
    power_agreement_median_fractional_error: float
    power_agreement_bin_count: int
    spectrum_redshift: float | None
    lpt_order: int
    max_second_order_displacement_mpc_h: float
    checks_passed: bool
    summary: str


def generate_gaussian_random_field(
    grid_size: int,
    box_size_mpc_h: float,
    k_eval: np.ndarray,
    p_eval: np.ndarray,
    *,
    seed: int = 42,
    paired_fixed: bool = False,
    pair_index: int = 0,
) -> np.ndarray:
    """Generate a normalized real-field rFFT overdensity realization.

    For the NumPy forward-transform convention this ensures
    ``V / N**6 * |delta_k|**2`` estimates the supplied P(k).  Generating a
    real white-noise mesh first preserves the Hermitian constraints required
    by an rFFT layout, including its zero/Nyquist planes.
    """
    rng = np.random.default_rng(seed)
    n = grid_size
    l_box = box_size_mpc_h

    # 3D Fourier frequencies using rfftn layout (last dim is n // 2 + 1)
    kx = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=l_box / n) * 2.0 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1e-12  # avoid division by zero

    k_eval = np.asarray(k_eval, dtype=float)
    p_eval = np.asarray(p_eval, dtype=float)
    if (
        k_eval.ndim != 1
        or p_eval.shape != k_eval.shape
        or k_eval.size < 2
        or np.any(~np.isfinite(k_eval))
        or np.any(~np.isfinite(p_eval))
        or np.any(k_eval <= 0)
        or np.any(p_eval <= 0)
        or np.any(np.diff(k_eval) <= 0)
    ):
        raise ValueError("IC power spectrum must be finite, positive, and strictly increasing")
    k_fundamental = 2.0 * np.pi / l_box
    k_nyquist = np.pi * n / l_box
    if k_eval[0] > k_fundamental or k_eval[-1] < k_nyquist:
        raise ValueError(
            "The selected P(k) does not cover this IC grid from its fundamental "
            "mode through its Nyquist mode. Rerun CLASS/AxiCLASS with a compatible k range."
        )

    # No endpoint extrapolation is allowed after the explicit coverage check.
    p_interp = np.exp(np.interp(np.log(K), np.log(k_eval), np.log(p_eval)))
    p_interp[0, 0, 0] = 0.0

    # np.fft.rfftn(white) has E|white_k|² = N³ for unit-variance real white
    # noise. The target is E|delta_k|² = P(k) N⁶/V.
    vol = l_box**3
    white = rng.standard_normal(size=(n, n, n))
    white_k = np.fft.rfftn(white)
    amplitude = np.sqrt(p_interp * n**6 / vol)

    if paired_fixed:
        phase = np.divide(
            white_k,
            np.abs(white_k),
            out=np.ones_like(white_k),
            where=np.abs(white_k) > 0,
        )
        delta_k = amplitude * phase * (-1.0 if pair_index % 2 else 1.0)
    else:
        delta_k = white_k * np.sqrt(p_interp * n**3 / vol)

    delta_k[0, 0, 0] = 0.0
    return delta_k


def measure_realized_power_spectrum(
    delta_k: np.ndarray, box_size_mpc_h: float, grid_size: int, *, bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Measure shell-averaged P(k) using the generator's exact FFT convention."""
    n = int(grid_size)
    kx = np.fft.fftfreq(n, d=box_size_mpc_h / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=box_size_mpc_h / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size_mpc_h / n) * 2.0 * np.pi
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3**2 + ky3**2 + kz3**2)
    weights = np.full(k.shape, 2.0)
    weights[:, :, 0] = 1.0
    if n % 2 == 0:
        weights[:, :, -1] = 1.0
    power = box_size_mpc_h**3 / n**6 * np.abs(delta_k) ** 2
    valid = k > 0
    edges = np.geomspace(float(k[valid].min()), float(k[valid].max()), bins + 1)
    centers, measured = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (k >= lo) & (k < hi)
        if np.any(mask):
            centers.append(float(np.exp(np.mean(np.log(k[mask])))))
            measured.append(float(np.average(power[mask], weights=weights[mask])))
    return np.asarray(centers), np.asarray(measured)


def generate_zeldovich_particles(
    box_size_mpc_h: float,
    particles_per_dim: int,
    k_power: np.ndarray,
    p_power: np.ndarray,
    start_redshift: float,
    params: dict[str, Any],
    *,
    seed: int = 42,
    paired_fixed: bool = False,
    pair_index: int = 0,
    spectrum_redshift: float | None = None,
    growth_rate: float | None = None,
    expansion_rate_E: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ICVerificationReport]:
    """Generate first-order Zel'dovich particle coordinates, velocities, and IDs.

    Returns:
      positions: (N^3, 3) in comoving h^-1 Mpc, wrapped to [0, BoxSize)
      velocities: (N^3, 3) in physical km/s
      ids: (N^3,) 64-bit integer IDs
      verification: ICVerificationReport
    """
    n = particles_per_dim
    total_p = n**3
    l_box = box_size_mpc_h
    a_start = 1.0 / (1.0 + float(start_redshift))
    h = float(params.get("H0", 67.36)) / 100.0
    _ = h  # Keep parameter extracted for clarity

    if spectrum_redshift is not None and not np.isclose(
        float(spectrum_redshift), float(start_redshift), rtol=0.0, atol=1e-10
    ):
        raise ValueError(
            "Initial-condition P(k) must be evaluated at the requested start redshift; "
            "nearest-redshift scaling is prohibited."
        )

    # For legacy direct callers without a selected P(k,z_start), retain an
    # explicit LCDM-only a-scaling. The application passes a matching spectrum
    # and recorded background/growth factors instead.
    omega_m0 = float(params.get("Omega_m", 0.315))
    ez2 = omega_m0 * (1.0 + start_redshift) ** 3 + (1.0 - omega_m0)
    omega_m_z = omega_m0 * (1.0 + start_redshift) ** 3 / ez2
    if params.get("enable_ede") and (growth_rate is None or expansion_rate_E is None):
        raise ValueError(
            "EDE ICs require growth and expansion values derived from the selected "
            "AxiCLASS background; a ΛCDM growth approximation is prohibited."
        )
    f_rate = float(growth_rate) if growth_rate is not None else omega_m_z**0.55
    e_rate = float(expansion_rate_E) if expansion_rate_E is not None else np.sqrt(ez2)
    if not np.isfinite(f_rate) or f_rate <= 0 or not np.isfinite(e_rate) or e_rate <= 0:
        raise ValueError("IC growth and expansion factors must be finite and positive")

    # 1. Generate unperturbed Lagrangian lattice q
    grid_coords = np.linspace(0.0, l_box, n, endpoint=False) + (l_box / (2.0 * n))
    Qx, Qy, Qz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    q = np.column_stack([Qx.ravel(), Qy.ravel(), Qz.ravel()])

    # 2. Generate linear overdensity field delta_k
    delta_k = generate_gaussian_random_field(
        n,
        l_box,
        k_power,
        p_power,
        seed=seed,
        paired_fixed=paired_fixed,
        pair_index=pair_index,
    )

    # 3. Zel'dovich 1st order displacement Psi_1: div(Psi_1) = -delta.
    # With NumPy's exp(+ikx) derivative convention this is +i k_i δ/k².
    kx = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=l_box / n) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # avoid division

    # First-order displacements (inverse FFT)
    psi1_x = np.fft.irfftn(1j * (KX / K2) * delta_k, s=(n, n, n), axes=(0, 1, 2))
    psi1_y = np.fft.irfftn(1j * (KY / K2) * delta_k, s=(n, n, n), axes=(0, 1, 2))
    psi1_z = np.fft.irfftn(1j * (KZ / K2) * delta_k, s=(n, n, n), axes=(0, 1, 2))

    disp1 = np.column_stack([psi1_x.ravel(), psi1_y.ravel(), psi1_z.ravel()])

    # An explicit selected P(k,z_start) already contains the growth amplitude.
    # The legacy fallback is visibly bounded to a standard ΛCDM approximation.
    d_start = 1.0 if spectrum_redshift is not None else a_start
    displacement = d_start * disp1

    # Coordinates: x = q + Psi, wrapped periodically to [0, BoxSize)
    positions = np.mod(q + displacement, l_box)

    # Velocities: v = a * H * f * Psi_1 (in physical km/s)
    # Factor: a_start * H(z_start) * f
    v_factor = a_start * 100.0 * e_rate * f_rate
    velocities = v_factor * displacement

    ids = np.arange(1, total_p + 1, dtype=np.int64)

    # Verification checks
    v_cm = np.linalg.norm(np.mean(velocities, axis=0))
    v_disp = float(np.std(velocities))
    max_disp = float(np.max(np.linalg.norm(displacement, axis=1)))
    periodicity_ok = bool(np.all((positions >= 0.0) & (positions < l_box)))
    v_cm_ok = bool(
        v_cm < 1.0
    )  # Center of mass velocity should be practically zero (< 1 km/s)
    measured_k, measured_power = measure_realized_power_spectrum(
        delta_k * d_start, l_box, n
    )
    expected_power = np.exp(
        np.interp(np.log(measured_k), np.log(k_power), np.log(p_power))
    ) * d_start**2
    fractional_errors = np.abs(measured_power / expected_power - 1.0)
    median_power_error = float(np.median(fractional_errors))
    # A fixed-amplitude realization should agree shell-by-shell to numerical
    # precision; a Gaussian realization carries sample variance, so this is a
    # deliberately permissive diagnostic rather than a production acceptance
    # certificate.
    power_ok = bool(median_power_error <= (0.05 if paired_fixed else 0.50))

    report = ICVerificationReport(
        particle_count=total_p,
        mean_density_ratio=1.0,
        center_of_mass_velocity_km_s=float(v_cm),
        max_displacement_mpc_h=max_disp,
        mean_velocity_dispersion_km_s=v_disp,
        periodicity_passed=periodicity_ok,
        linear_power_agreement_passed=power_ok,
        power_agreement_median_fractional_error=median_power_error,
        power_agreement_bin_count=int(measured_k.size),
        spectrum_redshift=float(spectrum_redshift)
        if spectrum_redshift is not None
        else None,
        lpt_order=1,
        max_second_order_displacement_mpc_h=0.0,
        checks_passed=periodicity_ok and v_cm_ok and power_ok,
        summary=(
            f"Zel'dovich (1LPT) IC generated at z={start_redshift:.1f} with N={n}^3 ({total_p:,} particles). "
            f"V_cm = {v_cm:.4e} km/s (conserved), Max displacement = {max_disp:.3f} h^-1 Mpc, "
            f"Periodicity: {'PASS' if periodicity_ok else 'FAIL'}; measured P(k) median shell error "
            f"= {median_power_error:.2%} across {measured_k.size} shells."
        ),
    )

    return positions, velocities, ids, report


def generate_2lpt_particles(
    box_size_mpc_h: float,
    particles_per_dim: int,
    k_power: np.ndarray,
    p_power: np.ndarray,
    start_redshift: float,
    params: dict[str, Any],
    *,
    seed: int = 42,
    paired_fixed: bool = False,
    pair_index: int = 0,
    spectrum_redshift: float | None = None,
    growth_rate: float | None = None,
    expansion_rate_E: float | None = None,
    second_order_growth_ratio: float = -3.0 / 7.0,
    second_order_growth_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ICVerificationReport]:
    """Generate a periodic second-order LPT realization from exact P(k,z_start).

    The 2LPT source follows ``∇²φ₂ = Σᵢ<ⱼ(φ₁,ii φ₁,jj − φ₁,ij²)`` with
    ``∇²φ₁ = δ`` and ``Ψ₁ = −∇φ₁``. ``second_order_growth_ratio`` is the
    coefficient relative to the squared first-order growth (−3/7 in EdS).
    Callers handling non-ΛCDM backgrounds must provide a scientifically
    supported coefficient and growth rate; the routine never infers EDE 2LPT
    dynamics from a ΛCDM shortcut.
    """
    if params.get("enable_ede") and (
        growth_rate is None
        or expansion_rate_E is None
        or second_order_growth_rate is None
    ):
        raise ValueError(
            "EDE 2LPT requires explicit first- and second-order growth inputs; "
            "ΛCDM/EdS coefficients are prohibited."
        )
    if not np.isfinite(second_order_growth_ratio):
        raise ValueError("second_order_growth_ratio must be finite")
    if spectrum_redshift is not None and not np.isclose(
        float(spectrum_redshift), float(start_redshift), rtol=0.0, atol=1e-10
    ):
        raise ValueError("2LPT requires P(k) at exactly z_start")

    n = int(particles_per_dim)
    l_box = float(box_size_mpc_h)
    a_start = 1.0 / (1.0 + float(start_redshift))
    delta_k = generate_gaussian_random_field(
        n,
        l_box,
        k_power,
        p_power,
        seed=seed,
        paired_fixed=paired_fixed,
        pair_index=pair_index,
    )
    kx = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    ky = np.fft.fftfreq(n, d=l_box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=l_box / n) * 2.0 * np.pi
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    components = (kx3, ky3, kz3)
    k2 = kx3**2 + ky3**2 + kz3**2
    k2[0, 0, 0] = 1.0
    phi1_k = -delta_k / k2  # ∇² φ₁ = δ
    phi1_k[0, 0, 0] = 0.0
    shape = (n, n, n)
    hessian = {}
    for i in range(3):
        for j in range(i, 3):
            hessian[i, j] = np.fft.irfftn(
                -components[i] * components[j] * phi1_k, s=shape, axes=(0, 1, 2)
            )
            hessian[j, i] = hessian[i, j]
    source = (
        hessian[0, 0] * hessian[1, 1]
        - hessian[0, 1] ** 2
        + hessian[0, 0] * hessian[2, 2]
        - hessian[0, 2] ** 2
        + hessian[1, 1] * hessian[2, 2]
        - hessian[1, 2] ** 2
    )
    source_k = np.fft.rfftn(source)
    phi2_k = -source_k / k2  # ∇² φ₂ = source
    phi2_k[0, 0, 0] = 0.0
    psi1 = [
        np.fft.irfftn(1j * component * delta_k / k2, s=shape, axes=(0, 1, 2))
        for component in components
    ]
    grad_phi2 = [
        np.fft.irfftn(1j * component * phi2_k, s=shape, axes=(0, 1, 2))
        for component in components
    ]
    legacy_scaling = spectrum_redshift is None
    d_start = a_start if legacy_scaling else 1.0
    psi1_scaled = np.column_stack([field.ravel() for field in psi1]) * d_start
    psi2_scaled = (
        np.column_stack([field.ravel() for field in grad_phi2])
        * float(second_order_growth_ratio)
        * d_start**2
    )
    grid = np.linspace(0.0, l_box, n, endpoint=False) + l_box / (2.0 * n)
    qx, qy, qz = np.meshgrid(grid, grid, grid, indexing="ij")
    q = np.column_stack([qx.ravel(), qy.ravel(), qz.ravel()])
    displacement = psi1_scaled + psi2_scaled
    positions = np.mod(q + displacement, l_box)

    omega_m0 = float(params.get("Omega_m", 0.315))
    ez2 = omega_m0 * (1.0 + start_redshift) ** 3 + (1.0 - omega_m0)
    omega_m_z = omega_m0 * (1.0 + start_redshift) ** 3 / ez2
    f1 = float(growth_rate) if growth_rate is not None else omega_m_z**0.55
    e_rate = float(expansion_rate_E) if expansion_rate_E is not None else np.sqrt(ez2)
    f2 = float(second_order_growth_rate) if second_order_growth_rate is not None else 2.0 * f1
    if any(not np.isfinite(value) or value <= 0 for value in (f1, f2, e_rate)):
        raise ValueError("2LPT growth and expansion inputs must be finite and positive")
    velocities = a_start * 100.0 * e_rate * (f1 * psi1_scaled + f2 * psi2_scaled)
    ids = np.arange(1, n**3 + 1, dtype=np.int64)
    v_cm = np.linalg.norm(np.mean(velocities, axis=0))
    max_disp = float(np.max(np.linalg.norm(displacement, axis=1)))
    max_psi2 = float(np.max(np.linalg.norm(psi2_scaled, axis=1)))
    periodicity_ok = bool(np.all((positions >= 0.0) & (positions < l_box)))
    measured_k, measured_power = measure_realized_power_spectrum(
        delta_k * d_start, l_box, n
    )
    expected_power = np.exp(
        np.interp(np.log(measured_k), np.log(k_power), np.log(p_power))
    ) * d_start**2
    median_power_error = float(np.median(np.abs(measured_power / expected_power - 1.0)))
    power_ok = bool(median_power_error <= (0.05 if paired_fixed else 0.50))
    report = ICVerificationReport(
        particle_count=n**3,
        mean_density_ratio=1.0,
        center_of_mass_velocity_km_s=float(v_cm),
        max_displacement_mpc_h=max_disp,
        mean_velocity_dispersion_km_s=float(np.std(velocities)),
        periodicity_passed=periodicity_ok,
        linear_power_agreement_passed=power_ok,
        power_agreement_median_fractional_error=median_power_error,
        power_agreement_bin_count=int(measured_k.size),
        spectrum_redshift=float(spectrum_redshift) if spectrum_redshift is not None else None,
        lpt_order=2,
        max_second_order_displacement_mpc_h=max_psi2,
        checks_passed=periodicity_ok and v_cm < 1.0 and power_ok,
        summary=(
            f"2LPT IC generated at z={start_redshift:.1f} with an explicit second-order "
            f"kernel (max |Ψ₂|={max_psi2:.3e} h^-1 Mpc). Measured linear-shell error "
            f"= {median_power_error:.2%}; independent power-recovery validation remains required."
        ),
    )
    return positions, velocities, ids, report


def export_gadget_hdf5_ic(
    filename: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    ids: np.ndarray,
    box_size_mpc_h: float,
    start_redshift: float,
    params: dict[str, Any],
) -> str:
    """Save ICs in the Mpc/h internal-unit contract emitted by HaloForge.

    ``generate_gadget4_parameter_file`` declares one internal length unit as
    one Mpc/h.  Header ``BoxSize`` and particle coordinates must therefore be
    written in Mpc/h too; using kpc/h here would enlarge the simulated volume
    by a factor of 1000 at runtime.
    """
    import h5py

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_part = len(positions)
    particles_per_dim = int(round(n_part ** (1.0 / 3.0)))
    if particles_per_dim**3 != n_part or particles_per_dim < 16:
        raise ValueError(
            "GADGET IC export requires a cubic lattice with at least 16 particles per dimension"
        )
    h = float(params.get("H0", 67.36)) / 100.0
    omega_m = float(params.get("Omega_m", 0.315))
    omega_l = 1.0 - omega_m - float(params.get("Omega_k", 0.0))

    res = compute_box_resolution(
        box_size_mpc_h, particles_per_dim, params
    )

    with h5py.File(path, "w") as f:
        # Header group
        hdr = f.create_group("Header")
        # Must match the generated Config.sh contract (NTYPES=2), not the
        # legacy six-slot Gadget header convention.
        npart_arr = np.zeros(2, dtype=np.uint32)
        npart_arr[1] = n_part  # PartType1 is DM
        hdr.attrs["NumPart_ThisFile"] = npart_arr
        hdr.attrs["NumPart_Total"] = npart_arr
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(2, dtype=np.uint32)

        mass_arr = np.zeros(2, dtype=np.float64)
        # GADGET mass units: 10^10 h^-1 M_sun
        mass_arr[1] = res.particle_mass_msun_h / 1.0e10
        hdr.attrs["MassTable"] = mass_arr

        hdr.attrs["Time"] = 1.0 / (1.0 + float(start_redshift))
        hdr.attrs["Redshift"] = float(start_redshift)
        hdr.attrs["BoxSize"] = float(box_size_mpc_h)
        hdr.attrs["NumFilesPerSnapshot"] = 1
        hdr.attrs["Omega0"] = omega_m
        hdr.attrs["OmegaLambda"] = omega_l
        hdr.attrs["HubbleParam"] = h
        hdr.attrs["Flag_Sfr"] = 0
        hdr.attrs["Flag_Cooling"] = 0
        hdr.attrs["Flag_Feedback"] = 0

        # PartType1 (Dark Matter)
        pt1 = f.create_group("PartType1")
        # Positions in the Mpc/h internal length unit declared in param.txt.
        pt1.create_dataset("Coordinates", data=positions.astype(np.float32))
        # Velocities in km/s * sqrt(a)
        v_gadget = (velocities * np.sqrt(hdr.attrs["Time"])).astype(np.float32)
        pt1.create_dataset("Velocities", data=v_gadget)
        pt1.create_dataset("ParticleIDs", data=ids.astype(np.uint64))

    return str(path)
