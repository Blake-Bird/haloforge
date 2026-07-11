# HaloForge Report: Run 002 — Custom Cosmology

Generated: 2026-07-10T22:43:02.656130+00:00
CLASS status: AXICLASS

## Parameters

| parameter | value |
|---|---:|
| A_s | 3.5e-09 |
| H0 | 55.0 |
| N_eff | 3.046 |
| Omega_b | 0.0493 |
| Omega_k | 0.0 |
| Omega_m | 0.45 |
| Tcmb | 2.7255 |
| custom_z_list |  |
| delta_c | 1.686 |
| delta_halo | 200.0 |
| enable_ede | True |
| f_EDE | 0.2 |
| fitting | Sheth-Tormen 2001 |
| k_max | 200.0 |
| k_min | 0.0001 |
| k_pivot | 0.05 |
| k_points | 1200 |
| log10_a_c | -2.5 |
| mass_max_exp | 15 |
| mass_min_exp | 8 |
| mass_points | 80 |
| mode | Learning |
| n_EDE | 3 |
| n_s | 0.9 |
| quad_limit | 200 |
| scf_parameters | 2.806,0.0 |
| selected_mass_exp | 12 |
| single_z | 0.0 |
| tau_reio | 0.06 |
| window_type | Top-hat |
| z_presets_selected | [0, 0.5, 1, 2, 5, 10, 100] |
| z_values | [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] |

## Derived Values

| quantity | value |
|---|---:|
| Omega_cdm | 0.4007 |
| Omega_de | 0.5498619181028628 |
| Omega_r | 0.00013808189713730698 |
| a_c | 0.0031622776601683794 |
| h | 0.55 |
| rho0 | 37774687500.00001 |
| rho_crit | 83943750000.00002 |
| z_c | 315.2277660168379 |
| rho0 | 37774687500.00001 |
| window_type | Top-hat |

## Selected Graph Descriptions

- Auto-exported matter P(k), sigma(M), and HMF products when available.

## Equations Used

- P_R(k) = A_s (k/k_pivot)^(n_s - 1)
- sigma^2(R) = (1 / 2 pi^2) integral k^2 P(k) W^2(kR) dk
- f_PS(sigma) = sqrt(2/pi) (delta_c/sigma) exp[-delta_c^2/(2 sigma^2)]
- dn/dlnM = (rho0 / M) f(sigma) |dlnsigma/dlnM|

## Notes

Auto-saved after Run New Cosmology.
