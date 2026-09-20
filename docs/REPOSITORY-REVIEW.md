# Repository review

The review is ongoing. Implementation and test results are recorded in [Implementation status](IMPLEMENTATION-STATUS.md); scientific conventions and sources are in [Science references](SCIENCE-REFERENCES.md). Passing the current tests does not establish that every workflow or scientific regime has been validated.

## Verified scope

- CLASS/AxiCLASS executes in an isolated worker with explicit input and output contracts. Invalid or incomplete spectra cannot enter the cache as successful results.
- Analytic tests cover window functions, variance integration, HMF multiplicities, finite-range cumulative abundance, interpolation, redshift selection, and Gaussian-field normalization.
- An independent CAMB fixture checks one flat, massless-neutrino ΛCDM model at z = 0, 2, and 10. Its settings and tolerances are stored with the reference. It does not validate EDE or halo abundances.
- Saved-run checks detect array-file and calculation-identity changes. Failed records remain inspectable and cannot be overwritten, exported as scientific results, or used in comparisons. This is local corruption detection, not protection against an adversary who can rewrite all files and hashes.
- Application tests cover startup, workspace rendering, draft recovery, guided comparison handoff, and repeated comparison-control changes. A user-reported NumPy widget-state crash demonstrated why initial-render tests alone were insufficient; a failing-then-passing interaction regression now covers it.
- Container checks exercise the compiled solver and static figure renderer. See the implementation status for the exact tested build and results.

## Remaining review

1. Check every fitting-function citation, calibration claim, mass definition, and supported domain against its primary source. In particular, distinguish a mathematical evaluation domain from simulation calibration.
2. Extend independent reference coverage to additional physical regimes and HMF implementations, with justified tolerances and explicit conventions.
3. Complete saved-array structural validation and interrupted-write recovery testing, including legacy records and partial export failures.
4. Exercise all multi-step UI workflows and responsive layouts, including keyboard navigation, comparisons after saved-run changes, unusual inputs, and error recovery.
5. Complete the source review, performance review, dependency audit, and clean-install release rehearsal. Retain reproducible evidence for the shipped revision.
6. Confirm project ownership and select a repository license. Third-party font notices are already included with the report fonts.

No grades or maturity ratings are assigned. Release approval requires evidence against the requirements in [OBJECTIVE.md](../OBJECTIVE.md), [REQUIREMENTS.md](../REQUIREMENTS.md), and [Release policy](RELEASE-POLICY.md).
