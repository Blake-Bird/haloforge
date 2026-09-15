## Summary

Describe the user-visible or scientific behavior changed. Keep claims proportional to the evidence.

## Verification

- [ ] `ruff format --check app.py config content engine state tests`
- [ ] `ruff check app.py config content engine state tests`
- [ ] `pytest -q`
- [ ] I performed any relevant manual UI, export, or Docker verification.

## Scientific and teaching impact

- [ ] This change does not overstate numerical precision, calibration, cosmology support, or publication readiness.
- [ ] Units, equations, assumptions, validity limits, and citations were updated where relevant.
- [ ] Teaching copy keeps the distinction between calculated curves, empirical fits, and observations clear.

## Data and security impact

- [ ] This change does not add telemetry, auto-upload, shared persistence, or personal-data collection.
- [ ] Any storage, export, import, or sharing change preserves the local-only / fail-closed hosting boundary.
- [ ] No generated runs, exports, caches, credentials, or private research artifacts are included.

## Evidence still needed

List independent validation, accessibility review, performance measurement, or other work that remains outside this pull request.
