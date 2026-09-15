# HaloForge repository review

This is a scope-aware engineering review of the repository, not an external scientific, security, accessibility, or teaching validation.

## Executive assessment

HaloForge is a notably mature local scientific application. Its strongest qualities are a clear scientific pipeline, unusually explicit provenance and validity boundaries, deterministic numerical work, atomic local persistence, and an interface that teaches causal reasoning instead of treating charts as decoration.

It is appropriate for **local exploratory learning and research workflow prototyping** when its documented limitations are respected. It is **not yet evidence-complete for publication claims, a shared research service, or a production classroom platform**. Those distinctions are deliberate safeguards, not defects to hide.

## What the application does

1. Stages cosmological and numerical inputs without running on every control change.
2. Runs real CLASS/AxiCLASS in an isolated subprocess when the compiled `classy` binding is available; there is no toy-spectrum fallback.
3. Computes sampled linear matter power, deterministic finite-range Simpson integration for `σ(M,z)`, and fit-constrained analytic halo-mass-function views.
4. Saves runs atomically with arrays separated from metadata, integrity checks, provenance, a reproducibility hash, notebook context, audit trail, and export bundles.
5. Supports guided causal experiments, comparisons, numerical diagnostics, teaching materials, accessible chart transcripts, local reports, and explicitly caveated figure exports.

## Evidence reviewed

| Area | Evidence | Assessment |
| --- | --- | --- |
| Numerical core | 185 deterministic tests | Strong local regression coverage; not an independent solver benchmark. |
| Static correctness | Formatting plus Ruff import, statement-structure, and undefined-name checks | Enforced in CI. |
| Application startup | Clean-storage Streamlit health-check and prediction-first UI tests | Confirms the assembled local app starts and protects its first-use teaching gate; it does not exercise a real AxiCLASS solve. |
| Real solver path | Exact pinned AxiCLASS source built in an isolated Python 3.11 environment; HaloForge's CLASS smoke solve and reduced AxiCLASS EDE solve completed | Confirms real solver integration when the compiled binding is present; it is not an independent scientific benchmark or Docker-image verification. |
| Solver isolation | `engine/class_runner.py` subprocess worker, timeout, bounded transient retry | Sound local containment design. |
| Persistence | `state/run_storage.py`, atomic replacement, hashes, import preflight | Strong local durability posture. |
| Privacy boundary | policy and Streamlit UI fail-closed tests | Correctly refuses to represent a server filesystem as private storage. |
| Shipping configuration | `docker compose config` validates; a regression test confirms every first-party app package is copied into the image | Docker daemon was unavailable during this review, so the image build remains unverified here. |
| UX | local first-use browser inspection | Prediction-first experience is coherent; completed-solver reveal still needs browser verification with a local binding configured for the app process. |

## Grading rubric

| Dimension | Grade | Reason |
| --- | --- | --- |
| Scientific software engineering | A- | Explicit units, contracts, fail-closed behavior, reproducibility records, and strong regression tests. |
| Research usability | A- | Runs, comparisons, notebook lineage, exports, and diagnostics are unusually integrated for a local tool. |
| Teaching design | A- | Prediction-first causal narrative, skepticism prompts, and accessible alternatives are excellent. |
| Visual/product craft | A- | Deliberate design system and optional causal motion; full cross-device/device-assistive review is still needed. |
| Production operations | B | Docker and CI are present, but the image was not runnable in this review environment. |
| Collaboration/classroom platform | C | Intentionally absent: no identity, roles, sharing service, classroom accounts, analytics, or grading. |
| External scientific validation | Incomplete | No independent CLASS/CAMB/Colossus/hmf or simulation-reference comparison has been bundled or audited. |

## Release gates before a public claim

- Run the Docker image from a clean machine and archive the build log, AxiCLASS commit, health check, and real solver smoke-test output.
- Initialize or restore the Git repository, verify the intended remote and default branch, then record the resulting commit in the release evidence. This supplied checkout has no `.git` metadata, so local provenance correctly reports the revision as unavailable.
- Add independent reference tables or reproducible comparison scripts for the declared solver/HMF regimes; review numerical tolerances with a domain expert.
- Obtain accessibility and responsive-device review from real users, including keyboard and screen-reader workflows.
- Choose and add a license after confirming ownership and third-party obligations.
- Do not enable hosted collaboration until authentication, per-user storage, authorization, retention/deletion policy, quotas, and security review exist.
- Gather real instructor/learner and researcher feedback before making adoption or learning-effect claims.

## Maintainer commands

```bash
ruff check app.py config content engine state tests
pytest -q
docker compose config
docker compose up --build -d
```

The final command requires a running Docker daemon and should be followed by a real solver smoke test in the app’s Diagnostics view.
