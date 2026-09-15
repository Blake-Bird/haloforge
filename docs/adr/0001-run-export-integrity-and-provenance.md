# ADR 0001: Run export integrity and provenance are separate from scientific validity

**Status:** Accepted  
**Date:** 2026-09-14  
**Decision area:** Durable local storage, reproducibility bundles, and research-table exports

## Context

HaloForge persists numerical arrays, submitted parameters, solver settings, notebook context, validity claims, figures, and tabular data. A corrupted or mismatched array payload must not hydrate an active analysis session. At the same time, a complete checksum manifest or reproducibility hash must not be presented as evidence that a cosmological conclusion is physically appropriate, calibrated by simulations, or publication-ready.

CSV files are broadly portable but cannot reliably carry field-level metadata. Parquet supports schema metadata, but detached files still need a human-readable companion that explains each column and its scientific scope.

## Decision

1. Save run arrays as atomic NPZ payloads and metadata as atomic JSON documents.
2. Store an SHA-256 digest of the saved array payload and independently recompute the declared calculation identity on load. A failed check is fail-closed for session hydration; legacy records remain readable but are visibly unverified.
3. Export a bundle manifest with hashes and byte counts for generated artifacts before ZIP compression.
4. Put field units, definitions, run identity, reproducibility identity, and links to provenance/validity/integrity/citation records into Parquet schema metadata.
5. Export `data_dictionary.json` from those same table definitions so CSV and detached-table users receive a field-by-field guide without a second source of truth.
6. Preserve scientific validity as a separate, versioned artifact. Integrity, provenance, and reproducibility records must explicitly state that they do not establish fit calibration, physical appropriateness, external agreement, or publication suitability.

## Consequences

- Interrupted writes are recoverable through atomic replacement rather than partial final files.
- A modified payload cannot silently become a completed result in the active session.
- A researcher can inspect table units and supporting records without reopening the application.
- Bundle size and export time increase because each artifact is retained and hashed.
- Checksums establish file identity and transport integrity only. Independent numerical benchmarks, calibration evidence, accessibility review, and scientific review remain separate work.

## Evidence and implementation

- Storage and export implementation: `state/run_storage.py`
- Run/document migrations: `state/schema.py`
- Calculation provenance: `state/provenance.py`
- Scientific validity artifact: `engine/validity.py`
- Export workflow and metadata checks: `tests/test_run_workflow.py`

## Revisit when

- Introducing authenticated multi-user storage, remote sharing, or encrypted-at-rest key management.
- Changing the calculation identity schema, cryptographic algorithm, table schemas, or archive layout.
- Adding externally validated benchmark references that need their own signed/versioned provenance record.
