# Security policy

## Supported releases

HaloForge is not yet a public hosted service. The supported configuration is a local, single-user installation. Do not put research runs, access tokens, or personal data in an issue.

## Reporting a vulnerability

Until a dedicated security contact is published, report a suspected vulnerability privately to the repository owner. Include the affected revision, reproduction steps, expected impact, and whether any user data could cross a storage boundary. Please do not open a public issue before the owner has had a reasonable opportunity to assess and fix it.

## Storage boundary

The app refuses `HALOFORGE_DEPLOYMENT=hosted` because its filesystem vault is not authenticated or per-user. This is an intentional fail-closed safeguard, not hosted multi-user support.
