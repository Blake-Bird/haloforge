# HaloForge known limitations

This registry is intentionally visible in the application. A curve being computed, smooth, or internally self-consistent does not erase these limits.

| ID | Area | Severity | Status | Limitation | Impact | Mitigation |
| --- | --- | --- | --- | --- | --- | --- |
| external-reference-benchmarks | Scientific validation | high | open | No independently reproduced external CLASS/CAMB/Colossus/hmf/simulation benchmark tables are published. | Internal numerical checks cannot establish external solver agreement or scientific accuracy. | Treat internal checks as implementation checks only; independently reproduce before research use. |
| ede-hmf-calibration | Empirical HMF | high | open | An HMF fit can be inside its parameter-range contract while still lacking demonstrated calibration for an EDE cosmology. | A predicted EDE halo-abundance shift may not be suitable for publication. | Keep calibration and cosmology-support states separate; compare fits and seek independently justified calibration. |
| finite-k-range | Numerics | medium | open | σ(M) is evaluated over a finite sampled AxiCLASS k range. | Endpoint removal diagnostics cannot reveal omitted power outside the solved range. | Conduct explicit k-range and sampling convergence studies. |
| linear-structure-field | Visualization | medium | open | The field view is a periodic Gaussian linear realization with shared phases, not an N-body simulation or halo catalogue. | Visual density features must not be interpreted as literal formed galaxies or clusters. | Use it only to compare linear amplitude and shape effects; label exported images as illustrative. |
| hosted-collaboration | Privacy and collaboration | high | open | No authenticated hosted workspace, roles, sharing links, classroom roster, or gradebook exists. | The application cannot safely support multi-user research or classroom data sharing. | Use local runs and explicit portable bundles only; do not deploy as an unauthenticated shared service. |
| accessibility-review | Accessibility | medium | open | The app has local light/high-contrast/reduced-motion options and chart transcripts, but no independent assistive-technology or device review. | Accessibility behavior is not yet externally verified. | Obtain keyboard, screen-reader, contrast, and responsive-layout review before release. |
