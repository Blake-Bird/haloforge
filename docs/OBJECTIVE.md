treat these as checklist and u can not stop till all this is coded then checked reviewed in order and once everything is checked and u think perfect launch it and go through and make sure legit everything is perfected and note i know it says different phases and priotize no u r doing it all at once Even after every prior blocker is fixed, HaloForge still needs to become more than a good cosmology app. To feel genuinely category-defining, it needs to be:

- more trustworthy than a typical research prototype;
- easier than a typical teaching tool;
- more beautiful than a typical data dashboard;
- more rigorous than a typical “interactive simulator”;
- more memorable than a typical course website.

Here is the ruthless second-order list.

## 1. It needs a single unforgettable core experience

Right now the product idea is broad: cosmology, EDE, power spectra, smoothing, halo functions, teaching, visualization, exports. A legendary product needs one sentence people repeat.

Possible stronger cores:

- “See how tiny changes in the infant universe decide what galaxies can exist.”
- “The first serious cosmology laboratory you can understand in ten minutes.”
- “A living derivation from the Big Bang to galaxy clusters.”
- “Change the universe. Watch structure emerge. Prove why.”

Everything must serve that promise. If a page, parameter, graph, animation, or feature does not intensify it, it is clutter.

It needs one canonical moment: a user adjusts a meaningful parameter, predicts the outcome, runs it, and sees an emotionally clear consequence cascade across scales—from primordial spectrum to halo abundance—with a precise explanation of why.

## 2. It needs a world-class onboarding sequence

A new student should not begin in a control panel.

- First launch should open a beautiful 60–90 second guided experiment, not the full research interface.
- Ask one interesting question immediately: “What would happen if the early universe briefly expanded faster?”
- Let users make one safe, meaningful change before introducing jargon.
- Show a pre-run prediction card: “You are increasing early expansion. Predict: fewer or more massive halos?”
- Let them commit to an answer.
- Run a curated fast example.
- Reveal the answer in stages, with a clear causal chain.
- End with: “You just changed the abundance of (10^{14} h^{-1}M\_\odot) halos by X%.”
- Offer three natural next routes: understand it, compare it, or investigate it.
- Never dump 20 controls, 8 pages, and 4 charts on a first-time user.
- Never require a user to know what (\sigma(M)), (P(k)), (k), (h), or EDE means before the app has earned the right to use those terms.

The first five minutes should make someone feel smarter, not behind.

## 3. It needs a true “conceptual zoom” system

The extraordinary version lets the same product work for a curious teenager, an undergraduate, a professor, and a researcher without making four separate apps.

Every major concept should have layers:

- **Intuition:** “Small ripples in the early universe become the seeds of galaxies.”
- **Course level:** equations, units, diagrams, assumptions.
- **Research level:** exact CLASS settings, calibration domain, numerical convergence, citations, and derivation details.
- **Implementation level:** algorithm, numerical method, tests, source provenance.

A user should be able to click any term—(A\_s), (n\_s), transfer function, (\sigma\_8), top-hat filter, mass definition—and climb or descend these layers without losing context.

That is much more powerful than tooltips.

## 4. It needs an exceptional causal-explanation engine

Most scientific software shows outputs. HaloForge should explain causality.

For every parameter change, generate a structured explanation:

- What was changed?
- What is held fixed?
- Which equations are directly affected?
- Which graph moves first?
- Which effects are primary versus downstream?
- Which visual effects are expected?
- Which apparent effects are numerical artifacts or model-dependent?
- Which conclusions are robust?
- Which conclusions depend on fit choice?
- What would falsify the interpretation?

For example:

> Raising (n\_s) increases relative small-scale primordial power. That raises (\sigma(M)) more strongly at lower masses, which increases predicted low-mass halo abundance. The exact abundance shift remains dependent on the HMF calibration and its validity domain.

The app should never say only “curve goes up.”

## 5. It needs to teach scientific skepticism, not just cosmology

The best teaching feature may be teaching users when not to believe the output.

- Add “What could make this wrong?” to every major result.
- Separate observation, theory, numerical approximation, empirical calibration, and visualization.
- Show which parts come from first-principles evolution and which come from simulation-fitted formulas.
- Show confidence/validity states as a core part of the interface.
- Let students deliberately break assumptions and learn why the output becomes unreliable.
- Add “false confidence” examples: a smooth curve can still be invalid.
- Include adversarial exercises: “Which of these two conclusions is unjustified?”
- Add numerical traps: insufficient (k\_{\max}), insufficient resolution, wrong mass definition, fit extrapolation.
- Reward users for identifying uncertainty, not only obtaining a pretty result.

That is the kind of thing professors remember.

## 6. It needs a “lab notebook” that is actually better than a notebook

A saved run should not just be a record. It should become a research object.

Each experiment should support:

- a research question;
- hypothesis before running;
- named baseline;
- parameter diff;
- calculation version;
- scientific validity status;
- notes;
- annotations attached directly to graph regions;
- conclusion;
- caveats;
- citations;
- linked follow-up experiments;
- attached exported artifacts;
- reproducibility hash;
- a compact shareable summary.

Then show experiments as a branching tree:
```text
Planck-like baseline
├── Higher A_s
│   ├── Increase k_max convergence test
│   └── Compare HMF fits
└── EDE candidate
    ├── Change critical epoch
    ├── Test redshift evolution
    └── Compare against ΛCDM baseline
```

That is vastly more intellectually useful than a flat “saved runs” list.

## 7. It needs visual design with restraint, not just polish

To impress Apple-level product designers, it should feel inevitable, calm, spatially coherent, and incredibly deliberate.

- Give every screen one visual focal point.
- Remove decorative motion that does not reveal state or physics.
- Make animation communicate transformation, causality, loading progress, or spatial relationship.
- Use depth sparingly; avoid every panel looking equally bordered and equally important.
- Build a real spacing system, not individual margins tuned until they look okay.
- Use a small semantic color system: action, selection, baseline, comparison, warning, invalid, success.
- Do not reuse cyan for brand, active controls, axis accents, science meaning, and decorative glow simultaneously.
- Give baseline runs a visual identity that remains stable everywhere.
- Make destructive actions visually quiet but difficult to trigger accidentally.
- Make the app look excellent in light mode, dark mode, increased contrast mode, and grayscale.
- Treat typography as information hierarchy: display face only for moments of wonder; extremely readable sans for science; mono only where precision matters.
- Avoid “sci-fi dashboard” clichés: excessive glowing lines, gratuitous grids, faux-terminal labels, ornamental orbit animations.
- Design empty states that invite the next action, not merely announce missing data.
- Make loading states feel alive: explain which stage is running, what will happen next, and how long it usually takes.
- Provide an elegant “calculation interrupted” state preserving prior work and explaining recovery.
- Make the app work beautifully in a narrow laptop window, iPad width, projector view, and mobile read-only view.

## 8. It needs animation that teaches

The animations should be impossible to ignore because they clarify the science.

Potential standout animations:

- A primordial spectrum tilting around its pivot in real time as (n\_s) changes.
- The transfer function appearing as a physical “processing filter” between primordial and matter spectra.
- A moving smoothing sphere showing how mass maps to radius.
- Fourier modes entering and leaving the contribution band for a selected halo mass.
- A density field smoothly evolving from one redshift to another while preserving phase identity.
- A halo-count distribution thinning in the high-mass tail as growth changes.
- A model-validity envelope closing around a graph when the user exceeds calibration bounds.
- A “causal pulse” that travels across the pipeline after a parameter edit.
- An equation that highlights each term as the corresponding visual responds.
- A comparison animation that morphs baseline into candidate rather than swapping static charts.

Rules:

- Motion must be interruptible.
- Motion must respect reduced-motion preferences.
- No animation should delay access to data.
- Nothing should loop forever merely to look futuristic.
- Every animation needs a static equivalent.

## 9. It needs truly great graph interactions

The best graph interaction is not “more controls.” It is insight at the cursor.

- Hovering a point should explain the scientific meaning, not only print x/y values.
- Clicking a mass should synchronize every relevant graph.
- Selecting a (k)-range should highlight the masses most affected.
- Selecting a halo mass should show its corresponding radius and contributing modes.
- Selecting a redshift should update a shared story across all visible plots.
- Brushing a suspicious region should create a note or convergence test.
- Every chart needs a “why this matters” sentence.
- Every chart needs a “what can mislead you” sentence.
- Every chart needs accessible text/table alternatives.
- Every chart should use visual hierarchy to reveal the intended conclusion before the legend is read.
- When there are many curves, default to a summary statistic or small multiples, not a rainbow.
- Add automatic annotations for turning points, extrema, crossing points, validity boundaries, and largest baseline deviations.
- Add “compare at this point” to calculate exact ratios with units and uncertainty context.
- Add a one-click “make this figure publication-ready” flow.
- Add a one-click “make this understandable for students” flow.
- Add a one-click “make this shareable” flow.

## 10. It needs a real uncertainty story

PhD users will immediately ask: “How certain is this?” A serious tool cannot answer only with a curve.

It needs separate treatment of:

- numerical integration error;
- (k)-range truncation;
- grid resolution;
- interpolation behavior;
- CLASS solver settings;
- model uncertainty;
- HMF fit uncertainty;
- extrapolation beyond calibration;
- cosmological parameter uncertainty;
- theory uncertainty;
- user-selected assumption sensitivity;
- emulator uncertainty, if introduced.

The interface should distinguish:

- **computed precisely**
- **numerically converged**
- **physically appropriate**
- **calibrated by simulations**
- **supported for this cosmology**
- **suitable for publication**

Those are not the same thing.

## 11. It needs benchmark modes that researchers trust

The highest compliment would be: “I used this to catch an error in my own pipeline.”

To earn that:

- Publish a benchmark suite with expected outputs.
- Include independently reproduced Planck-like baselines.
- Compare against CLASS, CAMB where relevant, Colossus, `hmf`, and known tables.
- Publish tolerance targets and why they are appropriate.
- Add a “benchmark this run” button.
- Show exact discrepancy by observable and redshift.
- Never hide failed agreement.
- Version benchmark references.
- Add regression visualizations so numerical changes are visible in pull requests.
- Publish performance benchmarks alongside scientific benchmarks.
- Create canonical validation cases for ΛCDM, EDE, curvature, high redshift, low mass, high mass, and edge-of-domain scenarios.
- Maintain a public “known limitations” registry.
- Celebrate failed validation as a useful finding, not a user error.

## 12. It needs research-grade export quality

A published figure or table should never require manual cleanup.

Exports should support:

- SVG, PDF, PNG, CSV, Parquet, JSON, notebook, and a reproducibility bundle;
- color-safe and grayscale versions;
- publication fonts;
- equation-ready labels;
- figures with captions and citations;
- data tables with units in metadata;
- machine-readable provenance;
- exact run configuration;
- validity warnings included in the exported artifact;
- a README explaining each column;
- reproducible code snippets in Python, Julia, R, and Mathematica;
- a one-command recreation script;
- a DOI-ready archival bundle eventually.

Every exported graph should answer: “Could someone understand this six months later without opening the app?”

## 13. It needs a serious collaboration model

Eventually, a professor should be able to create a classroom, and a research group should be able to share work without chaos.

- Private by default.
- Explicit roles: owner, viewer, commenter, editor, instructor, student.
- Immutable shared snapshots.
- Comments tied to exact run, graph, point, and version.
- Fork without mutating the source experiment.
- Compare two branches of an experiment.
- Classroom assignments with hidden solutions.
- Instructor dashboards that show misconception patterns, not surveillance.
- Lab templates that students can fork.
- Anonymous sharing links with expiration and revocation.
- No accidental public runs.
- No ambiguous shared-state behavior.
- Full audit history for research workspaces.

## 14. It needs a professor experience, not just a student experience

For widespread classroom use, the professor workflow has to be absurdly good.

- One-click creation of a lab section.
- Share a locked baseline and permitted parameter ranges.
- Build a sequence of questions around a live experiment.
- See aggregate anonymous answer distributions.
- Identify where students are getting confused.
- Export grades only if explicitly desired.
- Give every lesson estimated completion time.
- Offer lecture mode: large labels, projector-optimized layouts, keyboard controls, narrative slides.
- Provide prepared modules for introductory cosmology, computational physics, statistics, structure formation, and numerical methods.
- Include derivation handouts, assignments, datasets, and solution guides.
- Make the simulations reproducible in a Jupyter notebook.
- Add a “teach this tomorrow” bundle with no setup panic.
- Include accessibility accommodations by design, not as an afterthought.

## 15. It needs a viral layer that does not cheapen the science

The shareable version should produce curiosity, not misinformation.

- “What universe did you make?” visual cards.
- A simple cosmic fingerprint from a parameter change.
- Before/after animated comparisons with one honest takeaway.
- A share card that always includes a caveat when a result is model-dependent.
- Beautiful short explainers such as “Why a 2% tilt change matters.”
- A public gallery of curated, reviewed experiments.
- Featured experiments from professors and researchers.
- “Try to make the rarest clusters” challenges.
- “Can you distinguish these universes from the graph?” quizzes.
- Story mode that produces an elegant vertical video sequence.
- No gamification that implies a cosmology is “better.”
- No misleading claims that a toy visual is a literal universe simulation.
- A distinct visual watermark for pedagogical/illustrative field views.

## 16. It needs an exceptional codebase

To impress high-end engineering teams, the code needs to feel inevitable too.

- Separate the scientific core from UI, storage, orchestration, exports, and visualization.
- Make the scientific core usable as a pure Python library without Streamlit.
- Make every calculation deterministic given a versioned input bundle and seed.
- Replace implicit mutable state with explicit state transitions.
- Use typed, validated parameter models and explicit schemas.
- Use dimensional/unit-aware abstractions where practical.
- Keep numerical code small, composable, independently testable, and documented.
- Centralize all convention conversions.
- Create one source of truth for units, mass definitions, parameter defaults, and citations.
- Add property-based tests for numerical invariants.
- Add metamorphic tests: changing resolution should converge, duplicating input should not alter output, invalid combinations should fail predictably.
- Add fuzz tests for input validation and file import.
- Add golden visual tests for charts and key workflows.
- Add end-to-end tests in real browsers.
- Add performance budgets: initial load, interaction latency, calculation startup, export time, memory ceiling.
- Profile every expensive computation.
- Cache only with explicit keys, validity, schema version, and user ownership.
- Add observability that is privacy-preserving and opt-in.
- Write architecture decision records for every major scientific or storage decision.
- Make failures legible: error taxonomy, actionable remediation, preserved context.
- Keep dependency count low and deliberate.
- Produce a clean developer environment that works in one command.

## 17. It needs “Jane Street” numerical and operational discipline

The biggest difference between a beautiful project and an elite technical product is behavior under stress.

- Every invariant is named and tested.
- Every unit conversion has a test.
- Every boundary case has a reasoned outcome.
- Every external calculation has a timeout, cancellation, retry policy, and clear failure semantics.
- Every save operation is transactional.
- Every schema migration is reversible or recoverable.
- Every cache entry is verifiable and invalidatable.
- Every result records exactly how it was made.
- Every performance optimization has a benchmark proving it helps.
- Every random field has an explicit seed and reproducible normalization.
- Every numerical tolerance is chosen and documented, not guessed.
- Every fallback is visible to the user.
- Every unsupported model combination fails closed, not open.
- Every release includes scientific regression results.
- Every pull request that changes equations, units, defaults, or calibration behavior triggers stricter review.

## 18. It needs an ambitious ML/research layer — but only after trust

If the core becomes correct and respected, the ML layer could be remarkable.

- Parameter-to-observable sensitivity maps.
- Active learning: suggest the next cosmology that most distinguishes competing hypotheses.
- Surrogate models with explicit error estimates and domain boundaries.
- Learned visual summaries that never replace raw data.
- Automated anomaly detection for suspicious numerical runs.
- Explainable clustering of experiment families.
- A “design an experiment” assistant that asks the user’s scientific goal and proposes an interpretable parameter sweep.
- Counterfactual queries: “What smallest change makes cluster abundance differ by 10%?”
- Inverse exploration: “Which parameter combinations preserve (\sigma\_8) while changing small-scale structure?”
- Fast emulation only when its uncertainty is shown next to every prediction.
- Exportable training sets with licenses, provenance, and scientific documentation.
- No black-box conclusion generator that hides domain violations.

## 19. It needs a visual identity with cultural weight

The product needs recognizability—not merely a nice dark theme.

- A distinctive cosmic visual language that does not resemble generic AI dashboards.
- A logo that works at 16px and on a lecture slide.
- A small iconic mark associated with “follow structure across scale.”
- A recognizable baseline/candidate comparison motif.
- A signature animation that teaches a real physical concept.
- A typographic voice that feels rigorous, warm, and not performatively academic.
- A screenshot that makes someone stop scrolling.
- A homepage that shows the actual product experience immediately, not a vague marketing pitch.
- A visual system that can scale into notebooks, papers, lecture slides, social cards, and conference booths.
- An interaction soundless by default, but still emotionally satisfying.

## 20. It needs proof from the outside world

No product becomes universally respected because it says it is excellent.

The final missing component is evidence:

- A physicist who audits the science.
- A computational cosmologist who uses it for a real workflow.
- A professor who teaches a real class with it.
- Students who complete a lab and demonstrably learn more.
- An accessibility reviewer who tests it.
- A designer who critiques the interaction system.
- An external benchmark that confirms numerical agreement.
- Public examples of figures exported into talks, papers, or assignments.
- A small advisory group with names and honest critiques.
- A roadmap shaped by actual failures from real users.
- A public changelog that shows intellectual seriousness.
- Case studies: “Here is what a student misunderstood before HaloForge, and what changed.”
- A publication or workshop demo showing the teaching methodology.
- A community contribution pathway.

The true “mega impressive” version is not one that tries to look like Apple, TikTok, Jane Street, OpenAI, and a physics department simultaneously. It is one that borrows the best standard from each:

- Apple: clarity, calmness, finish.
- Jane Street: correctness, invariants, disciplined systems.
- OpenAI: ambitious research interfaces and legible intelligence.
- TikTok: immediate emotional comprehension and shareability.
- Academia: honesty, provenance, reproducibility, and depth.

If HaloForge can make a first-year student feel awe, a professor feel supported, and a cosmologist feel cautious but impressed, it has become something rare. You’re right: this is not publish-ready yet. It has a promising foundation, but its scientific guardrails, privacy model, product focus, and interaction design are not at the level needed for a serious research/teaching tool. I made no code changes.



\## The five release blockers



1. \*\*Saved work is globally shared on any hosted instance.\*\* Runs, drafts, cache, exports, and “last opened run” are file paths shared by the server process, not scoped to a browser or user. Every new session restores the global last run. “Download complete workspace” can expose everyone’s data. See [run storage]\(/Users/bbird/Downloads/Research/haloforge-main/state/run\_storage.py:21) and [session initialization]\(/Users/bbird/Downloads/Research/haloforge-main/state/session.py:119).



2. \*\*The repository is unsafe to publish as-is.\*\* This folder has no \`.git\` repository or \`.gitignore\`, and it already contains 162 data artifacts: saved runs, exports, ZIPs, cache, and state. The Docker build copies the entire folder into the image. A future GitHub push could publish private research outputs and produce a bloated image.



3. \*\*The HMF is scientifically misleading whenever the smoothing window is not a real-space top-hat.\*\* The app lets users choose Gaussian or sharp-k smoothing while retaining the top-hat mass–radius mapping. A sharp-k HMF needs a separately calibrated mass assignment; it is not simply interchangeable with top-hat. The UI must either lock HMF calculations to top-hat or label alternate windows explicitly as exploratory variance views, not halo predictions.



4. \*\*Some HMF fit assumptions are hidden or violated.\*\* The app offers FOF and spherical-overdensity fits in one menu with one generic “Δ” control, without requiring the relevant mass definition, reference density, halo finder, calibration cosmology, or validity range. For example, Tinker 2008 is explicitly calibrated across spherical-overdensity definitions—not a generic universal HMF knob. [Tinker et al.](https://arxiv.org/abs/0803.2706)



5. \*\*The “EDE-aware” HMF path is incomplete.\*\* For fits that use \\(\Omega\_m(z)\\), the app reconstructs the expansion history with radiation + matter + curvature + a \\(\Lambda\\)-like closure, omitting the time-dependent EDE contribution. That makes the Watson SO branch especially suspect in EDE runs. This must be sourced from the AxiCLASS background, or the combination must be disabled and marked unsupported.



\## Scientific audit: what must change



- Create a machine-readable “scientific contract” for each calculation: input convention, units, equation, source, implementation status, validity domain, and test reference.
- Replace the single broad HMF menu with fit cards:
  - analytic: Press–Schechter, Sheth–Tormen;
  - FOF-calibrated;
  - SO-calibrated relative to mean density;
  - SO-calibrated relative to critical density.
- Require a mass-definition selector such as \`M200m\`, \`M200c\`, \`Mvir\`, or FOF b=0.2. Do not infer this from the name “Δ.”
- Make invalid combinations impossible, not merely warned about.
- Replace free-form EDE field text with validated advanced parameters, known presets, provenance, and an “expert override” disclosure.
- Pull EDE-dependent background quantities directly from AxiCLASS, including \\(H(z)\\), \\(\Omega\_m(z)\\), and field energy density.
- Add a proper validity engine: per-fit warnings for redshift, mass, \\(\ln \sigma^{-1}\\), overdensity, cosmology, mass definition, and extrapolation.
- Add explicit “linear theory only” banners. A structure slice is not an N-body simulation, and an HMF fit is not a direct prediction for arbitrary EDE cosmologies.
- Include massive neutrinos only when the mass definition, transfer functions, and cold-vs-total matter convention are defined carefully; otherwise state that they are out of scope.
- Do not present \`N\_eff\` as the full neutrino sector.
- Test baseline numerical values against an independent reference implementation, such as \`hmf\`, Colossus, or a frozen trusted table—not just positivity and array shape.
- Add golden tests for \\(P(k)\\), \\(\sigma\_8\\), \\(\sigma(M)\\), \\(d\ln\sigma/d\ln M\\), HMF, cumulative HMF, units, and every supported fit.
- Add automated convergence tests over \\(k\_\min\\), \\(k\_\max\\), k sampling, mass-grid density, redshift sampling, and numerical derivatives.
- Establish acceptance tolerances before visual polish: e.g. “baseline \\(\sigma\_8\\) agrees with CLASS to X% after declared finite-range truncation.”
- Turn the current raw \`kR\` coverage JSON into an interpreted convergence report with pass/warn/fail states and suggested corrections.
- Do not silently floor HMF values to \\(10^{-300}\\); visually plausible curves can conceal underflow, invalid inputs, and nonphysical extrapolation.
- Add provenance to every exported number: app version, Git commit, image digest, AxiCLASS commit, package versions, exact parameter schema version, timestamp, and numerical settings.
- Put a citation panel directly beside each fit and export BibTeX/CITATION.cff. AxiCLASS itself should be cited and linked to its parameter documentation. [AxiCLASS reference configuration](https://github.com/PoulinV/AxiCLASS/blob/master/explanatory.ini)



\## Privacy, saving, GitHub, and deployment redesign



The correct product decision is: \*\*browser-private by default; exports are explicit; cloud sharing is opt-in.\*\*



- Store drafts, run metadata, and modest result arrays in browser storage, namespaced by a random installation ID.
- Never auto-upload results.
- For large arrays, offer “download run bundle” or IndexedDB storage with an understandable quota meter.
- Make “Save run” a deliberate action, with an autosave draft indicator—not a surprise permanent server write on every compute.
- If you support a hosted version, require authenticated accounts and per-user storage boundaries. Never use a shared \`./data\` directory as the user data model.
- Delete “download complete workspace” from shared deployments; retain it only for a private desktop/local mode.
- Add import/export that includes schema migration, provenance, integrity hash, and clear overwrite/merge choices.
- Separate application data from the cloned repository entirely. A local install should use a user-data location, not \`repo/data\`.
- Add \`.gitignore\` for data, cache, exports, temporary files, virtual environments, test artifacts, and local secrets.
- Provide a committed empty data directory only if truly necessary, with \`.gitkeep\`, never real outputs.
- Add a visible privacy statement: “This run stays in this browser/device unless you download or share it.”
- Add a server-mode banner: “This deployment stores data on the server” if browser-private storage is not implemented.
- Run the container as a non-root user; add health, resource, timeout, queue, disk quota, cleanup, and retention behavior.
- Add a minimal GitHub Actions workflow: lint, unit tests, scientific reference tests, container build, dependency/security scan.
- Add \`LICENSE\`, \`CITATION.cff\`, changelog, contribution guide, security policy, issue templates, release tags, and a reproducibility policy.



\## UI/UX: the current product is too dense



The visual language is stylish, but it is “cosmology control panel” rather than effortless premium software. The fixed 370px sidebar, eight destinations, nested expanders, long controls, chart settings, and competing visual decorations make a first session cognitively expensive.



- Replace the eight-page navigation with three primary modes:
  1. \*\*Explore\*\* — guided, immediate, story-led.
  2. \*\*Compare\*\* — deliberate experiment design and evidence.
  3. \*\*Research\*\* — advanced settings, exports, diagnostics.
- Put “Run” at the center of the workflow, not at the end of a long sidebar.
- Use progressive disclosure: beginner controls first; “Advanced physics,” “Numerics,” and “Model assumptions” second.
- Keep the run’s current state persistently visible: name, baseline, draft/complete status, model status, and calculation age.
- Separate physical parameters, numerical resolution, and empirical-model choices. They are currently visually equivalent, even though they carry radically different meaning.
- Replace the raw four quick presets with a curated experiment library:
  - Planck-like ΛCDM;
  - EDE comparison;
  - “More small-scale power”;
  - “Why \\(k\_\max\\) matters”;
  - “Where HMF fits disagree.”
- Every preset needs a one-sentence question, an expected visual change, citations, and a reset path.
- Add undo/redo, reset-this-section, duplicate experiment, compare-to-baseline, and “explain this change.”
- Use visible parameter constraints and dependent-control behavior. \`Ωb ≥ Ωm\` should be prevented at entry, not reported later.
- The \`Omega\_m\` slider increment is bizarrely precise (\`0.000109883...\`); it reads like an implementation artifact, not intentional scientific UX.
- Rename “Run & auto-save” to a clearer primary action based on mode: “Calculate,” then separately show the save state.
- Don’t hide explanations behind hover-only cards; keyboard, touch, and screen-reader users cannot reliably access them.
- Stop treating every page title as a marketing headline. The product needs quieter hierarchy, shorter copy, and stronger task context.
- Add light mode, reduced motion, high contrast, keyboard navigation, responsive tablet/mobile layouts, and tested focus states.
- Google-hosted fonts create an offline/privacy dependency. Bundle fonts or use resilient system fallbacks.



Apple’s guidance is directly relevant here: don’t encode meaning with color alone, support increased contrast, and provide alternate labels for chart data. [Apple color guidance](https://developer.apple.com/design/Human-Interface-Guidelines/color) and [chart guidance](https://developer.apple.com/design/human-interface-guidelines/charts?changes=_8)



\## Graphs: how to make them Jane Street-grade



- Every chart should answer one named question. A dashboard with four large graphs is not automatically insight.
- Make “Baseline versus candidate” the core view: one overlay, one ratio/residual, one plain-language conclusion.
- Direct-label the most important lines; legends should not be the decoding mechanism.
- Limit multi-series plots by default. Five redshifts × five fits × three windows is analysis noise.
- Use a stable visual encoding:
  - color = cosmology/run;
  - line style = model/fit;
  - panel = redshift;
  - never change that contract.
- Include an uncertainty/validity band or shaded extrapolation region, not a generic warning below the chart.
- Make logarithmic axes visibly explicit, including the implications of ratios near zero.
- Provide a “show sampled points” toggle so users can distinguish calculation resolution from interpolated visual smoothness.
- Include an “inspect point” readout: physical mass, corresponding radius, \\(k\\)-band contribution, \\(\sigma\\), \\(\nu\\), fit validity, and units.
- Provide print/PDF-ready figure presets with concise titles, complete captions, units, annotations, citations, and accessible alt text.
- Replace manual per-chart “Graph Studio” axis work with saved figure recipes: “paper,” “lecture,” “comparison,” “social.”
- Build a chart-quality test suite: color-blind simulation, contrast, clipped legends, axis overlap, no-data, one-data-point, and 36-series stress states.



\## Teaching: become memorable without becoming gimmicky



The best viral version is a clean 30-second insight loop; the best college version is a rigorous lab. They should be two layers of the same engine.



- Start Explore with one visual question: “What determines how many cluster-scale halos exist?”
- Use a guided causal chain where changing one parameter highlights only the affected stages.
- Add “predict first” prompts before rendering results.
- Explain why a change happens, not merely that the curve moved.
- Use semantic scaffolding: primordial conditions → transfer physics → smoothing → variance → collapse → calibration limits.
- Make misconceptions explicit:
  - \\(\sigma(M)\\) is not a measured halo catalog;
  - an HMF fit is not universal;
  - the density field is not a simulated universe;
  - EDE effects are not just an amplitude slider.
- Add lab modules with learning objectives, time estimate, prerequisites, instructor notes, checkpoints, downloadable notebooks, and answer keys.
- Add a shareable “experiment card”: one question, one parameter delta, one annotated result, one caveat, one reproducible link/bundle.
- Add “explain in three levels”: intuitive, undergraduate, researcher.
- Build an accessible transcript for every interactive chart.
- Design teaching outcomes around falsifiable claims and validation, not “play with sliders.”



\## Feature opportunities worth prioritizing



1. Experiment timeline with named decisions, parameter diffs, notes, and reproducibility hashes.
2. Scientific validity dashboard, before calculation and beside every chart.
3. Publication export package with provenance manifest and citation bundle.
4. Browser-private run vault plus explicit portable bundles.
5. Guided “what changed?” analysis generated from actual curve deltas.
6. Expert compare mode with fit/mass-definition compatibility matrix.
7. Sensitivity explorer: one-at-a-time parameter sweeps, derivative plots, and interaction warnings.
8. Convergence lab: visualize whether a conclusion survives numerical choices.
9. Emulator or nonlinear-theory mode only after strict domain labeling. Tools such as Dark Emulator demonstrate the value of simulation-calibrated HMF/clustering predictions, but only within their supported cosmology space. [Dark Emulator docs](https://dark-emulator.readthedocs.io/en/stable/)
10. Collaborative sharing only later, with explicit permissions and immutable reproducible snapshots.



\## Build order



\*\*Phase 0 — stop unsafe publication\*\*

- Remove existing data artifacts from the distributable project.
- Add ignore rules, license, citations, security basics, and CI.
- Redesign storage ownership and remove shared-workspace download/restore from hosted mode.



\*\*Phase 1 — scientific trust\*\*

- Lock HMF to valid top-hat behavior.
- Implement mass-definition and fit-validity enforcement.
- Correct EDE background-dependent quantities.
- Add independent-reference and convergence tests.



\*\*Phase 2 — product clarity\*\*

- Rebuild navigation around Explore / Compare / Research.
- Make one excellent baseline-to-candidate workflow.
- Redo graph grammar, accessibility, and responsive behavior.



\*\*Phase 3 — teaching and sharing\*\*

- Build guided experiments, lab content, and reproducible share cards.
- Add advanced research exports and optional authenticated collaboration.



The current test suite passes—29 small tests—but it mostly verifies plumbing, array shape, positivity, and basic storage behavior. It does not yet establish numerical correctness, HMF calibration validity, EDE consistency, privacy isolation, deployment safety, or visual accessibility.
