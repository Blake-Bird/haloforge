"""Searchable, keyboard-friendly navigation records for the application shell.

The index intentionally contains metadata only.  Loading a saved run remains a
deliberate UI action, so searching cannot mutate the active scientific state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Command:
    """One discoverable destination or safe quick action."""

    identifier: str
    title: str
    detail: str
    kind: str
    primary_mode: str
    workspace: str | None = None
    run_id: str | None = None


_DESTINATIONS = (
    (
        "explore",
        "Explore guided experiment",
        "A 60–90 second prediction-led first experiment",
        "Explore",
        None,
    ),
    (
        "compare",
        "Compare saved runs",
        "Ratios, differences, and uncertainty context",
        "Compare",
        "Compare lab",
    ),
    (
        "evolution",
        "Evolution studio",
        "Follow the linear spectrum through redshift",
        "Compare",
        "Evolution studio",
    ),
    (
        "sensitivity",
        "Sensitivity explorer",
        "Controlled parameter-to-observable comparisons",
        "Compare",
        "Sensitivity explorer",
    ),
    (
        "dashboard",
        "Research dashboard",
        "Calculate and inspect the active universe",
        "Research",
        "Dashboard",
    ),
    (
        "graphs",
        "Graph studio",
        "Power, variance, halo abundance, and figure recipes",
        "Research",
        "Graph studio",
    ),
    (
        "structure",
        "Structure field",
        "Phase-preserving linear density-field view",
        "Research",
        "Structure field",
    ),
    (
        "fits",
        "Fit and window atlas",
        "Mass definitions, fitting relations, and smoothing",
        "Research",
        "Fit + window atlas",
    ),
    (
        "design",
        "Design experiment",
        "Plan one interpretable parameter change",
        "Research",
        "Design experiment",
    ),
    (
        "teach",
        "Teaching lab",
        "Exercises, numerical traps, and uncertainty practice",
        "Research",
        "Teach",
    ),
    (
        "pipeline",
        "Learn the pipeline",
        "From primordial perturbations to halo abundance",
        "Research",
        "Learn the pipeline",
    ),
    (
        "notebook",
        "Research notebook",
        "Questions, annotations, follow-ups, and artifacts",
        "Research",
        "Notebook",
    ),
    (
        "runs",
        "Saved runs and exports",
        "Load, export, import, or recover local work",
        "Research",
        "Runs + export",
    ),
    (
        "diagnostics",
        "Diagnostics",
        "Validity, convergence, and runtime health",
        "Research",
        "Diagnostics",
    ),
)

_FIGURES = (
    "Primordial and matter power spectrum",
    "Mass variance σ(M)",
    "Halo mass function",
    "Cumulative halo abundance",
    "Density-field slice",
    "Redshift evolution",
)

_EQUATIONS = (
    "Primordial spectrum Pᵣ(k)",
    "Matter spectrum P(k, z)",
    "Top-hat smoothing window W(kR)",
    "Variance σ²(M)",
    "Halo mass function dn/dlnM",
)


def _normalise(value: object) -> str:
    return " ".join(str(value).casefold().replace("σ", "sigma").split())


def _score(query: str, command: Command) -> tuple[int, str]:
    haystack = _normalise(" ".join((command.title, command.detail, command.kind)))
    if not query:
        return (2, _normalise(command.title))
    if haystack.startswith(query):
        return (0, _normalise(command.title))
    if query in haystack:
        return (1, _normalise(command.title))
    return (3, _normalise(command.title))


def command_index(
    *,
    concepts: Iterable[object] = (),
    runs: Iterable[Mapping[str, object]] = (),
) -> list[dict]:
    """Build serializable commands from local, already-loaded metadata.

    ``concepts`` accepts labels or objects with a ``label`` field.  Run records
    are deliberately reduced to their names and IDs; numerical arrays never
    enter the command palette.
    """
    records = [
        Command(key, title, detail, "destination", mode, workspace)
        for key, title, detail, mode, workspace in _DESTINATIONS
    ]
    records.extend(
        Command(
            f"figure:{title}",
            title,
            "Open in Graph studio",
            "figure",
            "Research",
            "Graph studio",
        )
        for title in _FIGURES
    )
    records.extend(
        Command(
            f"equation:{title}",
            title,
            "Open with the learning pipeline",
            "equation",
            "Research",
            "Learn the pipeline",
        )
        for title in _EQUATIONS
    )
    for concept in concepts:
        label = getattr(concept, "label", concept)
        if not isinstance(label, str) or not label.strip():
            continue
        records.append(
            Command(
                f"concept:{label}",
                label,
                "Concept explanation at four depths",
                "concept",
                "Research",
                "Teach",
            )
        )
    for run in runs:
        run_id = run.get("run_id")
        if not run_id:
            continue
        name = str(run.get("name") or "Untitled run")
        records.append(
            Command(
                f"run:{run_id}",
                name,
                "Load this saved local run",
                "run",
                "Research",
                "Dashboard",
                str(run_id),
            )
        )
        if run.get("exports"):
            records.append(
                Command(
                    f"export:{run_id}",
                    f"Export — {name}",
                    "Open this run's export controls",
                    "export",
                    "Research",
                    "Runs + export",
                    str(run_id),
                )
            )
    return [asdict(record) for record in records]


def search_commands(
    query: object, commands: Iterable[Mapping[str, object]], *, limit: int = 8
) -> list[dict]:
    """Return stable, bounded substring matches suitable for immediate display."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    needle = _normalise(query)
    matches: list[Command] = []
    for raw in commands:
        try:
            command = Command(**dict(raw))
        except (TypeError, ValueError):
            continue
        if not needle or needle in _normalise(
            " ".join((command.title, command.detail, command.kind))
        ):
            matches.append(command)
    # An empty palette is a short list of intentional next steps, not an
    # alphabetized search dump.  ``command_index`` puts the three primary
    # modes first, followed by their immediately useful labs.
    if needle:
        matches.sort(key=lambda item: _score(needle, item))
    return [asdict(item) for item in matches[:limit]]
