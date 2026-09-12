"""What should be tested, laid out as a grid before anything is written.

A suite assembled case by case is a suite nobody can audit: twelve cases that
felt worth writing, with no way to say what they cover or what they miss. So
the coverage is decided first - behaviours down the side, reasons for testing
across the top - and authoring fills cells. A cell that stays empty is printed
as a gap, the same way `auto_eval.gaps` prints a question, rather than quietly
not existing.

Two things are deliberately not rows here:

* **Cost and latency** get no cases of their own. They are read off every run
  the suite already makes, so a cell for them would be cases that duplicate
  the capability cases at greater expense.
* **Comparative** gets none either. Comparing two agents is running the same
  suite twice, not writing different cases.

Both still appear in the report, as riders on the grid rather than rows in it.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field

from .schema import EvalType, EvidenceKind, EvidenceStatus, Priority, TaskSpec

# Eval types that imply cases of their own.
CELL_TYPES = [
    EvalType.CAPABILITY,
    EvalType.REGRESSION,
    EvalType.ROBUSTNESS,
    EvalType.SAFETY,
    EvalType.COMPLIANCE,
]

# Eval types measured on the cases the grid already holds.
RIDER_TYPES = {
    EvalType.PERFORMANCE: "read off every run in the suite; no cases of its own",
    EvalType.COST: "read off every run in the suite; no cases of its own",
    EvalType.COMPARATIVE: "run the whole suite against each candidate; no cases of its own",
}

# Cases per cell. Below five, a cell cannot distinguish a real difference from
# one bad run: at n=5 a single failure moves the cell's rate by 20 points, so
# treat anything thinner than this as an anecdote rather than a measurement.
DEFAULT_PER_CELL = 5

# The primary KPI is what the task succeeds or fails on, so its cells carry
# more weight than the ones that only have to not regress.
PRIMARY_MULTIPLIER = 2


class CellStatus(str, Enum):
    COVERED = "covered"  # at target
    THIN = "thin"  # has cases, fewer than asked for
    EMPTY = "empty"  # nothing was written for it


class Cell(BaseModel):
    """One behaviour, tested for one reason."""

    behaviour: str
    eval_type: EvalType
    target: int = Field(description="Cases this cell asked for.")
    case_ids: List[str] = Field(default_factory=list)

    @property
    def status(self) -> CellStatus:
        if not self.case_ids:
            return CellStatus.EMPTY
        return (
            CellStatus.COVERED if len(self.case_ids) >= self.target else CellStatus.THIN
        )


class CoverageMatrix(BaseModel):
    """The grid a suite is answerable to."""

    subject: str
    behaviours: List[str]
    eval_types: List[EvalType]
    cells: List[Cell]
    riders: List[str] = Field(
        default_factory=list,
        description="KPIs measured on every case rather than in a cell of their own.",
    )
    notes: List[str] = Field(default_factory=list)

    def cell(self, behaviour: str, eval_type: EvalType) -> Optional[Cell]:
        return next(
            (
                c
                for c in self.cells
                if c.behaviour == behaviour and c.eval_type is eval_type
            ),
            None,
        )

    @property
    def gaps(self) -> List[Cell]:
        return [c for c in self.cells if c.status is not CellStatus.COVERED]

    @property
    def target_total(self) -> int:
        return sum(c.target for c in self.cells)


def behaviours_for(spec: TaskSpec) -> List[str]:
    """What the suite has to exercise, in the user's words where they gave them.

    `out_of_scope` is honoured literally: a behaviour the user excluded does not
    come back as a row because a KPI happens to mention it.
    """
    excluded = {item.strip().lower() for item in spec.subject.out_of_scope}
    rows = [
        item.strip()
        for item in spec.subject.in_scope
        if item.strip() and item.strip().lower() not in excluded
    ]
    if rows:
        return list(dict.fromkeys(rows))
    # Nothing said. One row, named after the job the thing exists to do - a
    # thin grid that is honest about being thin beats a wide one we invented.
    return [f"the job {spec.subject.name or 'the agent'} exists to do"]


def _has_failures(spec: TaskSpec) -> bool:
    return any(
        item.kind is EvidenceKind.FAILURE_CASE
        and item.status is not EvidenceStatus.ABSENT
        for item in spec.evidence
    )


def _eval_types(spec: TaskSpec) -> List[EvalType]:
    """Columns of the grid: what the user asked for, plus what they handed us.

    A spec that carries an incident gets a regression column whether or not it
    asked for one. The case is going to be written - `auto_eval.authoring`
    harvests every failure it is given - and a case with no cell to land in is a
    case that goes uncounted.
    """
    chosen = [t for t in CELL_TYPES if t in spec.eval_types]
    if _has_failures(spec) and EvalType.REGRESSION not in chosen:
        chosen.append(EvalType.REGRESSION)
    return chosen or [EvalType.CAPABILITY]


def _riders(spec: TaskSpec) -> List[str]:
    out = [
        f"{kind.value}: {why}"
        for kind, why in RIDER_TYPES.items()
        if kind in spec.eval_types
    ]
    return out


def _primary_types(spec: TaskSpec) -> set:
    """Eval types the primary KPI is about, inferred from how it is measured.

    Crude on purpose: the spec does not tie a KPI to an eval type, and guessing
    with a model here would make the grid non-deterministic for no gain.
    """
    primary = [k for k in spec.kpis if k.priority is Priority.PRIMARY]
    if not primary:
        return set()
    wording = " ".join(f"{k.name} {k.definition}".lower() for k in primary)
    types = set()
    if any(word in wording for word in ("violat", "harm", "unsafe", "policy", "toxic")):
        types.add(EvalType.SAFETY)
    if any(word in wording for word in ("format", "schema", "valid")):
        types.add(EvalType.COMPLIANCE)
    if any(word in wording for word in ("malformed", "adversarial", "messy", "robust")):
        types.add(EvalType.ROBUSTNESS)
    return types or {EvalType.CAPABILITY}


def _target(eval_type: EvalType, per_cell: int, weighted: set) -> int:
    """How many cases a cell asks for.

    Regression is the exception: those cases are harvested, never generated, so
    asking for five of them would report a permanent gap that nothing but the
    user's memory can close. It asks for one.
    """
    if eval_type is EvalType.REGRESSION:
        return 1
    return per_cell * (PRIMARY_MULTIPLIER if eval_type in weighted else 1)


def surface(spec: TaskSpec, *, per_cell: int = DEFAULT_PER_CELL) -> CoverageMatrix:
    """The grid this spec implies. Deterministic; no model, no network."""
    rows = behaviours_for(spec)
    columns = _eval_types(spec)
    weighted = _primary_types(spec)

    cells = [
        Cell(
            behaviour=behaviour,
            eval_type=eval_type,
            target=_target(eval_type, per_cell, weighted),
        )
        for behaviour in rows
        for eval_type in columns
    ]

    notes: List[str] = []
    if not spec.subject.in_scope:
        notes.append(
            "The spec named no behaviours, so the grid has a single row standing in for the whole job. "
            "Listing what the agent is actually for is the cheapest way to make this suite mean something."
        )
    if spec.subject.out_of_scope:
        notes.append(
            "Excluded by the spec, and deliberately untested: "
            + ", ".join(spec.subject.out_of_scope)
            + "."
        )
    if EvalType.REGRESSION in columns:
        notes.append(
            "Regression cells are filled from failures you have already seen; nothing is generated for them, "
            "so they stay thin until you share more incidents."
        )
    if EvalType.REGRESSION in columns and EvalType.REGRESSION not in spec.eval_types:
        notes.append(
            "The regression column was added because the request came with an incident in it, not because it was asked for."
        )

    return CoverageMatrix(
        subject=spec.subject.name or "the agent",
        behaviours=rows,
        eval_types=columns,
        cells=cells,
        riders=_riders(spec),
        notes=notes,
    )


def fill(matrix: CoverageMatrix, cases: Iterable) -> CoverageMatrix:
    """Return a copy of the grid with each case counted against its cell.

    A case that names a behaviour the grid does not have is counted nowhere and
    reported as unplaced, rather than being silently added as a row: the grid is
    the claim about coverage, and authoring does not get to edit the claim.
    """
    filled = matrix.model_copy(deep=True)
    unplaced: List[str] = []

    for case in cases:
        placed = False
        for eval_type in getattr(case, "eval_types", []):
            cell = filled.cell(case.behaviour, eval_type)
            if cell is not None:
                cell.case_ids.append(case.id)
                placed = True
        if not placed:
            unplaced.append(case.id)

    if unplaced:
        filled.notes.append(
            f"{len(unplaced)} case(s) fall outside the grid and are not counted towards coverage."
        )
    return filled


def render_gaps(matrix: CoverageMatrix) -> List[str]:
    """One line per cell that did not get what it asked for."""
    lines = []
    for cell in matrix.gaps:
        have = len(cell.case_ids)
        lines.append(
            f"{cell.behaviour} x {cell.eval_type.value}: {have} of {cell.target} cases ({cell.status.value})"
        )
    return lines


def coverage_ratio(matrix: CoverageMatrix) -> float:
    """Share of the asked-for cases that exist. 1.0 only when nothing is thin."""
    target = matrix.target_total
    if not target:
        return 0.0
    have = sum(min(len(c.case_ids), c.target) for c in matrix.cells)
    return have / target


__all__ = [
    "CELL_TYPES",
    "DEFAULT_PER_CELL",
    "PRIMARY_MULTIPLIER",
    "RIDER_TYPES",
    "Cell",
    "CellStatus",
    "CoverageMatrix",
    "behaviours_for",
    "coverage_ratio",
    "fill",
    "render_gaps",
    "surface",
]
