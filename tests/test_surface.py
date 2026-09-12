from auto_eval.gaps import analyze
from auto_eval.schema import (
    EvalType,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    Priority,
)
from auto_eval.surface import (
    CellStatus,
    behaviours_for,
    coverage_ratio,
    fill,
    render_gaps,
    surface,
)


class FakeCase:
    """Just enough of a `Case` for the grid: an id, a row, and a column."""

    def __init__(self, case_id, behaviour, eval_types):
        self.id = case_id
        self.behaviour = behaviour
        self.eval_types = eval_types


def columns(matrix):
    return [t.value for t in matrix.eval_types]


# --- the rows --------------------------------------------------------------


def test_behaviours_come_from_in_scope(agent_spec):
    assert behaviours_for(agent_spec) == ["refund requests", "order lookups"]


def test_an_excluded_behaviour_is_not_a_row(agent_spec):
    agent_spec.subject.in_scope.append("phone calls")
    assert "phone calls" not in behaviours_for(agent_spec)


def test_a_spec_with_no_scope_gets_one_honest_row(sparse_spec):
    matrix = surface(sparse_spec)
    assert len(matrix.behaviours) == 1
    assert any("single row" in note for note in matrix.notes)


# --- the columns -----------------------------------------------------------


def test_cost_and_latency_get_no_cases_of_their_own(agent_spec):
    """They are read off runs the suite already makes; a cell would be duplication."""
    agent_spec.eval_types.extend([EvalType.COST, EvalType.PERFORMANCE])
    matrix = surface(agent_spec)
    assert "cost" not in columns(matrix)
    assert "performance" not in columns(matrix)
    assert len(matrix.riders) == 2


def test_an_incident_adds_a_regression_column_even_unasked(agent_spec):
    assert EvalType.REGRESSION not in agent_spec.eval_types
    matrix = surface(agent_spec)
    assert "regression" in columns(matrix)
    assert any("not because it was asked for" in note for note in matrix.notes)


def test_a_regression_cell_asks_for_one_case(agent_spec):
    """Nothing generates them, so asking for five reports a gap nobody can close."""
    matrix = surface(agent_spec)
    regression = [c for c in matrix.cells if c.eval_type is EvalType.REGRESSION]
    assert {c.target for c in regression} == {1}


def test_no_incident_means_no_regression_column(agent_spec):
    agent_spec.evidence = [
        item
        for item in agent_spec.evidence
        if item.kind is not EvidenceKind.FAILURE_CASE
    ]
    assert "regression" not in columns(surface(agent_spec))


def test_an_absent_incident_does_not_add_the_column(agent_spec):
    agent_spec.evidence = [
        EvidenceItem(
            kind=EvidenceKind.FAILURE_CASE,
            reference="something we no longer have",
            status=EvidenceStatus.ABSENT,
        )
    ]
    assert "regression" not in columns(surface(agent_spec))


# --- the targets -----------------------------------------------------------


def test_the_primary_kpis_column_asks_for_more(agent_spec):
    agent_spec.kpis[0].name = "policy violation rate"
    agent_spec.kpis[0].definition = "Share of runs that breach the refund policy."
    agent_spec.kpis[0].priority = Priority.PRIMARY
    matrix = surface(agent_spec, per_cell=5)
    safety = matrix.cell("refund requests", EvalType.SAFETY)
    capability = matrix.cell("refund requests", EvalType.CAPABILITY)
    assert safety.target == 10
    assert capability.target == 5


def test_per_cell_is_honoured(agent_spec):
    assert (
        surface(agent_spec, per_cell=2).cell("order lookups", EvalType.SAFETY).target
        == 2
    )


# --- filling it ------------------------------------------------------------


def test_cases_are_counted_against_their_cell(agent_spec):
    # per_cell=1 and a primary KPI about capability means that cell asks for two.
    matrix = surface(agent_spec, per_cell=1)
    filled = fill(
        matrix,
        [
            FakeCase("a", "refund requests", [EvalType.CAPABILITY]),
            FakeCase("b", "refund requests", [EvalType.CAPABILITY]),
        ],
    )
    cell = filled.cell("refund requests", EvalType.CAPABILITY)
    assert cell.status is CellStatus.COVERED
    assert filled.cell("order lookups", EvalType.CAPABILITY).status is CellStatus.EMPTY


def test_a_case_outside_the_grid_is_reported_not_absorbed(agent_spec):
    """Authoring does not get to edit the claim the grid makes."""
    filled = fill(
        surface(agent_spec), [FakeCase("x", "carrier pigeons", [EvalType.CAPABILITY])]
    )
    assert filled.behaviours == ["refund requests", "order lookups"]
    assert any("outside the grid" in note for note in filled.notes)


def test_a_case_counts_in_every_cell_it_belongs_to(agent_spec):
    filled = fill(
        surface(agent_spec, per_cell=1),
        [FakeCase("a", "refund requests", [EvalType.CAPABILITY, EvalType.SAFETY])],
    )
    assert filled.cell("refund requests", EvalType.CAPABILITY).case_ids == ["a"]
    assert filled.cell("refund requests", EvalType.SAFETY).case_ids == ["a"]


def test_coverage_ratio_does_not_reward_overfilling_one_cell(agent_spec):
    matrix = surface(agent_spec, per_cell=2)
    crowded = [
        FakeCase(str(i), "refund requests", [EvalType.CAPABILITY]) for i in range(20)
    ]
    assert coverage_ratio(fill(matrix, crowded)) < 0.5


def test_gaps_read_as_lines_a_person_can_act_on(agent_spec):
    lines = render_gaps(fill(surface(agent_spec, per_cell=2), []))
    assert lines
    assert all("of" in line and "cases" in line for line in lines)


def test_the_grid_is_deterministic(agent_spec):
    assert surface(analyze(agent_spec)) == surface(analyze(agent_spec))
