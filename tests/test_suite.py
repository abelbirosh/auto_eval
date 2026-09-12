import pytest

from auto_eval.agent import RunMode
from auto_eval.authoring import CaseSource
from auto_eval.schema import EvalType, SubjectKind
from auto_eval.suite import (
    Split,
    SuiteError,
    build,
    kpi_coverage,
    load,
    verify_digest,
    write,
)

# --- building --------------------------------------------------------------


def test_a_suite_is_built_offline_and_covers_the_grid(agent_spec):
    suite = build(agent_spec)
    assert suite.cases
    assert suite.name == "support-agent"
    assert suite.profile.run_mode is RunMode.HTTP
    assert verify_digest(suite) is True


def test_the_same_spec_produces_the_same_suite(agent_spec):
    """A suite that drifts makes every comparison it is used for meaningless."""
    first, second = build(agent_spec), build(agent_spec)
    assert first.digest == second.digest
    assert first.model_dump_json() == second.model_dump_json()


def test_changing_the_spec_changes_the_digest(agent_spec):
    before = build(agent_spec).digest
    agent_spec.subject.in_scope.append("address changes")
    assert build(agent_spec).digest != before


def test_editing_a_suite_breaks_its_digest(agent_spec):
    suite = build(agent_spec)
    tampered = suite.model_copy(update={"cases": suite.cases[:-1]})
    assert verify_digest(tampered) is False


def test_a_run_is_cases_times_samples(agent_spec):
    suite = build(agent_spec)
    assert suite.samples == 3
    assert suite.total_runs == len(suite.cases) * 3


# --- the gate --------------------------------------------------------------


def test_an_agent_nobody_can_start_is_refused(agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = []
    with pytest.raises(SuiteError, match="Not ready to author"):
        build(agent_spec)


def test_force_writes_it_anyway_and_says_so(agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = []
    suite = build(agent_spec, force=True)
    assert any("past a closed gate" in note for note in suite.notes)


def test_something_without_a_trajectory_is_refused(full_spec):
    full_spec.subject.kind = SubjectKind.CLASSIFIER
    with pytest.raises(SuiteError):
        build(full_spec)


# --- the split -------------------------------------------------------------


def test_regression_cases_are_always_held_out(agent_spec):
    """A known failure used for tuning stops being evidence of anything."""
    suite = build(agent_spec)
    regression = [c for c in suite.cases if EvalType.REGRESSION in c.eval_types]
    assert regression
    assert all(suite.split_of(c.id) is Split.HELD_OUT for c in regression)


def test_the_split_survives_a_rebuild(agent_spec):
    first, second = build(agent_spec), build(agent_spec)
    assert first.held_out == second.held_out


def test_both_sides_of_the_split_have_cases(agent_spec):
    suite = build(agent_spec)
    assert suite.cases_in(Split.DEV)
    assert suite.cases_in(Split.HELD_OUT)
    assert len(suite.cases_in(Split.DEV)) < len(suite.cases_in(Split.HELD_OUT))


# --- what the suite admits about itself ------------------------------------


def test_it_says_when_it_is_mostly_invented(agent_spec):
    agent_spec.evidence = []
    suite = build(agent_spec)
    assert any("generated from the spec" in warning for warning in suite.warnings)


def test_it_says_the_fixtures_do_not_exist_yet(agent_spec):
    assert any(
        "starting state that does not exist" in w for w in build(agent_spec).warnings
    )


def test_it_says_when_it_cannot_start_the_agent(agent_spec):
    agent_spec.subject.interface = None
    suite = build(agent_spec)
    assert suite.profile.run_mode is RunMode.TRANSCRIPT
    assert any("scores recorded traces" in warning for warning in suite.warnings)


def test_it_says_when_no_tools_were_found(sparse_spec):
    suite = build(sparse_spec, force=True)
    assert any("No tools were read off" in warning for warning in suite.warnings)


def test_a_kpi_nothing_checks_is_listed_as_unmeasured(agent_spec):
    suite = build(agent_spec)
    coverage = kpi_coverage(suite)
    assert set(coverage) == set(suite.kpis)
    assert coverage["task success rate"]


def test_the_source_mix_is_on_the_face_of_it(agent_spec):
    suite = build(agent_spec)
    assert suite.by_source(CaseSource.HARVESTED)
    assert any("Sources:" in note for note in suite.notes)


# --- writing and reading back ---------------------------------------------


def test_a_written_suite_reads_back_identically(agent_spec, tmp_path):
    suite = build(agent_spec)
    written = write(suite, tmp_path, document="# doc")
    assert (tmp_path / "suite.json").exists()
    assert (tmp_path / "suite.md").read_text().startswith("# doc")
    assert any("fixture(s) still have to be built" in line for line in written)
    assert load(tmp_path) == suite


def test_it_loads_from_the_file_as_well_as_the_directory(agent_spec, tmp_path):
    suite = build(agent_spec)
    write(suite, tmp_path)
    assert load(tmp_path / "suite.json") == suite


def test_a_missing_suite_says_how_to_make_one(tmp_path):
    with pytest.raises(SuiteError, match="auto-eval author"):
        load(tmp_path)


def test_a_corrupt_suite_is_refused_rather_than_half_read(tmp_path):
    (tmp_path / "suite.json").write_text("{not json")
    with pytest.raises(SuiteError, match="not a readable suite"):
        load(tmp_path)
