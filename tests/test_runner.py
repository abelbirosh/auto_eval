import json

import pytest

from auto_eval.contamination import parse_cutoff
from auto_eval.provider import MockClient
from auto_eval.runner import (
    CaseStatus,
    RunError,
    SampleRun,
    _case_status,
    list_runs,
    load_run,
    load_trace,
    run_suite,
    write_run,
)
from auto_eval.suite import Split, build
from auto_eval.trace import StopReason, Usage
from auto_eval.verify import CaseResult, VerifierKind


def _sample(status: CaseStatus, sample: int = 0) -> SampleRun:
    return SampleRun(
        case_id="c1",
        sample=sample,
        status=status,
        stop_reason=StopReason.TERMINAL,
        result=CaseResult(
            case_id="c1", sample=sample, passed=status is CaseStatus.PASSED
        ),
        usage=Usage(),
    )


@pytest.fixture
def suite(agent_spec):
    return build(agent_spec)


@pytest.fixture
def run(suite):
    return run_suite(suite, mock=True, samples=1, client=MockClient())


# --- running --------------------------------------------------------------


def test_every_case_is_run_and_accounted_for(suite, run):
    assert len(run.report.cases) == len(suite.cases)
    assert len(run.traces) == len(suite.cases)
    assert run.report.overall.runs == len(suite.cases)


def test_samples_multiply_the_runs_not_the_cases(suite):
    run = run_suite(suite, mock=True, samples=3, client=MockClient())
    assert run.report.overall.cases == len(suite.cases)
    assert run.report.overall.runs == len(suite.cases) * 3
    assert all(len(case.runs) == 3 for case in run.report.cases)


def test_only_the_chosen_split_is_run(suite):
    run = run_suite(
        suite, mock=True, samples=1, split=Split.HELD_OUT, client=MockClient()
    )
    assert {case.split for case in run.report.cases} == {Split.HELD_OUT}
    assert run.report.dev.cases == 0
    assert run.report.split == "held_out"


def test_a_limit_runs_the_first_n_cases(suite):
    run = run_suite(suite, mock=True, samples=1, limit=3, client=MockClient())
    assert len(run.report.cases) == 3
    assert run.report.limit == 3


def test_an_empty_selection_is_refused_rather_than_reported_as_a_pass(suite):
    empty = suite.model_copy(update={"held_out": []})
    with pytest.raises(RunError, match="Nothing to run"):
        run_suite(empty, mock=True, split=Split.HELD_OUT, client=MockClient())


def test_the_same_suite_and_client_produce_the_same_run(suite):
    first = run_suite(suite, mock=True, samples=1, client=MockClient())
    second = run_suite(suite, mock=True, samples=1, client=MockClient())
    assert [c.status for c in first.report.cases] == [
        c.status for c in second.report.cases
    ]
    assert first.report.overall.checks.passed == second.report.overall.checks.passed


# --- what the numbers mean ------------------------------------------------


def test_the_headline_is_the_held_out_split(suite, run):
    held = [c for c in run.report.cases if c.split is Split.HELD_OUT]
    assert run.report.headline.cases == len(held)
    assert run.report.headline.passed == sum(c.passed for c in held)


def test_blocked_checks_are_named_rather_than_dropped(run):
    """Dropping them would turn 'we could not check' into 'it was fine'."""
    assert run.report.overall.checks.blocked > 0
    kinds = {b.kind for case in run.report.cases for b in case.blocked}
    assert VerifierKind.NO_ESCAPE in kinds
    assert all(b.reason for case in run.report.cases for b in case.blocked)


def test_a_blocked_check_is_counted_in_neither_direction(run):
    checks = run.report.overall.checks
    assert checks.settled == checks.passed + checks.failed
    assert checks.total == checks.settled + checks.blocked


def test_a_case_with_nothing_runnable_is_blocked_not_passed(suite):
    stripped = suite.model_copy(
        update={
            "cases": [
                case.model_copy(
                    update={
                        "verifiers": [
                            v
                            for v in case.verifiers
                            if v.kind is VerifierKind.END_STATE
                        ]
                    }
                )
                for case in suite.cases
            ]
        }
    )
    run = run_suite(stripped, mock=True, samples=1, client=MockClient())
    assert {c.status for c in run.report.cases} == {CaseStatus.BLOCKED}
    assert run.report.headline.pass_rate is None
    assert any("no runnable check" in w for w in run.report.warnings)


def test_a_provider_failure_is_an_error_not_a_failure(suite):
    class Broken(MockClient):
        def create(self, **kwargs):
            raise RuntimeError("502 upstream")

    run = run_suite(suite, mock=True, samples=1, limit=2, client=Broken())
    assert {c.status for c in run.report.cases} == {CaseStatus.ERROR}
    assert run.report.overall.passed == 0 and run.report.overall.failed == 0
    assert any("never completed" in w for w in run.report.warnings)


def test_a_flaky_case_is_a_failing_case():
    """One sample in three passing is not a pass; it is an unreliable agent."""
    runs = [
        _sample(CaseStatus.PASSED),
        _sample(CaseStatus.PASSED, sample=1),
        _sample(CaseStatus.FAILED, sample=2),
    ]
    assert _case_status(runs) is CaseStatus.FAILED


def test_a_case_that_errored_on_every_sample_is_an_error():
    assert _case_status([_sample(CaseStatus.ERROR)]) is CaseStatus.ERROR


def test_one_completed_sample_is_enough_to_judge_a_case_on():
    runs = [_sample(CaseStatus.ERROR), _sample(CaseStatus.PASSED, sample=1)]
    assert _case_status(runs) is CaseStatus.PASSED


def test_every_kpi_is_reported_even_when_nothing_settled(suite, run):
    assert [group.label for group in run.report.by_kpi] == suite.kpis


def test_checks_are_grouped_by_kind_with_their_blocked_count(run):
    by_kind = {group.label: group.checks for group in run.report.by_kind}
    assert by_kind["no_escape"].blocked and not by_kind["no_escape"].settled
    assert by_kind["terminated"].settled


def test_a_mock_run_says_so_before_anything_else(run):
    assert "called no provider" in run.report.warnings[0]


def test_the_suite_s_own_warnings_are_carried_into_the_run(suite, run):
    for warning in suite.warnings:
        assert warning in run.report.warnings


def test_a_judge_that_is_the_model_under_test_is_called_out(suite):
    run = run_suite(
        suite,
        mock=True,
        samples=1,
        limit=2,
        model="gpt-5-mini",
        judge_model="gpt-5-mini",
        client=MockClient(),
    )
    assert any("self-preference bias" in w for w in run.report.warnings)


def test_skipping_the_judge_leaves_the_rubrics_unsettled(suite):
    run = run_suite(suite, mock=True, samples=1, judge=False, client=MockClient())
    assert run.report.judge_model is None
    assert any("No judge ran" in w for w in run.report.warnings)
    blocked = [b for case in run.report.cases for b in case.blocked]
    assert any(b.kind is VerifierKind.RUBRIC for b in blocked)


def test_a_tampered_suite_is_reported_as_untraceable(suite):
    tampered = suite.model_copy(update={"cases": suite.cases[:3]})
    run = run_suite(tampered, mock=True, samples=1, client=MockClient())
    assert run.report.suite_verified is False
    assert any("no longer hashes" in w for w in run.report.warnings)


def test_the_contamination_verdict_rides_along(suite):
    run = run_suite(
        suite,
        mock=True,
        samples=1,
        limit=2,
        model="mystery-model",
        cutoff=parse_cutoff("mystery-model", "2025-01-01"),
        client=MockClient(),
    )
    assert str(run.report.contamination.cutoff) == "2025-01-01"


def test_progress_is_reported_as_it_goes(suite):
    seen = []
    run_suite(
        suite,
        mock=True,
        samples=1,
        limit=4,
        client=MockClient(),
        progress=lambda done, total, case_id: seen.append((done, total)),
    )
    assert seen[-1] == (4, 4)


# --- writing it down ------------------------------------------------------


def test_a_run_survives_the_round_trip(suite, run, tmp_path):
    target = write_run(run, tmp_path)
    read_back = load_run(target)
    assert read_back.run_id == run.report.run_id
    assert read_back.headline.pass_rate == run.report.headline.pass_rate
    assert len(read_back.cases) == len(run.report.cases)


def test_the_trace_behind_a_verdict_is_kept(suite, run, tmp_path):
    target = write_run(run, tmp_path)
    case = run.report.cases[0]
    trace = load_trace(target, case.case_id, 0)
    assert trace.case_id == case.case_id
    assert trace.instruction == case.instruction
    assert trace.system  # the prompt as it was sent


def test_a_missing_trace_says_so(run, tmp_path):
    target = write_run(run, tmp_path)
    with pytest.raises(RunError, match="No trace"):
        load_trace(target, "nosuchcase", 0)


def test_the_run_list_is_newest_first(suite, tmp_path):
    for _ in range(2):
        written = run_suite(suite, mock=True, samples=1, limit=1, client=MockClient())
        write_run(written, tmp_path)
    entries = list_runs(tmp_path)
    assert entries == sorted(entries, key=lambda e: e.started_at, reverse=True)
    assert all(entry.mock for entry in entries)


def test_a_directory_with_no_runs_is_empty_not_an_error(tmp_path):
    assert list_runs(tmp_path / "nothing-here") == []


def test_an_unreadable_run_is_skipped_rather_than_breaking_the_list(tmp_path):
    broken = tmp_path / "half-written"
    broken.mkdir()
    (broken / "run.json").write_text("{not json", encoding="utf-8")
    assert list_runs(tmp_path) == []


def test_rates_survive_serialisation(run, tmp_path):
    """The dashboard reads these off the JSON; a property would not travel."""
    raw = json.loads(
        (write_run(run, tmp_path) / "run.json").read_text(encoding="utf-8")
    )
    assert "pass_rate" in raw["headline"]
    assert "blocked" in raw["overall"]["checks"] and "total" in raw["overall"]["checks"]
    assert "tokens" in raw["usage"]
