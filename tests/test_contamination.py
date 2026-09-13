from datetime import date

from auto_eval.contamination import (
    CLEAN,
    CUTOFFS,
    FRESH,
    Freshness,
    cutoff_for,
    for_suite,
    fresh_by_verdict,
    matching,
    parse_cutoff,
    verdict_for,
)
from auto_eval.suite import build

# --- the cutoffs ----------------------------------------------------------


def test_a_dated_snapshot_resolves_to_its_family():
    assert cutoff_for("gpt-5-mini-2025-08-07").model == "gpt-5-mini"
    assert cutoff_for("GPT-5").cutoff == date(2024, 9, 30)


def test_the_longest_match_wins_so_mini_is_not_read_as_gpt_5():
    assert cutoff_for("gpt-5-mini").cutoff == date(2024, 5, 31)


def test_an_unknown_model_has_no_cutoff_rather_than_a_guessed_one():
    """A wrong cutoff turns a contaminated suite into a clean-looking one."""
    assert cutoff_for("some-other-model") is None


def test_every_cutoff_on_file_says_where_it_was_read_and_when():
    for entry in CUTOFFS.values():
        assert entry.source.startswith("https://")
        assert entry.read_on >= entry.cutoff


def test_a_cutoff_can_be_supplied_by_hand():
    supplied = parse_cutoff("llama-4", "2025-02-01")
    assert supplied.cutoff == date(2025, 2, 1)
    assert "command line" in supplied.source


# --- the catalogue --------------------------------------------------------


def test_every_fresh_source_carries_a_link_a_date_and_a_caveat():
    for source in FRESH:
        assert source.url.startswith("https://")
        assert source.released.year >= 2024
        assert source.caveats and source.staying_clean


def test_held_out_and_rolling_sources_are_clean_whatever_the_cutoff():
    for source in FRESH:
        if source.protection in (Freshness.HELD_OUT, Freshness.ROLLING):
            assert verdict_for(source, None) in CLEAN


def test_a_dated_source_is_judged_against_the_cutoff():
    livebench = next(s for s in FRESH if s.name == "LiveBench")
    dated = livebench.model_copy(update={"protection": Freshness.POST_CUTOFF})
    assert (
        verdict_for(dated, cutoff_for("gpt-5-mini")) is Freshness.POST_CUTOFF
    )  # 2024-06 > 2024-05
    assert (
        verdict_for(dated, cutoff_for("gpt-5")) is Freshness.PRE_CUTOFF
    )  # 2024-06 < 2024-09


def test_without_a_cutoff_a_dated_source_is_unknown_not_clean():
    livebench = next(s for s in FRESH if s.name == "LiveBench")
    dated = livebench.model_copy(update={"protection": Freshness.POST_CUTOFF})
    assert verdict_for(dated, None) is Freshness.UNKNOWN


def test_the_catalogue_is_ordered_cleanest_first():
    verdicts = [verdict for _, verdict in fresh_by_verdict(cutoff_for("gpt-5"))]
    assert verdicts[0] is Freshness.HELD_OUT


def test_matching_finds_the_coding_suites_for_a_coding_spec(full_spec):
    full_spec.subject.description = "Writes Python code and fixes failing unit tests."
    names = [source.name for source in matching(full_spec)]
    assert "LiveCodeBench" in names


# --- a whole suite --------------------------------------------------------


def test_a_suite_of_its_own_cases_cannot_be_contaminated(agent_spec):
    suite = build(agent_spec)
    report = for_suite(suite, "gpt-5-mini")
    assert report.verdict is Freshness.HELD_OUT
    assert report.clean_cases == len(suite.cases)
    assert report.at_risk_cases == 0
    assert report.cutoff == date(2024, 5, 31)


def test_an_unknown_model_warns_rather_than_reporting_clean(agent_spec):
    report = for_suite(build(agent_spec), "mystery-model-v3")
    assert report.cutoff is None
    assert any("--cutoff" in warning for warning in report.warnings)


def test_a_case_adapted_from_an_undated_public_suite_is_flagged(agent_spec):
    suite = build(agent_spec)
    adapted = suite.cases[0].model_copy(
        update={
            "source": suite.cases[0].source.__class__.ADAPTED,
            "source_url": "https://example.org/suite",
        }
    )
    suite = suite.model_copy(update={"cases": [adapted, *suite.cases[1:]]})

    report = for_suite(suite, "gpt-5-mini")
    assert report.unknown_cases == 1
    assert report.verdict is Freshness.UNKNOWN
    assert any("no date on file" in warning for warning in report.warnings)


def test_a_case_adapted_from_a_rolling_suite_is_recognised(agent_spec):
    suite = build(agent_spec)
    adapted = suite.cases[0].model_copy(
        update={
            "source": suite.cases[0].source.__class__.ADAPTED,
            "source_url": "https://livecodebench.github.io/",
        }
    )
    suite = suite.model_copy(update={"cases": [adapted, *suite.cases[1:]]})

    report = for_suite(suite, "gpt-5-mini")
    origins = {risk.source: risk for risk in report.by_origin}
    assert origins["LiveCodeBench"].verdict is Freshness.ROLLING
    assert report.clean_cases == len(suite.cases)
