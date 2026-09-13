import pytest

from auto_eval.cohort import (
    Cohort,
    CohortError,
    Endpoint,
    Price,
    System,
    SystemKind,
    baseline_system,
    load,
    parse,
    with_baseline,
)

COHORT = """{
  "name": "search", "task": "factual lookup", "model": "gpt-5-mini",
  "systems": [
    {"label": "Beta", "configuration": "GET /search",
     "endpoint": {"method": "GET", "url": "https://beta.test/search", "params": {"q": "{query}"}}},
    {"label": "Alpha", "vendor": "Alpha", "configuration": "POST /search",
     "endpoint": {"url": "https://alpha.test/search", "headers": {"x-api-key": "${ALPHA_KEY}"},
                  "body": {"query": "{query}"}},
     "price": {"per_call_usd": 0.005, "note": "$5 / 1k"}}
  ]
}"""


def test_a_cohort_reads_back_with_its_systems():
    cohort = parse(COHORT)
    assert cohort.name == "search" and cohort.model == "gpt-5-mini"
    assert {s.label for s in cohort.systems} == {"Alpha", "Beta"}


def test_a_bare_list_of_systems_is_a_cohort_too():
    cohort = parse('[{"label": "A", "kind": "model_only"}]')
    assert cohort.systems[0].kind is SystemKind.MODEL_ONLY


def test_rows_are_alphabetical_and_not_by_anything_else():
    """No column decides the order, because no column decides the answer."""
    assert [s.label for s in parse(COHORT).ordered()] == ["Alpha", "Beta"]


def test_an_empty_or_unreadable_cohort_is_refused():
    with pytest.raises(CohortError, match="nothing to compare"):
        parse('{"systems": []}')
    with pytest.raises(CohortError, match="Not readable JSON"):
        parse("{nope")


def test_the_keys_an_endpoint_needs_are_named_not_stored():
    endpoint = parse(COHORT).systems[1].endpoint
    assert endpoint.secrets() == ["ALPHA_KEY"]
    assert (
        "${ALPHA_KEY}" in endpoint.headers["x-api-key"]
    )  # the reference, not the value


def test_a_system_missing_its_key_is_not_runnable(monkeypatch):
    monkeypatch.delenv("ALPHA_KEY", raising=False)
    alpha = next(s for s in parse(COHORT).systems if s.label == "Alpha")
    ok, why = alpha.readiness()
    assert not ok and "ALPHA_KEY" in why


def test_a_system_with_its_key_set_is_runnable(monkeypatch):
    monkeypatch.setenv("ALPHA_KEY", "sk-test")
    alpha = next(s for s in parse(COHORT).systems if s.label == "Alpha")
    assert alpha.readiness() == (True, "")


def test_runnable_splits_the_cohort_in_two(monkeypatch):
    monkeypatch.delenv("ALPHA_KEY", raising=False)
    ready, blocked = parse(COHORT).runnable()
    assert [s.label for s in ready] == ["Beta"]
    assert [s.label for s, _ in blocked] == ["Alpha"]


def test_a_row_that_needs_an_endpoint_and_has_none_is_not_runnable():
    ok, why = System(label="X", kind=SystemKind.SEARCH_API).readiness()
    assert not ok and "no endpoint" in why


def test_the_baseline_is_added_once_and_only_once():
    cohort = with_baseline(parse(COHORT))
    assert cohort.baseline is not None
    assert with_baseline(cohort).systems == cohort.systems


def test_the_baseline_calls_a_model_and_no_endpoint():
    baseline = baseline_system("gpt-5")
    assert baseline.calls_a_model and not baseline.needs_endpoint
    assert baseline.readiness() == (True, "")


def test_a_price_says_whether_it_was_stated_at_all():
    assert not Price().stated
    assert Price(per_call_usd=0.001).stated


def test_loading_from_a_file_names_the_cohort(tmp_path):
    path = tmp_path / "vendors.json"
    path.write_text(COHORT.replace('"name": "search", ', ""), encoding="utf-8")
    assert load(path).name == "vendors"


def test_a_missing_cohort_file_says_so(tmp_path):
    with pytest.raises(CohortError, match="No cohort at"):
        load(tmp_path / "nope.json")


def test_an_endpoint_declares_its_own_defaults():
    endpoint = Endpoint(url="https://x.test/s")
    assert endpoint.method == "POST" and endpoint.results_path == "results"
    assert endpoint.fields["snippet"] == "text"


def test_a_cohort_with_no_model_rows_needs_no_model():
    cohort = Cohort(
        systems=[System(label="A", endpoint=Endpoint(url="https://a.test/s"))]
    )
    assert not any(s.calls_a_model for s in cohort.systems)
