import types

import pytest

from auto_eval.analysis import (
    AnalysisReport,
    DatasetAssessment,
    Effort,
    PageAssessment,
    QuotedBaseline,
    Reachability,
    Usability,
    analyze_sources,
    choose_split,
    plan_for_dataset,
    verify_baselines,
)
from auto_eval.fetch import (
    DatasetMeta,
    Fetched,
    Split,
    check_url,
    hf_dataset_id,
    html_to_text,
)
from auto_eval.ground_truth import (
    Access,
    Availability,
    ExternalSource,
    Fit,
    GroundTruthReport,
    SourceKind,
)

PAGE = (
    "The 2024 results table. LayoutLMv3 reaches 92.4% field accuracy on the "
    "held-out set, up from 88.1% the year before. The evaluation set is "
    "available on request."
)


def meta(**kwargs) -> DatasetMeta:
    defaults = dict(
        dataset="acme/invoices",
        url="https://huggingface.co/datasets/acme/invoices",
        configs=["default"],
        config="default",
        columns=["image_text", "answer", "id"],
        splits=[Split(name="train", rows=8000), Split(name="validation", rows=1000)],
        licence="cc-by-4.0",
        gated=False,
        downloads=1234,
        description="Invoices with labelled totals.",
    )
    return DatasetMeta(**{**defaults, **kwargs})


def source(**kwargs) -> ExternalSource:
    defaults = dict(
        name="Acme invoices",
        kind=SourceKind.DATASET,
        url="https://huggingface.co/datasets/acme/invoices",
        description="Labelled invoices.",
        covers_kpis=["field accuracy"],
        fit=Fit.ADAPTABLE,
        access=Access.OPEN,
    )
    return ExternalSource(**{**defaults, **kwargs})


def report(*sources) -> GroundTruthReport:
    return GroundTruthReport(
        subject="invoice extractor",
        verdict=Availability.LABELLED_DATA,
        sources=list(sources),
    )


class FakeResponses:
    def __init__(self, *parsed):
        self._queue = list(parsed)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self._queue.pop(0) if self._queue else None
        return types.SimpleNamespace(
            output_parsed=parsed, status="completed", output=[]
        )


class FakeClient:
    def __init__(self, *parsed):
        self.responses = FakeResponses(*parsed)


def usable_assessment(**kwargs) -> DatasetAssessment:
    defaults = dict(
        usable=True,
        input_columns=["image_text"],
        expected_column="answer",
        kpi="field accuracy",
        effort=Effort.LOW,
        summary="Invoice lines with the total as the answer.",
    )
    return DatasetAssessment(**{**defaults, **kwargs})


# --- the guards -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/x",
        "http://127.0.0.1/x",
        "http://192.168.1.5/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://jenkins/job/build",
        "ftp://example.org/x",
        "file:///etc/passwd",
    ],
)
def test_private_and_non_http_urls_are_refused(url):
    allowed, why = check_url(url)
    assert allowed is False
    assert why


def test_a_public_url_is_allowed():
    allowed, why = check_url("https://example.com")
    assert allowed is True
    assert why == ""


def test_html_is_reduced_to_what_a_reader_would_see():
    text = html_to_text("<p>Accuracy is <b>92.4%</b></p><script>var x = 1;</script>")
    assert "92.4%" in text
    assert "var x" not in text


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://huggingface.co/datasets/acme/invoices", "acme/invoices"),
        ("https://huggingface.co/datasets/acme/invoices/tree/main", "acme/invoices"),
        ("https://example.org/datasets/acme", None),
    ],
)
def test_hugging_face_dataset_urls_are_recognised(url, expected):
    assert hf_dataset_id(url) == expected


# --- the plan, built deterministically ------------------------------------


def test_a_held_out_split_is_preferred_over_train():
    assert choose_split(meta()) == "validation"
    assert choose_split(meta(splits=[Split(name="train", rows=10)])) == "train"
    assert choose_split(meta(splits=[])) is None


def test_the_plan_names_what_to_fetch_without_fetching_it():
    plan, why = plan_for_dataset(meta(), usable_assessment())
    assert why == ""
    assert plan.dataset == "acme/invoices"
    assert plan.split == "validation"
    assert plan.rows_available == 1000
    assert plan.input_fields == ["image_text"]
    assert plan.expected_field == "answer"
    assert "1,000 rows from acme/invoices" in plan.what


def test_gating_blocks_a_fetch_but_a_licence_is_only_a_condition_on_it():
    plan, _ = plan_for_dataset(meta(gated=True), usable_assessment())
    assert "gated" in plan.blockers
    assert plan.licence == "cc-by-4.0"

    open_plan, _ = plan_for_dataset(meta(), usable_assessment())
    assert open_plan.blockers is None
    assert open_plan.licence == "cc-by-4.0"


def test_a_mapping_naming_a_column_that_does_not_exist_is_refused():
    plan, why = plan_for_dataset(
        meta(), usable_assessment(input_columns=["prompt"], expected_column="label")
    )
    assert plan is None
    assert "columns the dataset does not have" in why


def test_a_dataset_judged_unusable_yields_no_plan():
    plan, why = plan_for_dataset(meta(), DatasetAssessment(usable=False))
    assert plan is None
    assert "not to hold ground truth" in why


# --- quoted numbers, checked against the page -----------------------------


def test_a_quoted_number_is_kept_when_the_page_says_it():
    kept, discarded = verify_baselines(
        [
            QuotedBaseline(
                metric="field accuracy",
                value="92.4%",
                system="LayoutLMv3",
                quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
            )
        ],
        PAGE,
        source="s",
        url="https://x.org",
    )
    assert discarded == 0
    assert kept[0].value == "92.4%"


def test_an_invented_quote_is_discarded():
    kept, discarded = verify_baselines(
        [
            QuotedBaseline(
                metric="accuracy", value="99%", quote="LayoutLMv3 reaches 99% accuracy"
            )
        ],
        PAGE,
        source="s",
        url="https://x.org",
    )
    assert kept == []
    assert discarded == 1


def test_a_real_quote_cannot_smuggle_a_number_that_is_not_in_it():
    kept, discarded = verify_baselines(
        [
            QuotedBaseline(
                metric="field accuracy",
                value="97%",
                quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
            )
        ],
        PAGE,
        source="s",
        url="https://x.org",
    )
    assert kept == []
    assert discarded == 1


def test_whitespace_differences_do_not_discard_a_real_quote():
    kept, discarded = verify_baselines(
        [
            QuotedBaseline(
                metric="x",
                value="92.4%",
                quote="LayoutLMv3   reaches\n92.4% field accuracy",
            )
        ],
        PAGE,
        source="s",
        url="https://x.org",
    )
    assert len(kept) == 1
    assert discarded == 0


# --- the pass over the sources -------------------------------------------


def test_a_dataset_is_described_from_metadata_and_never_downloaded(full_spec):
    client = FakeClient(usable_assessment())
    seen = {}

    def fake_meta(dataset, url=""):
        seen["dataset"] = dataset
        return meta()

    analysis = analyze_sources(
        full_spec,
        report(source()),
        client=client,
        meta_fetcher=fake_meta,
        text_fetcher=lambda url: pytest.fail(
            "a dataset must not be scraped or downloaded"
        ),
    )

    resource = analysis.resources[0]
    assert seen["dataset"] == "acme/invoices"
    assert resource.reachability is Reachability.OK
    assert resource.usability is Usability.GROUND_TRUTH
    assert resource.dataset.columns == ["image_text", "answer", "id"]
    assert resource.plan.split == "validation"
    assert "metadata only" in resource.detail
    # The model sees columns and splits, never rows.
    assert "image_text" in client.responses.calls[0]["input"]
    assert "validation (1,000 rows)" in client.responses.calls[0]["input"]


def test_the_analysis_carries_a_fetch_plan_for_the_next_step(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(source()),
        client=FakeClient(usable_assessment()),
        meta_fetcher=lambda dataset, url="": meta(),
    )
    assert [p.dataset for p in analysis.plans] == ["acme/invoices"]
    assert analysis.with_ground_truth


def test_a_gated_dataset_reads_as_blocked_but_still_describes_itself(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(source()),
        client=FakeClient(usable_assessment()),
        meta_fetcher=lambda dataset, url="": meta(gated=True),
    )
    resource = analysis.resources[0]
    assert resource.reachability is Reachability.BLOCKED
    assert resource.dataset.columns  # the metadata is still public
    assert "gated" in resource.plan.blockers


def test_a_page_is_read_and_its_numbers_checked(full_spec):
    client = FakeClient(
        PageAssessment(
            contains=Usability.BASELINES,
            summary="A results table for document extraction.",
            kpi="field accuracy",
            baselines=[
                QuotedBaseline(
                    metric="field accuracy",
                    value="92.4%",
                    quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
                )
            ],
            download_hint="The evaluation set, which must be requested from the authors.",
        )
    )
    analysis = analyze_sources(
        full_spec,
        report(
            source(
                name="Results page",
                kind=SourceKind.LEADERBOARD,
                url="https://x.org/results",
            )
        ),
        client=client,
        text_fetcher=lambda url: Fetched(url=url, ok=True, status=200, text=PAGE),
    )

    resource = analysis.resources[0]
    assert resource.usability is Usability.BASELINES
    assert resource.baselines[0].value == "92.4%"
    assert resource.plan.what.startswith("The evaluation set")
    assert 'trust="untrusted"' in client.responses.calls[0]["input"]


def test_a_page_claiming_numbers_it_does_not_have_falls_back_to_background(full_spec):
    client = FakeClient(
        PageAssessment(
            contains=Usability.BASELINES,
            baselines=[
                QuotedBaseline(metric="accuracy", value="99%", quote="we hit 99%")
            ],
        )
    )
    analysis = analyze_sources(
        full_spec,
        report(
            source(name="Page", kind=SourceKind.PUBLISHED_RESULT, url="https://x.org/p")
        ),
        client=client,
        text_fetcher=lambda url: Fetched(url=url, ok=True, text=PAGE),
    )
    assert analysis.resources[0].usability is Usability.BACKGROUND
    assert analysis.resources[0].discarded == 1


def test_a_dead_link_is_reported_not_swallowed(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(
            source(
                name="Dead", kind=SourceKind.PUBLISHED_RESULT, url="https://x.org/404"
            )
        ),
        client=FakeClient(),
        text_fetcher=lambda url: Fetched(
            url=url, ok=False, status=404, error="HTTP 404."
        ),
    )
    assert analysis.resources[0].reachability is Reachability.UNREACHABLE
    assert analysis.resources[0].detail == "HTTP 404."


def test_a_url_our_own_guards_refuse_reads_as_blocked(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(
            source(
                name="Internal",
                kind=SourceKind.PUBLISHED_RESULT,
                url="http://jenkins/x",
            )
        ),
        client=FakeClient(),
        text_fetcher=lambda url: Fetched(
            url=url, ok=False, error="Refused: jenkins is a private or local address."
        ),
    )
    assert analysis.resources[0].reachability is Reachability.BLOCKED


def test_contextual_reading_material_is_not_opened_at_all(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(
            source(
                name="Blog",
                kind=SourceKind.OTHER,
                url="https://x.org/post",
                fit=Fit.CONTEXTUAL,
            )
        ),
        client=FakeClient(),
        text_fetcher=lambda url: pytest.fail("nothing here to open"),
    )
    resource = analysis.resources[0]
    assert resource.reachability is Reachability.NOT_CHECKED
    assert resource.usability is Usability.BACKGROUND


def test_a_dataset_index_that_will_not_describe_it_is_reported(full_spec):
    analysis = analyze_sources(
        full_spec,
        report(source()),
        client=FakeClient(),
        meta_fetcher=lambda dataset, url="": DatasetMeta(
            dataset=dataset, url=url, ok=False, error="No dataset metadata available."
        ),
    )
    assert analysis.resources[0].reachability is Reachability.UNREACHABLE


def test_an_empty_report_analyses_to_nothing(full_spec):
    analysis = analyze_sources(full_spec, report(), client=FakeClient())
    assert analysis == AnalysisReport(subject="invoice extractor")
