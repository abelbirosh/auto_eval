import types

import pytest

from auto_eval.extraction import (
    FieldMapping,
    GroundTruthSet,
    Origin,
    Outcome,
    PageExtraction,
    QuotedBaseline,
    QuotedExample,
    examples_from_rows,
    extract,
    to_jsonl,
    verify_page,
)
from auto_eval.fetch import Fetched, Rows, hf_dataset_id, html_to_text
from auto_eval.ground_truth import (
    Access,
    ExternalSource,
    Fit,
    GroundTruthReport,
    SourceKind,
)

PAGE = (
    "The 2024 results table. LayoutLMv3 reaches 92.4% field accuracy on the "
    "held-out set, up from 88.1% the year before. Worked example: given the "
    "invoice line 'TOTAL DUE 412.55', the extractor should return 412.55."
)


def rows(**kwargs) -> Rows:
    defaults = dict(
        dataset="acme/invoices",
        config="default",
        split="validation",
        columns=["image_text", "answer", "id"],
        rows=[
            {"image_text": "TOTAL DUE 412.55", "answer": "412.55", "id": 1},
            {"image_text": "TOTAL DUE 9.00", "answer": "9.00", "id": 2},
        ],
        total=2,
    )
    return Rows(**{**defaults, **kwargs})


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
    from auto_eval.ground_truth import Availability

    return GroundTruthReport(
        subject="invoice extractor",
        verdict=Availability.LABELLED_DATA,
        sources=list(sources),
    )


class FakeResponses:
    """Returns a queued object per call, so both model steps can be stubbed."""

    def __init__(self, *parsed):
        self._queue = list(parsed)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self._queue.pop(0) if self._queue else None
        return types.SimpleNamespace(output_parsed=parsed, status="completed", output=[])


class FakeClient:
    def __init__(self, *parsed):
        self.responses = FakeResponses(*parsed)


# --- the fetch layer ------------------------------------------------------


def test_html_is_reduced_to_what_a_reader_would_see():
    text = html_to_text("<p>Accuracy is <b>92.4%</b></p><script>var x = 1;</script>")
    assert "92.4%" in text
    assert "var x" not in text
    assert "<" not in text


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://huggingface.co/datasets/acme/invoices", "acme/invoices"),
        ("https://huggingface.co/datasets/acme/invoices/tree/main", "acme/invoices"),
        ("https://example.org/datasets/acme", None),
        ("https://huggingface.co/models/acme", None),
    ],
)
def test_hugging_face_dataset_urls_are_recognised(url, expected):
    assert hf_dataset_id(url) == expected


@pytest.mark.parametrize(
    "url", ["http://localhost:8000/x", "http://127.0.0.1/x", "http://192.168.1.5/x", "ftp://x.org/y"]
)
def test_private_and_non_http_urls_are_refused_before_any_request(url):
    from auto_eval.fetch import fetch_text

    result = fetch_text(url)
    assert result.ok is False
    assert result.error


# --- cases built from real rows ------------------------------------------


def test_cases_are_built_from_the_rows_not_from_the_model():
    built, why = examples_from_rows(
        rows(),
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer",
                     kpi="field accuracy"),
        source="Acme invoices",
        url="https://huggingface.co/datasets/acme/invoices",
        kpi_names=["field accuracy"],
        limit=10,
    )
    assert why == ""
    assert [e.input for e in built] == ["TOTAL DUE 412.55", "TOTAL DUE 9.00"]
    assert [e.expected for e in built] == ["412.55", "9.00"]
    assert built[0].origin is Origin.DATASET_ROWS
    assert built[0].split == "validation"
    assert built[0].kpi == "field accuracy"


def test_several_input_columns_are_labelled_in_the_case():
    built, _ = examples_from_rows(
        rows(),
        FieldMapping(usable=True, input_columns=["image_text", "id"], expected_column="answer"),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert built[0].input == "image_text: TOTAL DUE 412.55\nid: 1"


def test_a_mapping_naming_a_column_that_does_not_exist_is_refused():
    built, why = examples_from_rows(
        rows(),
        FieldMapping(usable=True, input_columns=["prompt"], expected_column="label"),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert built == []
    assert "columns the dataset does not have" in why


def test_an_unusable_dataset_reports_the_reason():
    built, why = examples_from_rows(
        rows(),
        FieldMapping(usable=False, reason="It is a summarisation set, not extraction."),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert built == []
    assert "summarisation" in why


def test_rows_with_no_answer_are_not_ground_truth():
    built, why = examples_from_rows(
        rows(rows=[{"image_text": "TOTAL DUE 5", "answer": ""}]),
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer"),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert built == []
    assert "missing an input or an answer" in why


def test_structured_cells_keep_their_structure():
    built, _ = examples_from_rows(
        rows(rows=[{"image_text": "x", "answer": {"text": ["412.55"], "start": [4]}}]),
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer"),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert built[0].expected == '{"text": ["412.55"], "start": [4]}'


# --- what a page says, checked against the page ---------------------------


def test_a_quoted_number_is_kept_when_the_page_says_it():
    _, baselines, discarded = verify_page(
        PageExtraction(
            baselines=[
                QuotedBaseline(
                    metric="field accuracy",
                    value="92.4%",
                    system="LayoutLMv3",
                    quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
                )
            ]
        ),
        PAGE, source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert discarded == 0
    assert baselines[0].value == "92.4%"
    assert baselines[0].url == "https://x.org"


def test_an_invented_quote_is_discarded():
    _, baselines, discarded = verify_page(
        PageExtraction(
            baselines=[
                QuotedBaseline(metric="accuracy", value="99%", quote="LayoutLMv3 reaches 99% accuracy")
            ]
        ),
        PAGE, source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert baselines == []
    assert discarded == 1


def test_a_real_quote_cannot_smuggle_a_number_that_is_not_in_it():
    _, baselines, discarded = verify_page(
        PageExtraction(
            baselines=[
                QuotedBaseline(
                    metric="field accuracy",
                    value="97%",  # nowhere in the quote it cites
                    quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
                )
            ]
        ),
        PAGE, source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert baselines == []
    assert discarded == 1


def test_an_example_survives_only_if_both_halves_are_on_the_page():
    examples, _, discarded = verify_page(
        PageExtraction(
            examples=[
                QuotedExample(input="TOTAL DUE 412.55", expected="412.55", kpi="Field Accuracy"),
                QuotedExample(input="TOTAL DUE 412.55", expected="999.00"),
            ]
        ),
        PAGE, source="s", url="https://x.org", kpi_names=["field accuracy"], limit=10,
    )
    assert len(examples) == 1
    assert discarded == 1
    assert examples[0].origin is Origin.PAGE_QUOTE
    assert examples[0].kpi == "field accuracy"  # matched back to the spec's spelling


def test_whitespace_differences_do_not_discard_a_real_quote():
    examples, _, discarded = verify_page(
        PageExtraction(examples=[QuotedExample(input="TOTAL   DUE\n412.55", expected="412.55")]),
        PAGE, source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    assert len(examples) == 1
    assert discarded == 0


# --- the pass over the sources -------------------------------------------


def test_extract_pulls_rows_through_the_dataset_api(full_spec):
    client = FakeClient(
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer",
                     kpi="field accuracy")
    )
    found = extract(
        full_spec,
        report(source()),
        client=client,
        row_fetcher=lambda dataset, limit: rows(),
        text_fetcher=lambda url: pytest.fail("a dataset URL should not be scraped"),
    )

    assert len(found.examples) == 2
    assert found.outcomes[0].outcome is Outcome.EXTRACTED
    assert "validation split" in found.outcomes[0].detail
    # The mapper sees the columns and a sample, never the whole dataset.
    assert "image_text" in client.responses.calls[0]["input"]


def test_extract_reads_a_page_when_the_source_is_not_a_dataset(full_spec):
    client = FakeClient(
        PageExtraction(
            baselines=[
                QuotedBaseline(
                    metric="field accuracy",
                    value="92.4%",
                    quote="LayoutLMv3 reaches 92.4% field accuracy on the held-out set",
                )
            ]
        )
    )
    found = extract(
        full_spec,
        report(source(name="Results page", kind=SourceKind.LEADERBOARD, url="https://x.org/results")),
        client=client,
        text_fetcher=lambda url: Fetched(url=url, ok=True, status=200, text=PAGE),
    )

    assert len(found.baselines) == 1
    assert found.outcomes[0].baselines == 1
    assert 'trust="untrusted"' in client.responses.calls[0]["input"]


def test_an_unreachable_source_is_reported_not_swallowed(full_spec):
    found = extract(
        full_spec,
        report(source(name="Dead link", kind=SourceKind.PUBLISHED_RESULT, url="https://x.org/404")),
        client=FakeClient(),
        text_fetcher=lambda url: Fetched(url=url, ok=False, status=404, error="HTTP 404."),
    )
    assert found.outcomes[0].outcome is Outcome.UNREACHABLE
    assert found.outcomes[0].detail == "HTTP 404."
    assert found.usable is False


def test_a_dataset_the_server_will_not_serve_is_reported(full_spec):
    found = extract(
        full_spec,
        report(source()),
        client=FakeClient(),
        row_fetcher=lambda dataset, limit: Rows(
            dataset=dataset, ok=False, error="Gated dataset; access must be requested."
        ),
    )
    assert found.outcomes[0].outcome is Outcome.UNREACHABLE
    assert "Gated" in found.outcomes[0].detail


def test_contextual_reading_material_is_skipped_without_a_fetch(full_spec):
    found = extract(
        full_spec,
        report(source(name="A blog post", kind=SourceKind.OTHER, url="https://x.org/post",
                      fit=Fit.CONTEXTUAL)),
        client=FakeClient(),
        text_fetcher=lambda url: pytest.fail("nothing to extract here"),
    )
    assert found.outcomes[0].outcome is Outcome.SKIPPED


def test_a_contextual_vendor_page_is_still_read_for_its_numbers(full_spec):
    client = FakeClient(PageExtraction())
    found = extract(
        full_spec,
        report(source(name="Vendor", kind=SourceKind.VENDOR_CLAIM, url="https://x.org/pricing",
                      fit=Fit.CONTEXTUAL)),
        client=client,
        text_fetcher=lambda url: Fetched(url=url, ok=True, text=PAGE),
    )
    assert found.outcomes[0].outcome is Outcome.NOTHING_FOUND
    assert client.responses.calls  # it was read


def test_the_example_cap_is_honoured_and_noted(full_spec):
    client = FakeClient(
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer")
    )
    found = extract(
        full_spec,
        report(source()),
        client=client,
        max_examples=1,
        row_fetcher=lambda dataset, limit: rows(),
    )
    assert len(found.examples) == 1
    assert any("cap" in note for note in found.notes)


def test_to_jsonl_writes_one_case_per_line(full_spec):
    built, _ = examples_from_rows(
        rows(),
        FieldMapping(usable=True, input_columns=["image_text"], expected_column="answer"),
        source="s", url="https://x.org", kpi_names=[], limit=10,
    )
    lines = to_jsonl(built).splitlines()
    assert len(lines) == 2
    import json

    assert json.loads(lines[0])["expected"] == "412.55"


def test_an_empty_set_still_says_what_each_source_did():
    empty = GroundTruthSet(subject="x")
    assert empty.usable is False
