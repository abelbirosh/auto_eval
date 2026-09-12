from auto_eval.gaps import analyze
from auto_eval.render import render_markdown, render_questions
from auto_eval.schema import Readiness


def test_full_spec_renders_every_section(full_spec):
    md = render_markdown(analyze(full_spec))
    for heading in (
        "## 1. What we're testing",
        "## 2. How success is judged",
        "## 3. Evidence",
        "## 4. Constraints",
        "## 5. Assumptions",
        "## 6. Open questions",
    ):
        assert heading in md
    assert "# Invoice extractor accuracy" in md
    assert "field accuracy" in md
    assert ">= 95%" in md
    assert "docs/extractor.md" in md
    assert "No open questions." in md
    assert f"`{Readiness.READY.value}`" in md


def test_sparse_spec_renders_without_kpis_or_evidence(sparse_spec):
    md = render_markdown(analyze(sparse_spec))
    assert "_No KPIs identified" in md
    assert "_Nothing supplied" in md
    assert "_not stated_" in md
    assert "**[blocking]**" in md


def test_questions_view_numbers_and_flags(sparse_spec):
    text = render_questions(analyze(sparse_spec))
    assert text.startswith("1. ")
    assert "**[blocking]**" in text
    assert "`subject.interface`" in text


def test_ground_truth_line_reflects_availability(full_spec, sparse_spec):
    assert "available — Hand-labelled invoices (n=400)" in render_markdown(full_spec)
    assert "**Ground truth:** none identified" in render_markdown(sparse_spec)


def test_unit_is_not_concatenated_onto_a_target_that_already_carries_it(full_spec):
    full_spec.kpis[0].unit = "%"
    full_spec.kpis[0].target = ">= 95%"
    assert ">= 95% %" not in render_markdown(full_spec)


def test_ground_truth_report_renders_sources_and_verdict(full_spec):
    from auto_eval.ground_truth import (
        Access,
        BaselineValue,
        ExternalSource,
        Fit,
        SourceFindings,
        SourceKind,
        assess,
    )
    from auto_eval.render import render_ground_truth

    report = assess(
        full_spec,
        SourceFindings(
            sources=[
                ExternalSource(
                    name="DocVQA",
                    kind=SourceKind.DATASET,
                    url="https://example.org/docvqa",
                    description="Labelled document QA set.",
                    covers_kpis=["field accuracy"],
                    fit=Fit.ADAPTABLE,
                    access=Access.OPEN,
                    baselines=[BaselineValue(metric="ANLS", value="0.87", system="LayoutLMv3")],
                    caveats="Forms, not invoices.",
                )
            ],
            recommendation="Start from DocVQA, relabel 100 invoices.",
        ),
        searches=["invoice extraction benchmark"],
    )
    md = render_ground_truth(report)

    assert "# Ground truth for invoice extractor" in md
    assert "`labelled_data`" in md
    assert "https://example.org/docvqa" in md
    assert "| ANLS | 0.87 | LayoutLMv3 |" in md
    assert "Caveat: Forms, not invoices." in md
    assert "invoice extraction benchmark" in md
    assert "for comparison, not scoring" in md


def test_an_empty_report_says_so_rather_than_rendering_blank(sparse_spec):
    from auto_eval.ground_truth import SourceFindings, assess
    from auto_eval.render import render_ground_truth

    md = render_ground_truth(assess(sparse_spec, SourceFindings(recommendation="")))
    assert "_Nothing found online for this task._" in md
    assert "_No recommendation returned._" in md
    assert "`none_found`" in md


def _analysis():
    from auto_eval.analysis import (
        AnalysisReport,
        CitedBaseline,
        DownloadPlan,
        Effort,
        Reachability,
        ResourceAnalysis,
        Usability,
    )
    from auto_eval.fetch import DatasetMeta, Split
    from auto_eval.ground_truth import SourceKind

    return AnalysisReport(
        subject="invoice extractor",
        resources=[
            ResourceAnalysis(
                source="Acme invoices",
                url="https://huggingface.co/datasets/acme/invoices",
                kind=SourceKind.DATASET,
                reachability=Reachability.OK,
                detail="3 column(s), 2 split(s), read from metadata only.",
                usability=Usability.GROUND_TRUTH,
                summary="Invoice lines with the total as the answer.",
                covers_kpis=["field accuracy"],
                effort=Effort.LOW,
                dataset=DatasetMeta(
                    dataset="acme/invoices",
                    url="https://huggingface.co/datasets/acme/invoices",
                    columns=["image_text", "answer"],
                    splits=[Split(name="validation", rows=1000)],
                    licence="cc-by-4.0",
                    gated=False,
                    downloads=1234,
                ),
                plan=DownloadPlan(
                    what="1,000 rows from acme/invoices (default/validation), image_text -> answer",
                    url="https://huggingface.co/datasets/acme/invoices",
                    dataset="acme/invoices",
                    split="validation",
                    rows_available=1000,
                    licence="cc-by-4.0",
                ),
            ),
            ResourceAnalysis(
                source="Dead link",
                url="https://x.org/404",
                kind=SourceKind.PUBLISHED_RESULT,
                reachability=Reachability.UNREACHABLE,
                detail="HTTP 404.",
                usability=Usability.UNUSABLE,
            ),
            ResourceAnalysis(
                source="Results page",
                url="https://x.org/results",
                kind=SourceKind.LEADERBOARD,
                reachability=Reachability.OK,
                usability=Usability.BASELINES,
                baselines=[
                    CitedBaseline(
                        metric="field accuracy",
                        value="92.4%",
                        system="LayoutLMv3",
                        quote="LayoutLMv3 reaches 92.4% field accuracy",
                        source="Results page",
                        url="https://x.org/results",
                    )
                ],
                discarded=1,
            ),
        ],
        notes=["1 source(s) can be fetched as they are; the plans say what to get."],
    )


def test_analysis_renders_the_plan_the_dataset_and_the_failures():
    from auto_eval.render import render_analysis

    md = render_analysis(_analysis())
    assert "# Source analysis for invoice extractor" in md
    assert "Nothing below was downloaded" in md
    assert "1,000 rows from acme/invoices (default/validation), image_text -> answer" in md
    assert "| Splits | validation (1,000 rows) |" in md
    assert "| Licence | cc-by-4.0 |" in md
    assert "could not read" in md
    assert "HTTP 404." in md
    assert "1 claim(s) discarded" in md


def test_a_long_quote_is_previewed_in_the_document_not_dumped():
    from auto_eval.render import PREVIEW_CHARS, render_analysis

    analysis = _analysis()
    analysis.resources[2].baselines[0].quote = "y" * (PREVIEW_CHARS + 400)
    md = render_analysis(analysis)
    assert "[…]" in md
    assert "y" * (PREVIEW_CHARS + 1) not in md


def test_an_empty_analysis_says_there_is_nothing_to_fetch():
    from auto_eval.analysis import AnalysisReport
    from auto_eval.render import render_analysis

    md = render_analysis(AnalysisReport(subject="support bot"))
    assert "_No source offered anything worth fetching._" in md
