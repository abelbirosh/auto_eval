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
