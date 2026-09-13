from auto_eval.endpoint import Hit
from auto_eval.score import (
    contains,
    matches,
    normalise,
    percentile,
    score_hits,
    score_reply,
    summarise,
)


def hit(rank, text):
    return Hit(rank=rank, title="", url="", snippet=text)


# --- matching -------------------------------------------------------------


def test_form_is_normalised_away_and_content_is_not():
    """'1,200 employees' and '1200 employees' are the same answer."""
    assert contains("Quux has 1,200 employees", "1200 employees")
    assert contains("The CEO is José Álvarez.", "Jose Alvarez")
    assert not contains("Acme acquired Bolt", "Acme acquired Volt")


def test_filler_words_do_not_decide_a_match():
    assert contains("closed a Series C round", "Series C")
    assert contains("the answer is Lisbon", "Lisbon")


def test_matches_returns_which_gold_answer_was_found():
    assert (
        matches("released on March 24, 2025", ["2025-03-24", "March 24, 2025"])
        == "March 24, 2025"
    )
    assert matches("nothing here", ["x"]) is None


def test_normalise_is_stable_on_punctuation_and_case():
    assert normalise("  The ANSWER: $1,200.  ") == normalise("answer $1200.")


# --- scoring what came back -----------------------------------------------


def test_the_rank_the_answer_was_found_at_is_kept():
    """Answered by the top result is a different product from answered at five."""
    verdict = score_hits(
        "i1", ["Lisbon"], [hit(1, "nothing"), hit(2, "office in Lisbon")]
    )
    assert verdict.correct and verdict.rank == 2
    assert "result 2" in verdict.evidence


def test_results_without_the_answer_fail_with_a_count():
    verdict = score_hits("i1", ["Lisbon"], [hit(1, "nothing"), hit(2, "also nothing")])
    assert not verdict.correct and "none of 2 result(s)" in verdict.evidence


def test_an_endpoint_error_is_an_error_not_a_wrong_answer():
    verdict = score_hits("i1", ["x"], [], error="HTTP 500")
    assert not verdict.correct and verdict.error == "HTTP 500"


def test_an_item_with_no_gold_answer_says_it_needs_a_judge():
    verdict = score_hits("i1", [], [hit(1, "anything")])
    assert not verdict.correct and "needs a judge" in verdict.note


def test_a_reply_is_scored_on_the_same_matching_rule():
    verdict = score_reply("i1", ["Series C"], "It closed a Series C in July.")
    assert verdict.correct and "Series C" in verdict.evidence


def test_a_wrong_reply_is_quoted_back():
    verdict = score_reply("i1", ["Series C"], "unknown")
    assert not verdict.correct and verdict.evidence == "unknown"


# --- adding it up ---------------------------------------------------------


def test_errors_are_out_of_the_denominator():
    verdicts = [
        score_hits("a", ["x"], [hit(1, "x")]),
        score_hits("b", ["x"], [hit(1, "no")]),
        score_hits("c", ["x"], [], error="timeout"),
    ]
    summary = summarise(verdicts)
    assert (summary.n, summary.scored, summary.correct, summary.errors) == (3, 2, 1, 1)
    assert summary.accuracy == 0.5


def test_recall_at_one_and_five_are_counted_separately():
    verdicts = [
        score_hits("a", ["x"], [hit(1, "x")]),
        score_hits("b", ["x"], [hit(1, "no"), hit(4, "x")]),
        score_hits("c", ["x"], [hit(9, "x")]),
    ]
    summary = summarise(verdicts)
    assert summary.at_1 == 1 and summary.at_5 == 2 and summary.correct == 3


def test_percentiles_need_no_interpolation_argument():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert percentile([], 0.5) is None


def test_a_number_is_matched_as_a_number_not_as_a_substring():
    """A gold answer of 43 must not be found inside 143."""
    assert contains("Tier 4 contains 43 problems", "43")
    assert not contains("the set has 143 problems", "43")
    assert not contains("scored 293.9% somehow", "93.9%")
    assert contains("saturated at 93.9% (congrats)", "93.9%")


def test_word_boundaries_do_not_break_ordinary_answers():
    assert contains("opened an office in Lisbon, Portugal", "Lisbon")
    assert contains("released on March 24, 2025.", "March 24, 2025")
    assert not contains("Lisbonne is a different city", "Lisbon")
