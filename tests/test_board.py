import pytest

from auto_eval.board import BoardError, list_boards, load_board, run_board, write_board
from auto_eval.cohort import SystemKind
from auto_eval.cohort import parse as parse_cohort
from auto_eval.dataset import parse as parse_dataset

ITEMS = "\n".join(
    [
        '{"id":"i1","query":"Who did Acme acquire?","answers":["Bolt Systems"],"published":"2026-06-04"}',
        '{"id":"i2","query":"What round did Foobar close?","answers":["Series C"],"published":"2026-07-11"}',
        '{"id":"i3","query":"Where did Zeta open an office?","answers":["Lisbon"],"published":"2026-08-01"}',
    ]
)

COHORT = """{
  "name": "vendors", "task": "factual lookup", "model": "fake-model",
  "systems": [
    {"label": "Alpha", "configuration": "POST /search",
     "endpoint": {"url": "https://alpha.test/search", "body": {"q": "{query}"}},
     "price": {"per_call_usd": 0.01, "note": "$10 / 1k"}},
    {"label": "Beta", "configuration": "GET /search",
     "endpoint": {"method": "GET", "url": "https://beta.test/search", "params": {"query": "{query}"},
                  "results_path": "data", "fields": {"title": "t", "url": "u", "snippet": "s"}}}
  ]
}"""

ANSWERS = {"Acme": "Bolt Systems", "Foobar": "Series C", "Zeta": "Lisbon"}


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeHTTP:
    """Alpha answers everything at rank 1; Beta buries the answer at rank 3."""

    def post(self, url, json=None, **kw):
        answer = self._answer(json.get("q", ""))
        return self._ok(
            {"results": [{"title": "n", "url": "https://n.test", "text": answer}]}
        )

    def get(self, url, params=None, **kw):
        answer = self._answer((params or {}).get("query", ""))
        data = [
            {"t": "a", "u": "", "s": "nothing"},
            {"t": "b", "u": "", "s": "nothing"},
            {"t": "c", "u": "", "s": answer},
        ]
        return self._ok({"data": data})

    def _answer(self, query):
        return next((v for k, v in ANSWERS.items() if k in query), "no idea")

    def _ok(self, payload):
        return _Obj(status_code=200, text="{}", json=lambda: payload)


class FakeModel:
    """Knows one of the three facts and admits the rest."""

    def __init__(self, knows="Acme"):
        self.knows = knows
        self.chat = _Obj(completions=self)

    def create(self, **kwargs):
        question = kwargs["messages"][-1]["content"]
        reply = (
            ANSWERS[self.knows] if self.knows and self.knows in question else "unknown"
        )
        return _Obj(
            choices=[
                _Obj(message=_Obj(content=reply, tool_calls=None), finish_reason="stop")
            ],
            usage=_Obj(prompt_tokens=50, completion_tokens=5),
        )


def board(**kwargs):
    return run_board(
        parse_dataset(ITEMS, name="news", task="factual lookup"),
        parse_cohort(COHORT),
        client=kwargs.pop("client", FakeModel()),
        http=kwargs.pop("http", FakeHTTP()),
        **kwargs,
    )


# --- the rows -------------------------------------------------------------


def test_every_system_gets_a_row_over_the_same_items():
    built = board()
    assert {r.label for r in built.rows} == {"Alpha", "Beta", "model only (no search)"}
    assert all(r.n == 3 for r in built.rows)
    assert built.items == 3


def test_rows_are_alphabetical_not_ranked():
    assert [r.label for r in board().rows] == [
        "Alpha",
        "Beta",
        "model only (no search)",
    ]


def test_where_the_answer_was_found_separates_two_systems_that_both_found_it():
    built = board()
    alpha = next(r for r in built.rows if r.label == "Alpha")
    beta = next(r for r in built.rows if r.label == "Beta")
    assert alpha.accuracy == beta.accuracy == 1.0
    assert alpha.recall_at_1 == 1.0 and beta.recall_at_1 == 0.0
    assert beta.recall_at_5 == 1.0


def test_the_baseline_row_measures_what_can_simply_be_recalled():
    built = board()
    baseline = built.baseline
    assert baseline is not None and baseline.kind is SystemKind.MODEL_ONLY
    assert baseline.accuracy == pytest.approx(1 / 3)
    assert any("memory test" in w for w in built.warnings)


def test_a_baseline_of_zero_is_reported_as_the_board_being_sound():
    built = board(client=FakeModel(knows=None))
    assert built.baseline.accuracy == 0.0
    assert any("cannot be recalled" in w for w in built.warnings)


def test_the_baseline_can_be_left_out_but_is_there_by_default():
    assert board(baseline=False).baseline is None
    assert board().baseline is not None


def test_spend_is_priced_only_where_a_price_was_published():
    built = board()
    alpha = next(r for r in built.rows if r.label == "Alpha")
    beta = next(r for r in built.rows if r.label == "Beta")
    assert alpha.usd == pytest.approx(
        0.03
    ) and alpha.usd_per_1k_correct == pytest.approx(10.0)
    assert beta.usd is None and beta.usd_per_1k_correct is None
    assert any("no published price" in w for w in built.warnings)


def test_a_system_missing_its_key_is_skipped_with_a_reason(monkeypatch):
    monkeypatch.delenv("GAMMA_KEY", raising=False)
    cohort = parse_cohort(
        COHORT.replace(
            '{"label": "Alpha"',
            '{"label": "Gamma", "endpoint": {"url": "https://gamma.test/s", "headers": {"k": "${GAMMA_KEY}"}, "body": {"q": "{query}"}}},'
            '{"label": "Alpha"',
        )
    )
    built = run_board(
        parse_dataset(ITEMS, name="news"), cohort, client=FakeModel(), http=FakeHTTP()
    )
    gamma = next(r for r in built.rows if r.label == "Gamma")
    assert gamma.skipped and "GAMMA_KEY" in gamma.skipped
    assert gamma.n == 0
    assert any("did not run" in w for w in built.warnings)


def test_an_endpoint_that_errors_is_not_scored_as_wrong():
    class Broken(FakeHTTP):
        def post(self, url, json=None, **kw):
            raise TimeoutError("timed out")

    built = board(http=Broken())
    alpha = next(r for r in built.rows if r.label == "Alpha")
    assert alpha.errors == 3 and alpha.accuracy is None
    assert any("Error rate over 10%" in w for w in built.warnings)


def test_only_the_first_n_items_are_scored_when_asked():
    built = board(limit=2)
    assert built.items == 2 and all(r.n in (0, 2) for r in built.rows)


def test_what_was_held_constant_is_recorded_with_the_board():
    built = board()
    assert built.model == "fake-model"
    assert any("held constant" in note for note in built.methodology)
    assert any("alphabetical" in note for note in built.methodology)


def test_the_contamination_verdict_rides_along():
    built = board()
    assert built.contamination is not None
    assert built.contamination.model == "fake-model"


def test_a_small_board_says_it_is_small():
    assert any("too few to separate" in w for w in board().warnings)


def test_an_empty_dataset_is_refused():
    with pytest.raises(BoardError, match="Nothing to run"):
        run_board(
            parse_dataset('{"query":"q","answer":"a"}', name="one"),
            parse_cohort(COHORT),
            client=FakeModel(),
            http=FakeHTTP(),
            limit=0,
        )


# --- writing it down ------------------------------------------------------


def test_a_board_survives_the_round_trip(tmp_path):
    built = board()
    target = write_board(built, tmp_path)
    read_back = load_board(target)
    assert read_back.board_id == built.board_id
    assert [r.accuracy for r in read_back.rows] == [r.accuracy for r in built.rows]
    assert read_back.rows[0].verdicts[0].evidence


def test_the_board_list_carries_the_leader_and_the_baseline(tmp_path):
    write_board(board(), tmp_path)
    entry = list_boards(tmp_path)[0]
    assert entry.items == 3 and entry.rows == 3
    assert entry.leader in {"Alpha", "Beta"} and entry.leader_accuracy == 1.0
    assert entry.baseline_accuracy == pytest.approx(1 / 3)


def test_an_unreadable_board_is_skipped_rather_than_breaking_the_list(tmp_path):
    broken = tmp_path / "half-written"
    broken.mkdir()
    (broken / "board.json").write_text("{not json", encoding="utf-8")
    assert list_boards(tmp_path) == []


def test_a_missing_board_says_so(tmp_path):
    with pytest.raises(BoardError, match="No board at"):
        load_board(tmp_path / "nope")
