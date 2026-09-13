import pytest

from auto_eval.dataset import DatasetError, from_rows, load, parse

JSONL = """
{"id": "a", "query": "who?", "answers": ["Bolt"], "published": "2026-06-04"}
{"question": "what?", "answer": "Series C"}
"""


def test_jsonl_is_read_row_by_row():
    dataset = parse(JSONL, name="news")
    assert [item.id for item in dataset.items] == ["a", "1"]
    assert dataset.items[0].answers == ["Bolt"]
    assert str(dataset.items[0].published) == "2026-06-04"


def test_the_common_column_names_are_accepted():
    """Datasets in the wild spell these differently; these are the usual ones."""
    dataset = parse(
        '{"input": "q", "gold": "g", "source_url": "https://x/y", "as_of": "2025-01-02"}'
    )
    item = dataset.items[0]
    assert (item.query, item.answers, item.url) == ("q", ["g"], "https://x/y")
    assert str(item.published) == "2025-01-02"


def test_a_column_we_cannot_place_is_refused_rather_than_guessed_at():
    """Guessing which column holds the question is how a board scores the wrong thing."""
    with pytest.raises(DatasetError, match="items"):
        parse('{"text": "who?", "answer": "x"}')
    with pytest.raises(DatasetError, match="no question"):
        parse('[{"text": "who?", "answer": "x"}]')


def test_a_json_array_and_a_json_object_both_work():
    assert len(parse('[{"query": "a", "answer": "1"}]').items) == 1
    wrapped = parse(
        '{"name": "n", "task": "t", "items": [{"query": "a", "answer": "1"}]}'
    )
    assert (wrapped.name, wrapped.task) == ("n", "t")


def test_unreadable_json_says_which_line():
    with pytest.raises(DatasetError, match="Line 2"):
        parse('{"query": "a"}\n{not json')


def test_an_empty_dataset_is_refused():
    with pytest.raises(DatasetError, match="Nothing to read"):
        parse("   ")
    with pytest.raises(DatasetError, match="No items"):
        from_rows([])


def test_the_digest_follows_the_items():
    """A board quoting a digest has to mean the same items every time."""
    first = parse(JSONL, name="news")
    assert first.digest == parse(JSONL, name="news").digest
    assert first.digest != parse(JSONL + '{"query": "extra", "answer": "x"}\n').digest


def test_an_item_with_no_gold_answer_needs_a_judge():
    dataset = parse('{"query": "describe x"}\n{"query": "y", "answer": "z"}')
    assert dataset.items[0].judged and not dataset.items[1].judged
    assert dataset.judged_share == 0.5


def test_the_sample_is_a_prefix_not_a_shuffle():
    """A board that scored a different subset each run could not be compared."""
    dataset = parse(
        "\n".join(f'{{"id": "{i}", "query": "q{i}", "answer": "a"}}' for i in range(5))
    )
    assert [i.id for i in dataset.sample(3)] == ["0", "1", "2"]
    assert len(dataset.sample(None)) == 5


def test_dates_bound_the_dataset():
    dataset = parse(JSONL)
    assert str(dataset.latest) == "2026-06-04" and str(dataset.earliest) == "2026-06-04"


def test_loading_from_a_file_names_it_and_records_where_it_came_from(tmp_path):
    path = tmp_path / "news.jsonl"
    path.write_text(JSONL, encoding="utf-8")
    dataset = load(path)
    assert dataset.name == "news" and dataset.source == str(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(DatasetError, match="No dataset at"):
        load(tmp_path / "nope.jsonl")


def test_extra_columns_are_kept_rather_than_dropped():
    dataset = parse('{"query": "q", "answer": "a", "difficulty": "hard"}')
    assert dataset.items[0].meta == {"difficulty": "hard"}
