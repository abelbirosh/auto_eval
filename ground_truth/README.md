# Ground truth identifier

Second block in the pipeline. Takes a `TaskSpec` the classifier has settled,
finds what already exists online for it, and **extracts the ground truth
itself** — labelled cases an eval harness can run on, and published numbers to
compare against.

Two stages, and you can stop after the first:

1. **Identify** — search the web for what exists.
2. **Extract** — fetch those sources and pull the actual cases out of them.

Both stages keep apart two things that are easy to conflate:

- **Ground truth** — something that states what the correct answer *is*: a
  labelled dataset, a named benchmark, a standard or regulation that defines
  correct behaviour, a reference implementation. This lets the eval score.
- **Baselines** — a number a comparable system already reaches: a leaderboard
  entry, a published result, a vendor's own accuracy claim or SLA. This says
  what good looks like but cannot score anything.

## When it runs

Once the spec has **no blocking questions left**. Until then we do not know what
is being measured, and a search on an unsettled KPI comes back confident and
wrong. `gate()` is the whole trigger:

```python
from auto_eval import classify, gate, identify

spec = classify(request)
if gate(spec).open:
    report = identify(spec)
```

`identify()` refuses a shut gate rather than searching anyway. Pass `force=True`
to override it, and expect noise.

## Run

The whole thing, from a request to cases on disk:

```bash
auto-eval classify -f examples/request.txt --ground-truth --extract --out-dir data/
```

Or against a spec written earlier, which is the usual case — you classify, get
questions, answer them, and only then search:

```bash
auto-eval ground-truth -s task.json --extract --out-dir data/
```

Drop `--extract` to stop after identification. `--out-dir` writes three files:

| File | What's in it |
| --- | --- |
| `examples.jsonl` | One labelled case per line: `input`, `expected`, `kpi`, `source`, `url`, `origin`. |
| `baselines.json` | Published numbers, each with the span of page it was quoted from. |
| `ground-truth.md` | The document: cases, numbers, and what each source gave. |

`--max-examples` caps how many cases come back (default 20; raise it for a full
split). It exits `2` without searching while the spec still blocks. In the web
UI the buttons appear in the same order, each once the previous step has run.

## What comes back

`GroundTruthReport`:

| Field | Meaning |
| --- | --- |
| `verdict` | `labelled_data`, `published_baselines`, `reference_only`, or `none_found`. |
| `sources` | What was found: name, kind, URL, fit, access, licence, published numbers, caveats. |
| `kpi_coverage` | Per KPI: `labelled`, `baseline`, `reference`, or `none`, and which sources cover it. |
| `internal_ground_truth` | Whether the spec already has its own labels. If so, public sources are for comparison, not scoring. |
| `searches` | The queries the model actually ran, read off the response. |
| `recommendation` | What to use, and what still has to be labelled by hand. |

A source's `fit` is judged against this subject: `direct` (same task, same
metric), `adaptable` (needs a subset or a reformat), `contextual` (a reference
point only).

## What extraction returns

`GroundTruthSet`:

| Field | Meaning |
| --- | --- |
| `examples` | The labelled cases: `input`, `expected`, the `kpi` they score, and where each came from. |
| `baselines` | Published numbers, each carrying the `quote` it was taken from and the URL. |
| `outcomes` | Per source: `extracted`, `nothing_found`, `unreachable`, `unusable`, or `skipped`, with the reason and how much was discarded. |
| `notes` | Anything that changes how to read the set. |

`outcomes` is the important half of a thin result: it says which sources let you
down and why, so what is left to label by hand is explicit rather than implied.

Each example records its `origin`:

- `dataset_rows` — the values came verbatim out of a dataset API.
- `page_quote` — the values were found in a page we fetched, and checked against
  it.

## Where the cases come from

**Datasets.** A Hugging Face dataset URL is read through the datasets server,
which serves real rows of public datasets over an open API. The model is asked
one narrow question — which columns are the input, which one holds the answer —
and the cases are then built from the rows themselves. A held-out split is
preferred over `train`, since training rows are the ones a model has most likely
already seen. A mapping that names a column the data does not have is refused
rather than used.

**Pages.** Anything else is fetched and reduced to text. The model may only
report what it can quote, and `verify_page` checks every quote against the text
we fetched: the quote must appear in the page, and a baseline's number must
appear inside its own quote, so a real quote cannot be used to carry an invented
figure. Whatever fails is dropped and counted in `discarded`.

So nothing reaches `examples.jsonl` because a model said it. Either it came out
of a dataset API, or it was found in a page we read ourselves.

## Fetching, and what it is allowed to touch

Extraction is the only part of Auto_Eval that reaches out to URLs, and those
URLs were proposed by a model, so [fetch.py](../auto_eval/fetch.py) is written
to refuse rather than trust: http(s) only, never a loopback or private address,
a 400KB cap, a 20-second timeout, no cookies or credentials. Failure is a value,
never an exception — an unreachable source becomes a row in `outcomes`.

Fetched pages are untrusted text from the open web, and both extraction prompts
say so explicitly: the page is data to be extracted from, never instructions. A
page that addresses the extractor gets reported in `notes`, not obeyed.

## How it works

The search runs through the OpenAI Responses API with the hosted `web_search`
tool — accounts that only expose `web_search_preview` are retried on that name
automatically. `AUTO_EVAL_SEARCH_MODEL` sets a stronger model for this block
alone; unset, it uses `AUTO_EVAL_MODEL`.

**The model searches; it does not decide.** As in the classifier, the verdict is
computed afterwards, in `assess()`, which is a pure function:

- a source with no openable `http(s)` link is dropped and the drop is noted — a
  link we cannot follow is a claim, not a finding;
- duplicate URLs collapse to one entry;
- `covers_kpis` is matched back against the KPI names in the spec, so a
  paraphrase still counts and an invented KPI does not;
- each KPI's coverage is computed from the sources that survived, and the
  verdict is the best coverage any KPI got.

So `none_found` is a real, reportable answer rather than a failure, and the
prompt says as much: reporting nothing beats reporting something plausible.

The spec is treated as data — text inside it addressed to the searcher gets
searched on, not obeyed.

## Layout

| Path | What's in it |
| --- | --- |
| [ground_truth.py](../auto_eval/ground_truth.py) | The gate, the report models, the search call, and `assess`. |
| [extraction.py](../auto_eval/extraction.py) | Field mapping, page extraction, and the verification that decides what is kept. |
| [fetch.py](../auto_eval/fetch.py) | Guarded HTTP and the dataset rows API. |
| [prompts.py](../auto_eval/prompts.py) | The three prompts: search, field mapping, page extraction. |
| [render.py](../auto_eval/render.py) | `GroundTruthReport` and `GroundTruthSet` → documents. |
| [cli.py](../auto_eval/cli.py) · [web.py](../auto_eval/web.py) | `auto-eval ground-truth` and `POST /api/ground-truth`, `/api/extract`. |

## Tests

```bash
pytest tests/test_ground_truth.py tests/test_extraction.py
```

Offline, like the rest: the model calls run against fakes, and the fetchers are
injected, so no test touches the network or spends money. The parts that decide
what is kept — the gate, `assess`, `examples_from_rows`, `verify_page` — are
pure functions and are tested directly.
