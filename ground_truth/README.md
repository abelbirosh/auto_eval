# Ground truth identifier

Second block in the pipeline. Takes a `TaskSpec` the classifier has settled,
finds what already exists online for it, and reports **what is behind each
link** — so the step that actually downloads data knows exactly what it is
getting before it gets it.

Two stages, and you can stop after the first:

1. **Identify** — search the web for what exists.
2. **Analyse** — open each source and describe it: does the link resolve, what
   columns and splits does the dataset have, what licence, what would a later
   step have to fetch.

**Nothing is downloaded here.** Datasets are read through their metadata only;
pages are read to be described. Pulling the data itself belongs to a later
block, and this one exists to make that pull an informed decision.

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

The whole thing, from a request to a fetch plan on disk:

```bash
auto-eval classify -f examples/request.txt --ground-truth --analyze --out-dir data/
```

Or against a spec written earlier, which is the usual case — you classify, get
questions, answer them, and only then search:

```bash
auto-eval ground-truth -s task.json --analyze --out-dir data/
```

Drop `--analyze` to stop after identification. `--out-dir` writes three files:

| File | What's in it |
| --- | --- |
| `fetch-plan.json` | What a later step should fetch: dataset, config, split, input fields, answer field, how many rows, licence, blockers. |
| `sources.json` | The full analysis of every source, including the ones that failed. |
| `sources.md` | The same as a document. |

It exits `2` without searching while the spec still blocks. In the web UI the
buttons appear in the same order, each once the previous step has run.

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

## A system cannot be the ground truth for its own score

Ask for a comparison of four named vendors and a web search will hand back four
vendor documentation sites, because that is what ranks. Their own pages are
evidence about what their API returns and what it charges. They are not evidence
about whether it is right, and on a comparison every vendor on the list is a
party to the result.

So each source is checked against what the spec says it is testing — the
registrable name in the URL, and the publisher — and one published by a system
under test is marked `self_reported`. The mark is computed in `assess()` from
the spec, never taken from the model, and it has teeth:

- the source is **left out of the KPI coverage and the verdict**, so four vendor
  pages read as `reference_only` rather than `published_baselines`;
- it is still listed, with its numbers, labelled on its face in the report, the
  document, and the page — a vendor's own figure is worth knowing, as long as
  nobody mistakes it for the measurement;
- the analysis stage will not promote it to `ground_truth` however the page
  reads, and adds the caveat.

Hosts that publish other people's work — GitHub, Hugging Face, arXiv, Wikipedia
and the like — are exempt, so a benchmark repository is not demoted for living
on a site the spec happens to name.

## What the analysis returns

One `ResourceAnalysis` per source:

| Field | Meaning |
| --- | --- |
| `reachability` | `ok`, `unreachable` (dead link, timeout), `blocked` (refused by our guards, or gated), `not_checked`. |
| `usability` | `ground_truth`, `baselines`, `background`, `unusable`. |
| `summary` | What the resource actually is. |
| `dataset` | For a dataset: columns, splits and their row counts, licence, gated, downloads, last modified. Metadata only. |
| `baselines` | Published numbers, each carrying the `quote` it came from. |
| `plan` | The `DownloadPlan` — what a later step should fetch. |
| `effort` | `low` (fetch and score), `medium` (reformat or relabel a subset), `high`. |
| `caveats` · `discarded` | What would change the decision, and how many claims the source did not bear out. |

The `DownloadPlan` is the handover: dataset, config, split, `input_fields`,
`expected_field`, `rows_available`, `licence`, `blockers`. A later block can act
on it without re-reading anything.

A source that failed is still a row in the report, with the reason. That is the
honest half of a thin result — it says which links let you down, rather than
going quiet.

## How each source is looked at

**Datasets.** A Hugging Face URL is described from two metadata endpoints —
columns, split sizes, licence, gating, downloads, last modified. No rows are
requested. The model is then asked one narrow question: which columns are the
input, which one holds the answer. It never sees row contents, so the prompt
tells it to say what it cannot know rather than assume a column named `answer`
holds what this task means by an answer. A mapping naming a column the dataset
does not have is refused rather than used, and **the split is chosen here, not
by the model** — held-out over `train`, because training rows are the ones a
model has most likely already seen.

**Pages.** Anything else is read as text and described. Numbers are the one
thing carried forward, and only if quoted: `verify_baselines` checks each quote
against the text we read, and requires the number to be inside its own quote, so
a real quote cannot be used to carry an invented figure. Whatever fails is
dropped and counted in `discarded`. A page claiming to hold baselines that none
of its quotes support falls back to `background`.

A source whose fit is only contextual is never opened at all.

## What it is allowed to touch

This stage is the only part of Auto_Eval that opens a URL, and those URLs were
proposed by a model, so [fetch.py](../auto_eval/fetch.py) refuses rather than
trusts:

- **http(s) only.**
- **No private address, at any hop.** The hostname is resolved and every address
  it answers with is checked, so a public name pointing at `127.0.0.1` or at a
  cloud metadata endpoint is refused instead of being fetched from inside
  whatever network this runs on. Redirects are followed **by hand**, with the
  same check on every hop — a client following them itself would jump straight
  past the first check.
- **A 400KB cap and a 20-second timeout**, so a link to a huge file reads the
  first 400KB and stops.
- **No cookies, no auth headers, no credentials.** The request carries nothing
  identifying you, and nothing the model names can be fetched *as* you.
- **Failure is a value.** An unreachable source becomes a row in the report
  rather than the end of the run.

One hole is left, and it is worth knowing about: a hostname that answers with a
public address at check time and a private one at connection time would still
get through. Closing it needs a connection pinned to the address that was
checked.

Fetched pages are untrusted text from the open web, and both prompts say so
explicitly: the page is data to be described, never instructions. A page that
addresses the model gets reported in `caveats`, not obeyed. Verification is the
harder guarantee underneath that — a prompt can be talked around, a substring
check cannot.

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
| [analysis.py](../auto_eval/analysis.py) | Dataset and page assessment, the download plan, and the verification that decides what is kept. |
| [fetch.py](../auto_eval/fetch.py) | Guarded HTTP and the dataset metadata endpoints. |
| [prompts.py](../auto_eval/prompts.py) | The three prompts: search, dataset assessment, page assessment. |
| [render.py](../auto_eval/render.py) | `GroundTruthReport` and `AnalysisReport` → documents. |
| [cli.py](../auto_eval/cli.py) · [web.py](../auto_eval/web.py) | `auto-eval ground-truth` and `POST /api/ground-truth`, `/api/analyze`. |

## Tests

```bash
pytest tests/test_ground_truth.py tests/test_analysis.py
```

Offline, like the rest: the model calls run against fakes, and the fetchers are
injected, so no test touches the network or spends money. The parts that decide
what is kept — the gate, `assess`, `plan_for_dataset`, `choose_split`,
`verify_baselines`, and the URL guards — are pure functions and are tested
directly.

## Dates, and why they decide what a source is worth

Every source the search reports carries `released` — when the data itself was
published, as the page states it, and null rather than guessed when it does not.
That field is what separates a measurement from a memory test: anything
published before a model's training cutoff may already be in that model's
training data, and a high score on it cannot tell the two apart. The report
names the sources that arrived without a date, so nobody quotes a number off one
by accident.

The search is told to prefer ground truth that resists this — suites refreshed
on a schedule, suites whose answers were never published, anything released
after the model stopped training — and to report the older standard suite as
well where it fits, saying which is which.
[`auto_eval/contamination.py`](../auto_eval/contamination.py) is the
deterministic counterpart: published cutoffs with the page each was read from,
and a small catalogue of contamination-resistant suites, printed by `auto-eval
fresh --model <model>`.
