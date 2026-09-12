# Ground truth identifier

Second block in the pipeline. Takes a `TaskSpec` the classifier has settled and
asks one question: **is there anything online we could anchor this evaluation
on, instead of labelling everything by hand?**

It searches the web and reports two different things, which are easy to
conflate and should not be:

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

Chained onto a classification:

```bash
auto-eval classify -f examples/request.txt --ground-truth
```

Or against a spec written earlier, which is the usual case — you classify, get
questions, answer them, and only then search:

```bash
auto-eval ground-truth -s task.json --md baselines.md
```

It exits `2` without searching while the spec still blocks. In the web UI, the
button appears under the task spec once nothing is blocking, and is replaced by
an explanation when something is.

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
| [prompts.py](../auto_eval/prompts.py) | `GROUND_TRUTH_SYSTEM_PROMPT` and the spec rendering sent with it. |
| [render.py](../auto_eval/render.py) | `GroundTruthReport` → report document. |
| [cli.py](../auto_eval/cli.py) · [web.py](../auto_eval/web.py) | `auto-eval ground-truth` and `POST /api/ground-truth`. |

## Tests

```bash
pytest tests/test_ground_truth.py
```

Offline, like the rest — the search is driven by a fake client, and the gate and
the verdict are pure functions.
