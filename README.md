# Auto-Eval

**Turn a description of what you want tested into a reproducible benchmark.**

Auto-Eval is an open-source system for automatically designing, building, running, and maintaining evaluation suites for software and AI systems.

Give it a goal like:

```text
Compare company enrichment APIs on accuracy, coverage, latency, and cost.
```

Auto-Eval turns that into:

```text
Free-form request
→ structured evaluation spec
→ benchmark / ground-truth discovery
→ cohort + test-case construction
→ evaluation harness
→ execution
→ scoring
→ comparison + reporting
```

## What it handles

* Extracts the system under test, KPIs, scope, and success criteria
* Finds relevant public benchmarks, datasets, and baselines
* Builds representative test cases and coverage matrices
* Constructs or verifies ground truth when needed
* Generates adapters for APIs, agents, CLIs, and model endpoints
* Runs systems under identical conditions
* Measures accuracy, coverage, latency, cost, tokens, and failures
* Uses deterministic checks, LLM judges, or human review as appropriate
* Produces versioned, auditable benchmark results
* Re-runs benchmarks over time and detects regressions

## Example

```text
"Compare Apollo, People Data Labs, Exa, and Parallel
for company enrichment."
```

Auto-Eval can produce:

```text
benchmark/
├── spec.json
├── methodology.md
├── cohort.jsonl
├── ground_truth.jsonl
├── adapters/
├── cases/
├── scorers/
├── raw_runs/
└── results.json
```

## Design Principles

**Deterministic where possible.**
Models assist with extraction, research, and semantic judgment; rules handle validation, sampling, scoring, and aggregation where possible.

**Evidence over confidence.**
Ground truth and judgments retain their supporting evidence.

**Ground truth ≠ scoring policy.**
What is true and what counts as correct are stored separately.

**Reproducible by default.**
Specs, suites, cases, scoring rules, and results are versioned and content-addressed.

## Running it

Everything after the request is a button. `auto-eval serve` puts the pipeline on
one page: describe what you want tested, answer whatever blocks, find the ground
truth, then **Run the benchmark** — one click writes the suite for that spec and
runs every case against the provider, with the scores underneath and a link
through to `/dashboard` for every verdict and the transcript it was read off. A
dry run needs no API key.

The same steps from the command line:

```bash
auto-eval classify "..." --json task.json   # request → spec
auto-eval ground-truth -s task.json         # what already exists publicly
auto-eval author -s task.json -o suites/x   # the cases and their checks
auto-eval run suites/x                      # run them; --mock calls no provider
auto-eval serve                             # the UI, dashboard at /dashboard
auto-eval fresh --model gpt-5               # ground truth this model cannot have memorised
```

A run's tools are declared to the model and answered inside the harness, so
nothing touches a live system. That makes the trajectory checks real and the
end-state checks impossible, so those are reported as **blocked, with the reason**
rather than dropped — every pass rate sits next to the count of what did not run.
[`runner/`](runner/) has the detail.

## Comparing several systems

Not everything under test has a trajectory. When the subject is an endpoint, a
model, or a retrieval API, the question is which of several systems does the job
best, and the answer is a board: one dataset, many systems, one row each, with
accuracy, answer recall at 1 and 5, latency, errors and spend at the vendor's
published price.

```bash
auto-eval board items.jsonl -c cohort.json    # or "Run the board" on the page
auto-eval boards
```

| System | Endpoint & configuration | Accuracy | AR@1 | AR@5 | p50 | Total $ | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exa auto | POST /search type=auto | 100.0% | 80.0% | 100.0% | 1.27s | $0.0350 | $7.00 |
| Exa fast | POST /search type=fast | 100.0% | 80.0% | 100.0% | 416ms | $0.0350 | $7.00 |
| Exa fast, model with tool | the model, with that endpoint as its only tool | 80.0% | — | — | 7.39s | $0.1352 | $33.80 |
| Firecrawl | POST /v2/search limit=10 | 80.0% | 20.0% | 60.0% | 944ms | — | — |
| Firecrawl, model with tool | the model, with that endpoint as its only tool | 40.0% | — | — | 14.94s | $0.0160 | $7.99 |
| model only (no search) | the model answering from memory | 0.0% | — | — | 2.88s | — | — |
| TinyFish | GET api.search.tinyfish.ai | 100.0% | 20.0% | 100.0% | 689ms | $0.0000 | $0.0000 |
| TinyFish, model with tool | the model, with that endpoint as its only tool | 100.0% | — | — | 9.10s | $0.0058 | $1.16 |

The first row is the point. **Model only** is the same model with no endpoint at
all: on items published after its training cutoff it should score near zero, and
when it does, every other row is measuring retrieval rather than recall. A vendor
that prices in credits rather than per call has blank cost cells rather than a
converted guess. What is
held constant — the items, the model, the judge — is recorded on the board, rows
are alphabetical because no single column decides the answer, and a vendor whose
key is missing is reported as not run rather than scored as bad. [`board/`](board/)
has the detail.

## Contamination

A public suite older than the model measures capability and recall at once. Every
run report says which of its cases the model could already have seen: cases
harvested from your own runs or generated from your spec are clean by
construction, and one adapted from a public suite is only as clean as that
suite's publication date. `auto-eval fresh` lists the ground truth that resists
the problem — suites refreshed on a schedule, suites whose answers were never
published — judged against a model's published cutoff.

## Vision

Building a serious benchmark currently requires manual research, dataset design, integration work, scoring logic, and reporting.

Auto-Eval aims to reduce that workflow to:

```text
Describe the evaluation
→ inspect the methodology
→ run the benchmark
→ reproduce the result
```

**Auto-Eval is infrastructure for turning evaluation questions into evidence.**
