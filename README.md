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
