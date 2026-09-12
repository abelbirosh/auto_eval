# The task spec

`TaskSpec` is the contract between the input classifier and everything
downstream. One free-form request in, one `TaskSpec` out. Print the machine
readable version with `auto-eval schema`.

It answers three questions, and records what it could not answer.

## 1. What are we testing — `subject`

| Field | Meaning |
| --- | --- |
| `name` | Short name for the thing under test. |
| `kind` | `prompt`, `agent`, `rag_pipeline`, `model`, `tool`, `workflow`, `api_endpoint`, `classifier`, `dataset`, `other`. |
| `description` | A sentence or two on what it does. |
| `interface` | How it is invoked — CLI command, endpoint, function, prompt template. **Required to build a harness**, so its absence is a blocking gap. |
| `inputs` / `outputs` | What goes in, what comes out. |
| `in_scope` / `out_of_scope` | The boundary of this evaluation. |

`eval_types` sits alongside it and says *why* we are evaluating — `capability`,
`regression`, `comparative`, `safety`, `robustness`, `performance`, `cost`,
`compliance`. A task is often several at once.

## 2. KPIs — `kpis`

Each KPI is one thing we will measure. Quantitative and qualitative KPIs use the
same record; `kind` distinguishes them.

| Field | Meaning |
| --- | --- |
| `name` | What it is called. |
| `kind` | `quantitative` (a number) or `qualitative` (a judged quality). |
| `definition` | Precise enough to implement without a follow-up question. |
| `measurement` | `programmatic`, `execution`, `llm_judge`, `human_review`, `telemetry`, or `unknown`. |
| `direction` | `maximize`, `minimize`, `target_range`. |
| `unit` | `%`, `ms`, `USD`, `1-5`. |
| `target` | The pass threshold, verbatim from the user where given. |
| `baseline` | Where it stands today. |
| `priority` | `primary` (the task succeeds or fails on it), `secondary`, or `guardrail` (must not regress). |
| `source` | `stated` by the user, or `inferred` by the classifier. |

The classifier may propose an `inferred` KPI when the task obviously implies one,
but never marks an inferred KPI `primary` — what the task is *for* is the user's
call, not the model's. Every inference also lands in `assumptions`.

`ground_truth` records whether we can tell right from wrong without a human:
what the labels are, where they come from, how many there are.

## 3. Evidence — `evidence`

Documentation, specs, datasets, existing evals, and above all **successful runs**
and **failure cases**. Each item has a `kind`, a `reference` in the user's own
words, an optional `location`, and a `status`:

- `provided` — the content is in hand and can be used now.
- `mentioned_not_provided` — referenced but not attached. This is a fetch, not an
  unknown, and it becomes a specific "can you share X" question.
- `absent` — the user says it does not exist. Does not count as available.

## What the spec could not answer

- `assumptions` — everything the classifier filled in that the user did not say,
  phrased so it can be contradicted in one sentence.
- `open_questions` — what we need back. Each carries the `field` it would fill,
  a `why`, and a `blocking` flag.
- `confidence` — per section, how much came from the user (1.0) rather than from
  inference (0.0).
- `readiness` — `ready`, `needs_input`, or `insufficient`.

## Where readiness comes from

The model extracts; it does not decide. `auto_eval.gaps` recomputes
`open_questions` and `readiness` from rules after every classification, and the
model's own guess is discarded. The rules that block are:

| Rule | Why it blocks |
| --- | --- |
| No `subject.interface` | The harness cannot call the system under test. |
| No `kpis` | Nothing to score. |
| A `programmatic` KPI with no ground truth | The grader has nothing to compare against. |

Everything else — no primary KPI, an untargeted KPI, a judged quality with no
anchoring examples, missing docs, missing successful runs, missing failure cases,
evidence referenced but not supplied — produces a non-blocking question.

Two rules fire on *every* spec that lacks them, because they are what separates
an anchored eval from a guessed one: ask for documentation, and ask for examples
of runs that came out the way the user wanted.

Model questions are merged in after the rules and deduplicated against them by
token overlap, so a reworded echo of a rule question is dropped and a genuinely
new question survives.

## A note on required fields

The spec is extracted through OpenAI structured outputs in strict mode, which
makes every property required. The model therefore returns all twelve top-level
fields on every call, emitting empty lists and nulls for what it did not find,
rather than omitting them. Nothing downstream changes: an empty `kpis` is still
"found nothing", and the blocking rule above fires on it.
