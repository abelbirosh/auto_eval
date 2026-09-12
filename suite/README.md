# Agent suite builder

Fourth block in the pipeline. Takes a `TaskSpec` whose subject is an **agent**
(or a workflow) and writes the eval suite for it: the cases, the checks on each
case, and the coverage grid the whole thing is answerable to.

Everything before this block grades a string. An agent does not return a string
— it returns a **trajectory and a changed world** — so an output-only grader
cannot tell the difference between an agent that did the job and one that
deleted the wrong record and then wrote a convincing summary of its success.
That difference is what this block exists to measure.

**Deterministic and offline.** No model call, no network, no API key. The same
spec produces the same suite, digest and all, so a suite can be rebuilt and
diffed rather than regenerated into something subtly different.

## The four stages

| Module | Question | Output |
| --- | --- | --- |
| [`agent.py`](../auto_eval/agent.py) | What can we drive, and is it safe to drive it? | `AgentProfile` + a gate |
| [`surface.py`](../auto_eval/surface.py) | What should be tested? | `CoverageMatrix` |
| [`authoring.py`](../auto_eval/authoring.py) | Where do the cases come from? | `Case[]` |
| [`verify.py`](../auto_eval/verify.py) | How does each case get scored? | `Verifier[]` |

[`suite.py`](../auto_eval/suite.py) freezes the result into the artefact block 5
runs.

## 1. Profile, and the gate

Three things have to be pinned down before a case can be written, and `TaskSpec`
holds none of them: the **entry point** (how one episode starts and how we know
it ended), the **tools** (what it can reach, and which of those reach outside the
test), and the **budget** (what stops a run that would otherwise never
terminate). `profile()` reads all three off the spec by rule and records every
one of them in `assumptions` — it is a draft to correct, not a claim about your
system.

The gate blocks on exactly two things, both unrecoverable:

- **Nothing to run.** No entry point *and* no recorded runs to score instead.
  With recorded runs, the suite falls back to `transcript` mode and scores those.
- **Nowhere safe to run it.** A destructive or external tool with no sandbox. An
  eval that exercises the refund tool against production is an incident, not a
  test.

Everything else — an unknown tool list, no stated ceiling — is a guess the
profile records and you correct.

## 2. Coverage, decided before anything is written

Behaviours down the side, reasons for testing across the top. Authoring fills
cells; a cell that stays empty is **printed as a gap**, not quietly absent.

Two things are deliberately not rows. **Cost and latency** get no cases of their
own — they are read off every run the suite already makes. **Comparative** gets
none either — comparing two agents is running the same suite twice. Both appear
in the report as riders on the grid.

A regression column is added whenever the spec carries an incident, whether or
not it was asked for, because that case is going to be written and a case with
no cell to land in goes uncounted.

## 3. Where cases come from

In the order they are trusted:

1. **Harvested** — your own runs. A run you were happy with is a case whose end
   state is known; an incident you remember is a regression case that is, by
   construction, about something that really breaks. Best data in the building,
   and it arrives free with the spec.
2. **Adapted** — a public suite the [catalogue](../auto_eval/benchmarks.py)
   matched. Real tasks, real scoring, and a contamination problem: anything
   public may already be in the agent's training data, so every adapted case
   carries its source.
3. **Synthesised** — generated from the spec by rule. Cheap, unlimited, and the
   weakest of the three, because a generated case measures the generator as much
   as the agent.

Harvested and adapted cases are written **first** and counted against the grid,
so synthesis only fills what is genuinely still empty. Above 50% synthesised, the
suite says on its own face that a headline number from it is partly a
measurement of the generator.

Nothing generates a regression case. Inventing an incident tests our imagination.

## 4. Checks

Each KPI's `Measurement` picks its check, so the spec keeps deciding what "good"
means:

| Measurement | Check | Reads |
| --- | --- | --- |
| `programmatic` | end-state assertion | environment |
| `execution` | run the project's own tests against the result | environment |
| `telemetry` | steps, time, tokens, spend | trace |
| `llm_judge` | rubric over the trace **and** the diff | both |
| `human_review` | queued for a person, trace attached | both |

On top of those, every case carries guardrails with no non-agent analogue, and
they are where agents actually fail: **terminated** (stopped rather than looped),
**trace budget**, **no escape** from the sandbox, **forbidden tool**, and per
family **recovered from a tool error** and **ignored an instruction planted in
tool output**. A forbidden-tool check is written per case, not per agent — a case
about issuing refunds cannot also forbid the refund tool.

One rule is imposed on top, the same one the rest of the pipeline runs on:
**nothing is believed because a model said it.** A verdict must carry the span it
was read off, and `accept()` fails a pass that arrives without one, however
confident it sounds. A check that never ran counts as failed, not as absent.

## What this block does not do

- **It does not run anything.** That is block 5.
- **It does not build fixtures.** We do not have your environment, so what comes
  out is the *specification* of each starting state. The suite counts them and
  the document lists them; nothing runs until they exist.
- **It does not score.** `CaseResult` and `settle()` define the contract the
  runner fills in.

## Run

```bash
auto-eval profile -s task.json          # what it takes to run it, and whether we can
auto-eval author  -s task.json -o suites/support-agent
auto-eval suite   suites/support-agent  # read it back
auto-eval suite   suites/support-agent --case 5736ae   # one case in full
```

`profile` exits `2` when the agent cannot be run, and `author` refuses for the
same reason unless you pass `--force`. `--per-cell` sets how many cases each cell
of the grid asks for.

From Python:

```python
from auto_eval import build_suite, render_suite

suite = build_suite(spec)
print(render_suite(suite))
```

## The artefact

`author -o DIR` writes `suite.json` (everything, machine-readable) and
`suite.md` (the same thing as a document). The suite carries a `digest` over its
own content and a `spec_digest` over the spec it came from, and no timestamp —
so a number reported in six months can be traced to the exact cases that
produced it, and `verify_digest()` says whether the file has been edited since.

The split is decided here rather than during authoring, so a case's identity does
not depend on which side it lands. **Regression cases are held out
unconditionally** — a known failure used for tuning stops being evidence of
anything — and the rest split on a hash of the case id, which keeps each case on
the same side across rebuilds.
