# Run infrastructure and the dashboard

Fifth block in the pipeline. Takes the suite block 4 wrote, runs every case
against the provider API, scores each run against the case's own checks, and
writes a report the dashboard reads.

```bash
auto-eval run suites/support-agent            # run it, write runs/<run-id>/
auto-eval run suites/support-agent --mock     # the whole path, no API call
auto-eval runs                                # what has been run
auto-eval report runs/support-agent-20260912-174501
auto-eval serve                               # then open /dashboard
```

## What "running an agent suite" actually means here

The suite describes an agent. What this block can drive is a model behind an
API. The substitution is made deliberately and recorded in every trace:

- **The tools are declared, never connected.** Each tool in the profile is
  offered to the model as a function, and every call it makes is answered inside
  the harness — from the case's own `stubs` where it specified one, and from a
  plain acknowledgement where it did not, marked as improvised. That is what
  makes a suite about refunds runnable without a payments account.
- **The ceilings come from the case.** A run still calling tools at its last
  allowed step is recorded as cut off, which is a different outcome from
  finishing, and the checks treat it as one.
- **The task is presented as a live task.** The model is not told it is being
  evaluated: a model that knows it is on camera is not the thing we want the
  number to be about. The report says so next to every number.

## Three kinds of check, and the third one is the point

| | Who settles it | Examples |
| --- | --- | --- |
| **Settled** | [`grade.py`](../auto_eval/grade.py), off the trace | tools called and not called, termination, ceilings, recovery after a scripted failure, a planted instruction ignored, a reply that parses |
| **Judged** | [`judge.py`](../auto_eval/judge.py), a model | the rubrics, one criterion at a time |
| **Blocked** | nobody — reported by name | end state, execution, containment, anything queued for a person |

A blocked check is one this harness cannot run, with the reason attached:

- **end state / execution** — the suite's fixtures are specifications, not
  materialised environments, so there is no end state to read. Build the fixture
  and the check becomes runnable.
- **containment** — every tool call is answered inside the harness, so no run
  *can* reach a live system, and none can demonstrate that it would not. Passing
  that would be evidence about the harness.
- **human review** — queued, with the trace stored next to it.

The easy way to report a run is to grade what is gradable and drop the rest,
which quietly turns *we could not check the end state* into *the end state was
fine*. So every pass rate here sits next to the count of what did not run, and a
case whose every check was blocked is reported as **blocked**, not as a pass.

## The two rules on the judge

1. **Quote or fail.** A ruling must carry a verbatim span from the transcript.
   `verify.accept` voids a pass with an empty quote; `judge.py` also voids a
   quote that is not in the transcript, which is the failure mode that matters —
   an invented quote reads exactly like a real one in a report.
2. **One criterion at a time.** The rubric's lines are put as questions, not as
   a score out of five. A judge asked for a number returns its taste.

The judge model is named next to the model under test, and the report warns when
they are the same: MT-Bench documents self-preference bias in exactly that setup.

## Contamination

[`contamination.py`](../auto_eval/contamination.py) asks what a score is worth
given what the model may already have seen. It holds published training cutoffs
(with the page each was read from), a catalogue of ground truth that resists
contamination, and a per-suite verdict: harvested and generated cases are clean
by construction, a case adapted from a public suite is only as clean as that
suite's publication date.

```bash
auto-eval fresh --model gpt-5     # the catalogue, judged against that cutoff
```

## What a run leaves behind

```
runs/<suite>-<timestamp>/
  run.json          # the report: scores, groups, warnings, every verdict
  run.md            # the same thing as a document
  traces/<case>-<sample>.json   # what each verdict was read off
```

`run.json` carries the suite digest and the spec digest, so a number can be
traced back to the exact cases that produced it, and the report says outright
when the suite no longer hashes to its own digest.

## From the page

The classifier page carries the pipeline to the end. Once the ground-truth stage
is done it offers **Run the benchmark**, which posts the spec to
`POST /api/benchmark`: that authors the suite inline — deterministic and offline,
so there is nothing to wait for — writes it to `suites/<name>/`, and starts the
run as a background job the page polls. Authoring and running are not two
decisions, so they are not two buttons.

A closed agent gate comes back as a `409` with its reason rather than an error,
and the page turns it into a question with a *write it anyway* button next to it.
The result renders in tables under the ground-truth section and links through to
`/dashboard?run=<run-id>`.

| | |
| --- | --- |
| `POST /api/benchmark` | spec → suite → run, returns a job |
| `POST /api/run` | run a suite already on disk, returns a job |
| `GET /api/run/{job}` | progress: done, total, the case in flight |
| `GET /api/runs` | every run on disk, newest first |
| `GET /api/runs/{id}` | one report; `/markdown` for the document |
| `GET /api/runs/{id}/trace/{case}?sample=` | what a verdict was read off |
| `GET /api/suites` | the suites the picker offers |

## The dashboard

`auto-eval serve`, then `/dashboard`. It lists the runs on disk, shows one in
full — the held-out number, what settled, what could not be checked, the
contamination verdict, then every case — and opens each case down to the
individual verdicts and the trace they were read off. It can also start a run:
pick a suite, a model, samples, and watch it go. `?run=<run-id>` opens straight
onto one.

Local only, like the rest of the UI: no authentication, and the key never leaves
the machine.

## The offline client

`--mock` runs the whole path — prompt building, the tool loop, the stubs,
grading, the judge, the report — against a canned client. It exists for the test
suite, which must not need a key, and for wiring up the dashboard: a run that
passes everything tells you as little about the UI as a run that errors out, so
the canned client fails a deterministic minority of cases in the two ways the
trajectory checks exist to catch — reaching for a forbidden tool, and never
stopping.
