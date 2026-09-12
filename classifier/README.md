# Classifier

Turns a free-form "can you test this for me" into a structured evaluation task
spec. First stage of the pipeline — everything downstream reads its `TaskSpec`.

It extracts three things and then says what's missing:

1. **What's being tested** — the artifact, how it's invoked, what's in and out of scope.
2. **KPIs** — metrics and qualities that decide success, with thresholds where given.
3. **Evidence** — docs, datasets, successful runs, known failures.

Anything the user didn't supply becomes an explicit question rather than a
guess. See [task-spec.md](task-spec.md) for the field-by-field contract.

Once the spec has no blocking questions left, the [ground truth
identifier](../ground_truth/) takes over and searches for public data and
baselines to anchor the eval on.

## Setup

```bash
pip install -e ".[dev,web]"
```

```bash
cp .env.example .env
```

Put your key in `.env` as `OPENAI_API_KEY`. It's gitignored. `AUTO_EVAL_MODEL`
and `AUTO_EVAL_BASE_URL` are optional overrides; an exported variable beats the
file.

## Run

```bash
auto-eval serve
```

The UI at <http://127.0.0.1:8000> is the quickest way to try it — paste a
request, hit Classify. Localhost, no auth; it's a test harness, not a
deployment target.

From the command line:

```bash
auto-eval classify -f examples/request.txt --md task.md --json task.json
```

`classify` exits `0` when the spec is usable and `2` when it has a blocking
question, so a pipeline can gate on it. `auto-eval models` lists what your key
can reach; `auto-eval schema` prints the `TaskSpec` JSON schema.

From Python:

```python
from auto_eval import classify, render_markdown

spec = classify("our summarizer is too verbose and sometimes invents dates")
print(spec.readiness, [q.question for q in spec.open_questions])
```

## How it works

Extraction runs on `gpt-5-mini` via OpenAI structured outputs. Strict mode makes
every field required, so the model returns empty lists rather than omitting
fields — an empty `kpis` means "found nothing".

**The model extracts; it doesn't judge completeness.** `gaps.py` recomputes the
open questions and the readiness verdict from rules after every run and discards
the model's own guess, so that decision is deterministic and testable offline.
Only one rule blocks — no KPIs, which leaves nothing to score. Everything else
comes back as a normal question and reads `needs_input`.

Two rules fire on every spec that lacks them: ask for documentation, and ask for
examples of runs that came out the way the user wanted. Those are what separate
an anchored eval from a guessed one.

The request is treated as data — text inside it addressed to the classifier gets
classified, not obeyed.

## Layout

| Path | What's in it |
| --- | --- |
| [schema.py](../auto_eval/schema.py) | `TaskSpec` — the contract everything downstream reads. |
| [prompts.py](../auto_eval/prompts.py) | The extraction system prompt. |
| [classifier.py](../auto_eval/classifier.py) | The API call, error mapping, input limits. |
| [gaps.py](../auto_eval/gaps.py) | Rule-based gap analysis and the readiness verdict. |
| [render.py](../auto_eval/render.py) | `TaskSpec` → task document. |
| [config.py](../auto_eval/config.py) | `.env` loading and provider settings. |
| [cli.py](../auto_eval/cli.py) · [web.py](../auto_eval/web.py) | The `auto-eval` command and the web UI. |

## Tests

```bash
pytest
```

80 tests, all offline — the classifier and web tests drive fakes, the gap and
render layers are pure functions. No test spends money.
