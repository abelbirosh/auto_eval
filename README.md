# auto_eval

Turn a free-form "can you test this for me" into a structured evaluation task
spec — before any eval gets built.

The classifier reads a request and pulls out three things:

1. **What we're testing** — the artifact, how it's invoked, what's in and out of scope.
2. **KPIs** — the metrics and qualities that decide success, with definitions
   precise enough to implement and thresholds where the user gave them.
3. **Evidence** — documentation, specs, datasets, and above all examples of
   successful runs and known failures.

Then it says what's missing. Anything the user didn't supply becomes an explicit
question rather than a guess, and a spec with a blocking hole is reported as
`insufficient` instead of being quietly built on.

## Install

```bash
pip install -e ".[dev]"
```

Classification calls the OpenAI API. Copy the example env file and paste your
key into it:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-proj-...
```

`.env` is gitignored. An exported `OPENAI_API_KEY` takes precedence over the
file, so a shell export still works if you prefer that. Two optional settings
live there too: `AUTO_EVAL_MODEL` to change the model, and `AUTO_EVAL_BASE_URL`
to point at an OpenAI-compatible endpoint.

## The web UI

The quickest way to try it. Install the extra and start the server:

```bash
pip install -e ".[web]"
```

```bash
auto-eval serve
```

Then open <http://127.0.0.1:8000>. Paste a request, hit **Classify** (or
&#8984;&#8617;), and you get the readiness verdict, the KPI table, the evidence,
and the open questions, with the task document and raw JSON collapsed underneath.
**Load example** fills the box with a realistic request if you just want to see
it work.

It binds to localhost and has no authentication — it's a test harness for your
own machine, not something to expose. Your key stays in `.env` and is only ever
read server-side.

## Use from the command line

```bash
auto-eval classify -f examples/request.txt
```

```bash
echo "test our RAG pipeline for hallucinations" | auto-eval classify
```

Write both artifacts out and print only what's still needed:

```bash
auto-eval classify -f examples/request.txt --md task.md --json task.json --questions-only
```

`classify` exits `0` when the spec is usable and `2` when it has blocking
questions, so a pipeline can gate on it. `auto-eval schema` prints the JSON
schema for `TaskSpec`, and `auto-eval models` lists the models your key can use.

From Python:

```python
from auto_eval import classify, render_markdown

spec = classify("our summarizer is too verbose and sometimes invents dates")
print(spec.readiness, [q.question for q in spec.open_questions])
print(render_markdown(spec))
```

## How it works

Classification runs on `gpt-5-mini` via structured outputs — it's an extraction
task on a short input that runs on every request, so it goes to the cheap tier.
Override with `--model` or `AUTO_EVAL_MODEL` if specs come back thin, and run
`auto-eval models` to see what your key can actually reach.

Because OpenAI's strict mode makes every field required, the model must emit all
twelve top-level fields rather than omitting the ones it found nothing for. An
empty `kpis` list means "found nothing", and the gap rules treat it as such.

The model extracts; it does not judge completeness. `auto_eval.gaps` recomputes
the open questions and the readiness verdict from rules after every
classification and discards the model's own guess, so "is this enough to build an
eval" stays deterministic and testable without a network call. Two of those rules
fire on every spec that lacks them: ask for documentation, and ask for examples of
runs that came out the way the user wanted.

The request itself is wrapped and treated as data — text inside it addressed to
the classifier is classified, not obeyed.

## Layout

| Path | What's in it |
| --- | --- |
| [auto_eval/schema.py](auto_eval/schema.py) | `TaskSpec` and friends — the contract everything downstream reads. |
| [auto_eval/prompts.py](auto_eval/prompts.py) | The extraction system prompt. |
| [auto_eval/config.py](auto_eval/config.py) | `.env` loading and provider settings. |
| [auto_eval/classifier.py](auto_eval/classifier.py) | The API call, error mapping, input limits. |
| [auto_eval/gaps.py](auto_eval/gaps.py) | Rule-based gap analysis and the readiness verdict. |
| [auto_eval/render.py](auto_eval/render.py) | `TaskSpec` → task document. |
| [auto_eval/cli.py](auto_eval/cli.py) | The `auto-eval` command. |
| [auto_eval/web.py](auto_eval/web.py) | FastAPI app behind the web UI. |
| [auto_eval/static/index.html](auto_eval/static/index.html) | The UI itself — one file, no build step. |
| [docs/task-spec.md](docs/task-spec.md) | Field-by-field reference for the spec. |

## Tests

```bash
pytest
```

The suite runs offline — the classifier and web tests drive fakes, and the gap
and render layers are pure functions. No test spends money.
