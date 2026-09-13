# auto_eval

Builds evaluations for software and AI systems from a plain description of what
you want tested.

## Components

| | Component | Status |
| --- | --- | --- |
| 1 | [Classifier](classifier/) — free-form request → structured task spec | built |
| 2 | [Ground truth identifier](ground_truth/) — find public baselines and labelled data, and say what is behind each link | built |
| 3 | [Benchmark catalogue](auto_eval/benchmarks.py) — ten real public suites, matched onto a spec without a search | built |
| 4 | [Agent suite builder](suite/) — profile an agent, then write the cases and the checks for it | built |
| 5 | [Run infrastructure and dashboard](runner/) — run every case against the provider API, and show what the numbers are allowed to mean | built |

Two more to come.

```bash
auto-eval benchmarks                    # the catalogue
auto-eval benchmarks -s task.json       # the ones that fit your spec
auto-eval profile -s task.json          # what it takes to run the agent, and whether we can
auto-eval author  -s task.json -o suites/support-agent
auto-eval suite   suites/support-agent  # read the written suite back
auto-eval run     suites/support-agent  # run every case against the provider API
auto-eval runs                          # what has been run
auto-eval serve                         # the UI; the dashboard is at /dashboard
auto-eval fresh --model gpt-5           # ground truth this model cannot have memorised
```

Blocks 3 and 4 are both offline and deterministic: no API key, no network, same
answer every time. A suite carries a digest over its own contents, so a number
reported later can be traced to the exact cases that produced it.

## Running a suite, and what the numbers mean

Block 5 runs each case against the provider API with the agent's tools declared
but answered inside the harness, so nothing touches a live system. That makes
every trajectory check real — which tools were called, whether the run stopped
on its own, whether it stayed inside its ceilings, whether it followed an
instruction planted in a tool result — and it makes the end-state checks
impossible, because the suite's fixtures are specifications rather than
materialised environments.

Those checks are therefore reported as **blocked**, by name and with the reason,
rather than dropped. Dropping them is how "we could not check the end state"
becomes "the end state was fine". Every pass rate in a report sits next to the
count of what did not run, the headline is the held-out split, and a case whose
every check was blocked comes back blocked rather than passed.

```bash
auto-eval run suites/support-agent --mock    # the whole path, no API call
auto-eval run suites/support-agent --split held_out --samples 3
auto-eval report runs/support-agent-20260912-174501
```

A run writes `runs/<run-id>/run.json`, the same report as a document, and one
trace per run — the transcript each verdict was read off. `auto-eval serve` then
serves the dashboard at `/dashboard`: the runs on disk, one of them in full, and
every case openable down to the individual verdicts and their traces. It can
start runs too. Local only, no authentication.

## Ground truth the model has not already seen

A public suite that predates the model measures two things at once — whether the
agent can do the job, and whether the answers were in its training data — and
after the fact there is no separating them. [`contamination.py`](auto_eval/contamination.py)
holds published training cutoffs with the page each was read from, a catalogue
of suites that resist contamination (answers never published, refreshed on a
schedule, or simply released later), and the verdict for one suite: cases
harvested from your own runs or generated from your spec are clean by
construction, and a case adapted from a public suite is only as clean as that
suite's publication date. Every run report carries that verdict; `auto-eval
fresh --model <model>` prints the catalogue judged against one cutoff.

The catalogue is the deterministic counterpart to the ground-truth search: the
model is good at finding something specific to your task and bad at reliably
recalling whether a standard suite exists, so the standard suites are a lookup
table. No API key, no network, same answer every time. It holds no scores — a
stale leaderboard number is worse than none, so `analysis` quotes those from the
live page instead.

## Development

```bash
pip install -e ".[dev,web]"
pytest                     # the suite stubs the model client - no API key needed
ruff check . && ruff format --check .
mypy
```

`pre-commit install` runs the ruff checks on each commit, so CI is rarely the
first place a lint failure shows up.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same three
commands on every push and pull request, tests on 3.10 and 3.14 - the ends of
the supported range - and builds the wheel to check the CLI and the packaged UI
still work from a clean install.

Releases ([`.github/workflows/release.yml`](.github/workflows/release.yml)) are
cut by pushing a `vX.Y.Z` tag that matches `version` in `pyproject.toml`; the
built distributions are attached to a GitHub release. Publishing to PyPI is off
until the repository variable `PUBLISH_TO_PYPI` is set to `true` and a trusted
publisher is configured.
