# auto_eval

Builds evaluations for software and AI systems from a plain description of what
you want tested.

## Components

| | Component | Status |
| --- | --- | --- |
| 1 | [Classifier](classifier/) — free-form request → structured task spec | built |
| 2 | [Ground truth identifier](ground_truth/) — find public baselines and labelled data, and say what is behind each link | built |
| 3 | [Agent suite builder](suite/) — profile an agent, then write the cases and the checks for it | built |

Four more to come.

```bash
auto-eval profile -s task.json          # what it takes to run the agent, and whether we can
auto-eval author  -s task.json -o suites/support-agent
auto-eval suite   suites/support-agent  # read the written suite back
```

Block 3 is offline and deterministic: no API key, no network, same answer every
time. A suite carries a digest over its own contents, so a number reported later
can be traced to the exact cases that produced it.

Public benchmark suites are reported by the ground-truth search and never turned
into cases. A case pulled from a public suite is data the agent may well have
trained on, which makes a score against it a floor rather than a capability —
useful context for whoever is building the eval, not something to smuggle into
their suite.

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
