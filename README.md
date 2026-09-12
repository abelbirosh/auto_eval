# auto_eval

Builds evaluations for software and AI systems from a plain description of what
you want tested.

## Components

| | Component | Status |
| --- | --- | --- |
| 1 | [Classifier](classifier/) — free-form request → structured task spec | built |

Six more to come.

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
commands on every push and pull request, tests against Python 3.10 through 3.14,
and builds the wheel to check the CLI and the packaged UI still work from a
clean install.

Releases ([`.github/workflows/release.yml`](.github/workflows/release.yml)) are
cut by pushing a `vX.Y.Z` tag that matches `version` in `pyproject.toml`; the
built distributions are attached to a GitHub release. Publishing to PyPI is off
until the repository variable `PUBLISH_TO_PYPI` is set to `true` and a trusted
publisher is configured.
