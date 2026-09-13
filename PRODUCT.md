# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: someone choosing between providers — enrichment APIs, model endpoints,
agents — who needs a defensible comparison rather than a vendor's own numbers.
They arrive with a question ("which of these four is actually right more often?"),
not with a suite. They care about the headline comparison and about whether the
methodology holds up; per-case traces matter to them as proof that the headline
is real, not as their daily work.

Secondary, observed in the codebase rather than confirmed: the engineer who built
a system and runs Auto-Eval against it before shipping.

## Product Purpose

Turn a free-form evaluation request into a reproducible benchmark, and report the
result so the number cannot be read for more than it says. Success is a buyer
trusting a comparison enough to act on it, and being able to reproduce it months
later.

## Positioning

The mechanism a neighboring tool could not truthfully copy: every run reports what
it could *not* settle. Checks needing a real environment are surfaced as blocked
with a reason rather than dropped or guessed, so a pass rate always sits next to
the count of what did not run. Paired with contamination accounting — which cases
the model under test could already have seen, judged against its published cutoff
— and content-addressed suites that make a number traceable to the exact cases
that produced it.

## Operating Context

Run locally from a terminal: `auto-eval serve` opens the UI on localhost while the
CLI does the same work. Two surfaces — a classifier front door where the request
is described and sharpened, and a runs dashboard where suites are executed and
read. Runs are long enough that the dashboard is left open and returned to.
The API key lives in a local `.env` and never leaves the machine.

## Capabilities and Constraints

- Pipeline: classify → ground truth → author → run → grade.
- Suites are content-addressed and carry no timestamp: same spec, same bytes.
- Grading reads a recorded trace, not the final string, so trajectory is checkable.
- Rubric checks are settled by an LLM judge; deterministic checks by code.
- Tools are declared to the model but answered inside the harness — nothing
  touches a live system, which is exactly why end-state checks report blocked.
- `--mock` runs the whole pipeline calling no provider.
- No auth on the local server; it is not to be bound to a public interface.
  This is a deployment constraint, not a limit on design ambition.

## Brand Commitments

Name: Auto-Eval. Voice, as established in the README and module docstrings:
plain, exact, and willing to say what a number does not prove. Explains the
reasoning behind a decision rather than asserting it.

User-pinned visual constraints for this redesign: elegant and blocky; hard
rectangular edges, no rounded corners; no glassmorphism or translucent panels;
dark ground.

## Evidence on Hand

Real: the pipeline, the CLI, run reports on disk, the ten-benchmark catalogue in
`auto_eval/benchmarks.py`. No customers, testimonials, published case studies, or
pricing exist — future work must not invent them. The example request in the UI is
illustrative and is labelled as such.

## Product Principles

1. A number ships with what it excludes. Blocked is a result, not a gap.
2. Provenance over polish: every figure traces to the cases that produced it.
3. Say what was assumed. Inferred KPIs and derived fields are marked, not hidden.
4. The tool should be reproducible by someone who does not trust us.

## Accessibility & Inclusion

No product-specific standard was established. The result vocabulary — passed,
failed, blocked — must never be carried by color alone, since it is the one thing
every reading of this UI depends on.
