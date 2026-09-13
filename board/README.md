# Comparison boards

Sixth block. Block 5 asks *is this agent good enough*; this asks a different
question — **which of these systems is better at this job** — and the different
question changes the shape of everything else.

```bash
auto-eval board examples/dataset-benchmark-facts.jsonl -c examples/cohort-web-search.json
auto-eval boards                       # what has been run
auto-eval serve                        # or click "Run the board" on the page
```

Rows whose key is not in the environment are reported as not run, so the same
cohort file works whether you hold one vendor's key or six.

A board is one dataset, many systems, one row each:

| System | Endpoint & configuration | Accuracy | AR@1 | AR@5 | p50 | Total $ | $ / 1k correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exa auto | POST /search type=auto | 100.0% | 80.0% | 100.0% | 1.27s | $0.0350 | $7.00 |
| Exa fast | POST /search type=fast | 100.0% | 80.0% | 100.0% | 416ms | $0.0350 | $7.00 |
| Exa fast, model with tool | the model, with that endpoint as its only tool | 80.0% | — | — | 7.39s | $0.1352 | $33.80 |
| Firecrawl | POST /v2/search limit=10 | 80.0% | 20.0% | 60.0% | 944ms | — | — |
| Firecrawl, model with tool | the model, with that endpoint as its only tool | 40.0% | — | — | 14.94s | $0.0160 | $7.99 |
| model only (no search) | the model answering from memory | 0.0% | — | — | 2.88s | — | — |
| TinyFish | GET api.search.tinyfish.ai | 100.0% | 20.0% | 100.0% | 689ms | $0.0000 | $0.0000 |
| TinyFish, model with tool | the model, with that endpoint as its only tool | 100.0% | — | — | 9.10s | $0.0058 | $1.16 |

Five items is far too few to separate these rows and the board says so in its
own warnings — run it again and TinyFish moves between 80% and 100%. What five
items *can* show is the shape of the comparison: two endpoints at the same
accuracy and three times apart on latency, one endpoint that finds the answer as
often as another but a rank lower, and a tool row that costs twenty times its own
endpoint because the model searched seventeen times to answer five questions.

## The row nobody asks for

The first row is the point. **Model only** is the same model with no endpoint at
all, answering from memory. On a dataset of facts published after the model's
training cutoff it should score near zero — and when it does, every other row on
the board is measuring retrieval rather than recall. When it does not, the board
says so at the top and the number to fix is the dataset, not the vendor.

This is the same argument [`contamination.py`](../auto_eval/contamination.py)
makes about public suites, run as an experiment instead of an inference: rather
than reasoning about what the model might have seen, ask it.

## What is held constant

The items, the model that reads results, and the judge. Change any of those
between rows and the board is measuring that instead, so they are recorded on
the board and printed in its methodology. Rows are **alphabetical**: a board
sorted by its own headline invites the reader to treat that column as the
answer, and with accuracy, latency and cost on the same table it usually is not.

## Three shapes of row

| `kind` | What runs | What the score means |
| --- | --- | --- |
| `search_api` | the endpoint alone | retrieval quality: is the answer in the results, and how far down |
| `model_only` | the model, no endpoint | the baseline: what can simply be recalled |
| `model_with_tool` | the model, with that endpoint as its only tool | the agent case: including whether the model used what it was handed |

`AR@1` and `AR@5` are why the rank is kept: two vendors can both "find it" and be
different products if one puts the answer first and the other puts it fifth.

## The cohort file

Declarative, and secrets are not part of it:

```json
{
  "label": "Exa fast",
  "vendor": "Exa",
  "kind": "search_api",
  "configuration": "POST /search type=fast",
  "endpoint": {
    "method": "POST",
    "url": "https://api.exa.ai/search",
    "headers": { "x-api-key": "${EXA_API_KEY}" },
    "body": { "query": "{query}", "type": "fast", "numResults": 10 },
    "results_path": "results",
    "fields": { "title": "title", "url": "url", "snippet": "text" }
  },
  "price": { "per_call_usd": 0.005, "note": "$5 / 1k", "source": "https://exa.ai/pricing" }
}
```

`${EXA_API_KEY}` is read from the environment at call time and never stored in a
cohort, a board, or a log. A row whose key is missing is reported as **not run**,
with the reason — an unconfigured vendor scored as zero is a libel, not a
measurement. `{query}` is the only thing substituted into the request: the gold
answer never leaves the machine.

Two files ship as a starting point: [`examples/cohort-web-search.json`](../examples/cohort-web-search.json)
(three vendors and one tool-using row) and [`examples/cohort-tinyfish.json`](../examples/cohort-tinyfish.json)
(one vendor, both ways).

## The dataset

One JSON object per line — JSONL, a JSON array, or an object with an `items`
list. The common column spellings are accepted (`query`/`question`/`input`,
`answers`/`answer`/`gold`); anything else has to be renamed, because guessing
which column holds the answer is how a board ends up scoring the wrong thing.

```json
{"id": "1", "query": "On what date was ARC-AGI-2 released?", "answers": ["March 24, 2025", "2025-03-24"], "published": "2025-03-24", "url": "https://arcprize.org/..."}
```

`published` is when the fact became public, and it is what the contamination
check reads. `answers` should list every wording that counts: a scorer that
accepts only one spelling manufactures a difference between two systems that
both found the answer. Matching normalises case, accents, punctuation and
thousands separators — `1,200 employees` and `1200 employees` are the same
answer — and an item with no gold answer is settled by the judge, or reported as
unscored when there is none.

## What a board is careful about

- **An endpoint failure is an error, not a wrong answer.** Accuracy is over the
  items that came back, and the error count sits next to it.
- **A run that searches until it runs out of searches is a wrong answer**, not an
  error. The searches ran; what failed was the answering, and setting that aside
  would flatter the row by shrinking its denominator.
- **Spend is the published list price applied to the calls actually made.** A row
  with no price on file has blank cost columns rather than an estimate.
- **A short board says it is short.** Under thirty items the ordering is a hint.
- **Every verdict carries the span it was read off** — which result, and the text
  the answer was found in — so a three-point gap between vendors can be checked
  rather than believed.

## From the page

A spec whose subject is an endpoint, a model, or a retrieval API has no
trajectory, so the classifier page offers a board instead of a suite: paste the
items, paste or point at the cohort, and click **Run the board**. Which subjects
have a trajectory is read from `/api/health`, so the form on the page and the
gate on the server cannot drift apart and offer you a suite the server then
refuses to write.

The systems are filled in for you where the page can tell who they are. The
ground-truth search marks the sources published by a system under test — it has
to, because a vendor cannot be the ground truth for its own score — and that is
also the most reliable list of who is being compared, so each one arrives as a
row with its docs link, its `${VENDOR_API_KEY}`, and an empty endpoint URL to
fill in. A row still holding that empty URL is reported as **not run** with the
reason and the link, exactly like a row whose key is missing: it is never called
and never scored. What is still missing — items, endpoints, unreadable JSON — is
listed under the button before you press it rather than after.
