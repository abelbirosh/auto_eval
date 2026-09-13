# Examples

Starting points, not results. Copy them, change the names, and point
`auto-eval board` at your own.

| File | What it is |
| --- | --- |
| [`cohort-web-search.json`](cohort-web-search.json) | Six vendor configurations — Exa (auto and fast), Firecrawl, Tavily, Brave, TinyFish — plus three rows that hand one of those endpoints to the model as a tool. Each row names the `${VENDOR}_API_KEY` it needs; rows whose key is missing are reported as not run, so the file works whether you hold one key or six. Request shapes and response paths are from each vendor's own documentation. |
| [`cohort-tinyfish.json`](cohort-tinyfish.json) | One vendor, measured two ways: the endpoint alone, and the same endpoint handed to the model as a tool. |
| [`dataset-benchmark-facts.jsonl`](dataset-benchmark-facts.jsonl) | Five facts about AI benchmarks, each published in 2025 — after the cutoff of the models this project defaults to, which is what makes the model-only row score zero. |
| [`dataset-company-news.jsonl`](dataset-company-news.jsonl) | The shape to copy for your own items. The rows are placeholders; replace them with facts you can check. |
| [`request.txt`](request.txt) | A free-form request, for `auto-eval classify -f`. |

The prices in the cohort files are the published list prices at the time they
were written and they move. Check the vendor's pricing page before quoting a
cost column.
