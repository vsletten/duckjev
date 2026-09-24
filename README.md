# duckjev

**duckjev** installs *semantic operators* into DuckDB, backed by TypeSafe's
**Jev** System One model. Jev takes text `state` plus one or more typed questions
and returns typed answers with calibrated probabilities: a `noul` (probability of
yes), a `choice` (one option from a closed set, with a full probability
distribution and a confidence), or a `score` (a position on an ordered scale). It
never generates text.

That output shape is why this works where LLM-backed `ai_classify`-style functions
are awkward. A Jev answer is a STRUCT whose schema is known when the function is
registered, so it composes with the rest of SQL. Calibrated probabilities also
aggregate: `SUM(p)` over a table estimates how many rows have the property, and
it comes with a variance. A hard-label classifier gives you a count; a calibrated
one gives you an estimator with error bars, and DuckDB can express it in one query.

## Install

```bash
uv add git+ssh://git@github.com/vsletten/duckjev.git   # private repo
# or, from a checkout:
uv sync --extra dev
export TYPESAFE_API_KEY=...                             # only needed for live calls
```

```python
import duckdb, duckjev

con = duckdb.connect()
duckjev.register(con)   # installs functions, macros and the answer cache
```

`register()` installs `jev(state, questions_json)` (the fused primitive: many
questions over one state in one request), `jev_noul(state, q[, criteria_json])`,
`jev_choice(state, q, criteria_json)`, `jev_score(state, q, levels_json)`, and the
macros `sem_where`, `expected_count`, `expected_count_var`,
`expected_count_stderr`, `jev_argmax`, `jev_p` and `jev_runner_up`. Question specs
are JSON strings. Answers are cached by
`sha256(model, state, questions)` in `~/.cache/duckjev/cache.duckdb`, so re-running
a query is free. `duckjev.usage()` reports requests, tokens, cache hits, 429s and
estimated dollars. The API key is only read at the first call that needs the
network, so registering, macros and fully cached queries all work offline.

Options: `register(con, model="jev-1.13.0", concurrency=16, cache=True,
cache_path=None, max_input_tokens=None, api_key=None, base_url=None)`.
`max_input_tokens` is a job budget. Once billed input tokens pass it,
`JevBudgetExceeded` is raised.

## Three examples

A choice column, typed and inspectable like any other STRUCT:

```sql
SELECT text,
       jev_choice(text, 'Which team should handle this?',
                  '{"billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"}') AS team
FROM tickets;
-- team.choice, team.confidence, team.probabilities['billing'], jev_runner_up(team)
```

The soft group-by explodes each row's distribution into (option, p) pairs and sums
p per option. That gives expected rows per class instead of argmax counts:

```sql
SELECT e.key AS intent, SUM(e.value) AS expected_rows
FROM tickets t,
     UNNEST(map_entries(jev_choice(t.text, $instr, $criteria).probabilities)) AS u(e)
GROUP BY intent ORDER BY expected_rows DESC;
```

A semantic filter next to a calibrated count with its standard error:

```sql
SELECT count(*) FILTER (WHERE sem_where(text, $q, 0.5)) AS filtered_rows,
       expected_count(jev_noul(text, $q))              AS expected,
       expected_count_stderr(jev_noul(text, $q))       AS stderr
FROM tickets;
```

## Banking77 numbers

These come from a live run on 2026-09-24 over the full Banking77 test split: 3,080
customer-support messages and 77 gold intents, one 77-option Choice per row,
`jev-1.13.0`, concurrency 16. Full tables, SQL and discussion are in
[docs/results/banking77.md](docs/results/banking77.md).

| metric | value |
|---|---|
| throughput | **147.6 rows/s** (3,080 rows in 20.9 s; 0 × 429, 0 × 529) |
| cost | **$0.091 per 1,000 rows** ($0.281 total; about 2,170 input tokens per request) |
| accuracy (argmax = gold) | **0.841** |
| ECE, 10 bins | **0.074** over `confidence`, 0.079 over top-1 probability |
| Σ over intents of abs(hard − true) | 602 |
| Σ over intents of abs(expected − true) | 593.9 |
| intents with abs(expected − true) ≤ 2·SE | 29 / 77 (38%) |
| cache re-run | 0.21 s, 0 requests, $0 |

The honest reading: Jev is reasonably calibrated in aggregate, but on Banking77
its errors are *systematic by intent*. For example, it files reverted card
payments under `declined_card_payment` and pending transfers under
`transfer_timing`. Soft counts move with the hard counts there, so they beat
argmax only slightly, and the Bernoulli standard error, which has no term for
model bias, covers the truth for only 38% of intents.

## What is calibrated aggregation

If a model's probability p for "row has property X" is calibrated, then among rows
where it says 0.7, about 70% have X. Each row is then a Bernoulli trial with
success probability p. The sum of p over a table is an unbiased estimate of how
many rows have X, and it keeps the partial evidence that a 0.5 threshold throws
away. `expected_count(p)` is `SUM(p)`. `expected_count_var(p)` is `SUM(p·(1−p))`,
the variance of a sum of *independent* Bernoullis (the tuple-independence
assumption of probabilistic databases). `expected_count_stderr(p)` is its square
root. The interval covers only the randomness of the rows given the model. If the
model's probabilities are biased for some class, because it reads the rubric
differently from whoever labeled the data, the bias passes straight into the sum
and the standard error does not grow to show it. The Banking77 run above is a
worked example of that limit.

## Development

```bash
uv sync --extra dev
uv run pytest -q                 # offline: a fake transport, sockets blocked
uv run ruff check . && uv run ruff format --check .
uv run python bench/banking77.py sample && uv run python bench/banking77.py full   # live
```

Notes:

- DuckDB calls each Arrow UDF once per vector of about 2,048 rows. duckjev dedupes
  within the vector, serves cache hits, and fans the misses out concurrently with
  backoff on 429/529. Fusion across different questions only happens through
  `jev()`. Two typed calls in one query are two requests per row.
- With DuckDB 1.5, `numpy` is a runtime dependency because `create_function`
  refuses to register a Python UDF without it. NULL handling is `'special'`: a NULL
  argument or an empty state gives NULL without a request.

Design and decisions: [docs/HANDOFF.md](docs/HANDOFF.md).

State at hand-off and next milestones: [docs/NEXT.md](docs/NEXT.md). Original build spec: [docs/HANDOFF.md](docs/HANDOFF.md).
