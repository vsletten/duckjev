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
`jev_choice(state, q, criteria_json)`, `jev_score(state, q, levels_json)`,
`jev_extract(state, fields_json)` (select, don't generate: see below), and the
macros `sem_where`, `expected_count`, `expected_count_var`,
`expected_count_stderr`, `jev_argmax`, `jev_p`, `jev_runner_up`, `jev_field`,
`jev_money_spans`, `jev_date_spans` and `jev_line_windows`. Question specs
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

## Extraction: select, don't generate

`jev_extract(state, fields_json)` pulls fields out of text without letting a model
write them. SQL proposes candidate values for each row, and Jev picks one per field.
The answer is always a verbatim copy of a candidate, so it cannot invent a value or
transpose a digit. Each row is one fused request with one Choice per field, over
that row's own candidates plus a `none` option.

```sql
SELECT id,
       x['total'].value AS total, x['total'].p AS total_p,
       x['date'].value  AS issued
FROM (SELECT id, jev_extract(text, json_object(
        'total', jev_field('Which amount is the final total the customer paid, after tax '
                           'and rounding? Not the subtotal, the tax, the cash handed over '
                           'or the change.', jev_money_spans(text)),
        'date',  jev_field('On what date was this receipt issued?', jev_date_spans(text))
      )) AS x
      FROM receipts);
```

The result is `MAP(VARCHAR, STRUCT(value, p, p_none, confidence, n_candidates,
probabilities))`, keyed by field name:

- `value` is the chosen candidate. It is NULL when `none` wins, or when the field had
  no candidates, in which case it is not asked at all.
- `p` is the probability of what was returned, and `p_none` is the mass on `none`.
- `probabilities` is the whole distribution over the candidates.

`jev_field(instructions, candidates[, none])` builds one field's spec. Candidates can
be a list, or a JSON object mapping each candidate to a description. `none` is a
description, or `false` to drop the option. The builders over-find on purpose, since
Jev can only pick what it is offered:

- `jev_money_spans(text)` finds amounts with two decimals.
- `jev_date_spans(text)` finds numeric, month-name and compact dates.
- `jev_line_windows(lines, k)` returns every run of 1 to k consecutive lines, for
  multi-line names and addresses.

A Choice takes at most 255 options, so a field with more than 254 distinct
candidates raises an error. Narrow such a list in SQL with `candidates[1:254]`.

Candidate coverage is the number to design for. On the SROIE receipts benchmark
below, when the gold value is among the candidates, Jev picks it 97.6% of the time.

## SROIE receipts numbers

These come from live runs on 2026-09-24 against ICDAR 2019 SROIE: 973 scanned
Malaysian receipts, with the task-1 line transcriptions as text and the four task-3
key fields as gold. The data is `rth/sroie-2019-v2`, and the model is `jev-1.13.0`.
Six rounds were tuned on the 626 train receipts, and the round with the best dev
score ran once on the 347 held-out test receipts. Full tables, SQL and every round
are in [docs/results/sroie.md](docs/results/sroie.md). Nothing is trained, and the
input is the transcription, not the image, so these numbers are not comparable with
image-based SROIE leaderboards.

| held-out test, 347 receipts | coverage | exact | selection given coverage |
|---|---|---|---|
| company | 97.4% | 92.5% | 95.0% |
| date | 98.8% | 98.8% | 100.0% |
| address | 84.7% | 81.6% | 96.3% |
| total | 99.7% | 98.6% | 98.8% |
| **all fields** | **95.2%** | **92.9%** | **97.6%** |

All four fields were exact on 73.5% of receipts, against 46.4% for the untuned
baseline. Throughput was 114 receipts/s at concurrency 16 with no 429s. Cost was
**$0.25 per 1,000 receipts**, at about 6,000 input tokens per request. On covered
fields, calibration error is 0.020 (ECE, 10 bins). Answers with `p ≥ 0.99` cover
67% of fields at 98.0% exact.

Reading the numbers:

- **Most exact-match misses are coverage misses, and most of those are the gold.**
  Annotators typed the SROIE gold values, and in places they corrected a misprint
  (`SDN BHD` for the printed `SDN BND`) or added punctuation. For 56 of the 67
  held-out fields that no candidate matched, Jev returned the printed text the gold
  was typed from. Letters-and-digits matching lifts address from 81.6% to 91.1%.
- **Tuning was the inputs.**
  - Wording that says what to leave out (registration numbers, the company line in
    an address) lifted dev company selection from 67% to 90%.
  - Offering the longest runs of lines first lifted address selection from 93% to 97%.
  - Rebuilding the receipt into visual rows changed nothing.
  - Two hypotheses failed and are kept in the results doc: R4 asked for a trailing
    branch line in the address, and R5 changed the company preference.
- **R5 shows Jev following the rule it is given.** Switching the company question
  from "the registered name wins" to "the first name printed wins" gained 44 dev
  receipts and lost 45. Jev applied each rule consistently. The gold uses both
  conventions on receipts that print a shop name above a registered name, so that
  company ceiling comes from the labels.

## Banking77 numbers

These come from live runs on 2026-09-24 over the full Banking77 test split: 3,080
customer-support messages, 77 gold intents, `jev-1.13.0`, concurrency 16. The test split
was run only with the baseline and the round chosen on a dev split of train (1,540 rows,
20 per intent) by a fixed rule. Full tables, the six dev rounds, SQL and question text are
in [docs/results/banking77.md](docs/results/banking77.md), generated from the committed
run log `docs/results/banking77_runs.json`.

| metric | R0: one 77-option Choice, one-line glosses | R3: structured criteria on the 29 confusable intents |
|---|---|---|
| accuracy (argmax = gold) | **0.841** ± 0.007 | **0.864** ± 0.006 |
| ECE, 10 bins, over `confidence` / top-1 p | 0.073 / 0.078 | 0.057 / 0.062 |
| Σ over intents of abs(hard − true) | 602 | 470 |
| Σ over intents of abs(expected − true) | 594.8 | 484.1 |
| intents with abs(expected − true) ≤ 2·SE | 29 / 77 | 34 / 77 |
| input tokens per request | 2,172 | 5,501 |
| cost | **$0.091 per 1,000 rows** | **$0.231 per 1,000 rows** |
| throughput | 171 rows/s (0 × 429) | 81 rows/s (2,459 × 429, retried) |
| cache re-run | 0.14 s, 0 requests, $0 | 0.31 s, 0 requests, $0 |

What the dev rounds taught:

- Structured criteria (`what` / `not_for` / `examples` from train) on the intents in the
  top confusions are the lever: +3.2 points on dev and +2.3 on the held-out split, with
  better calibration, at 2.5× the tokens, because the option descriptions ride on every
  request and the message itself is about 20 tokens.
- A two-level Choice (division, then intent, all in one request) lost 3.8 points on dev.
  The division question was right 90.3% of the time while the flat answer already lands in
  the right division 93.2% of the time. Summing the flat distribution per division gives
  that division readout, and a deferral rule, with no second question.
- Reversing the option order moved 3.6% of rows and 0.1 points: order is noise for a flat
  list, unlike the nested spans in the SROIE benchmark above.
- Fusing the top-up Noul into the intent request cost 22 tokens per request against 294
  for a separate query, and moved neither the intent answers (99.0% same argmax) nor the
  top-up probabilities (99.4% within 0.05).
- Short glosses cut tokens by 18% at no accuracy cost.

Soft counts still do not beat argmax counts on this corpus. The remaining errors are
systematic by intent (for example `beneficiary_not_allowed` filed under
`failed_transfer`), the residual mass leaks to the same neighbours row after row, and the
Bernoulli standard error has no term for that; see the next section.

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
uv run python bench/banking77.py prepare && uv run python bench/banking77.py run R3 --split test  # live
uv run python bench/sroie.py prepare && uv run python bench/sroie.py coverage     # offline
uv run python bench/sroie.py run R3 --split test && uv run python bench/sroie.py report # live
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
