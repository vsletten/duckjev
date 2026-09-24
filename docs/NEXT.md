# duckjev — state at hand-off and what comes next

Written 2026-09-24 13:20 PDT by the Claude Fable 5.1 session that ran the build.
Read `docs/HANDOFF.md` first: it is the original build spec and still describes the
package design, the verified Jev API contract, the DuckDB UDF facts, and the repo
conventions. This file says what actually shipped, what the numbers mean, and the
next milestones in priority order.

## 1. What shipped (origin/main a91ad78, PR #1, 2026-09-24)

Tier one, complete against HANDOFF §7:

- `duckjev.register(con)` installs `jev` (fused, many questions over one state),
  `jev_noul` (macro over `jev_noul2` / `jev_noul3`, since DuckDB cannot overload
  Python UDFs), `jev_choice`, `jev_score`; macros `sem_where`, `expected_count`,
  `expected_count_var`, `expected_count_stderr`, `jev_argmax`, `jev_p`,
  `jev_runner_up`; `duckjev.usage()`, `flush()`, `cache_table()`.
- Arrow-vectorized execution: per-vector dedupe on the cache key, bounded async
  fan-out on a helper-thread event loop, backoff on 429/529, content-addressed cache
  in a separate DuckDB file (`~/.cache/duckjev/cache.duckdb`), model id in every key.
- 43 offline tests (fake transport, sockets blocked, key unset); ruff clean.
- Benchmark script `bench/banking77.py` with `bench/banking77_criteria.json` (77 one-line
  glosses), results in `docs/results/banking77.md`, headline numbers in the README.

Deviations from HANDOFF forced by DuckDB 1.5.5 or the data (all recorded in PR #1):
`numpy` is a runtime dependency; types come from `duckdb.sqltypes`; NULL handling is
`'special'`; the soft group-by is written `UNNEST(...) AS u(e)`; data is
`mteb/banking77` `test.jsonl` (3,080 rows; the parquet has 3,076); Markdown is
excluded from `ruff format`.

## 2. The numbers and what they mean

Banking77 test split, 3,080 rows, `jev-1.13.0`, concurrency 16:

| metric | value |
|---|---|
| throughput | 147.6 rows/s, 20.9 s wall, zero 429/529 |
| cost | $0.091 per 1,000 rows, about 2,170 input tokens per request |
| accuracy (argmax = gold, 77 options) | 0.841 |
| ECE, 10 bins | 0.074 over `confidence`, 0.079 over top-1 probability |
| Σ over intents of abs(hard − true) / abs(expected − true) | 602 / 594 |
| intents within 2 SE | 29 / 77 |
| cache re-run | 0.21 s, 0 requests, $0, identical rows |

Two things to say plainly when presenting this:

1. **Cost ran three times the estimate** because the 77 option descriptions ride on
   every request (~2,170 tokens instead of ~800). The levers are shorter criteria and
   question fusion (several questions per state in one `jev()` call is where the
   12x cost win in the TypeSafe cookbook comes from). Neither was tuned yet.
2. **Soft counts beat argmax only slightly** on this corpus because Jev's errors
   cluster by intent (16 `reverted_card_payment?` messages were labeled
   `declined_card_payment`). Calibration is good (ECE 0.074) but `√Σp(1−p)` measures
   only per-row noise, so it cannot see a systematic confusion between two options.
   The reliability table in the results doc is the honest picture. This is a
   corpus-shape result, not a ceiling: on a cleaner label space the estimator's
   advantage grows.

Neither point was tuned before the benchmark ran. Treat these as the baseline, not the
verdict. Before anyone quotes them externally, run round two: tighten the 77
glosses to the confusable pairs, try a two-level Choice (division, then intent,
per the classification-using-confidence cookbook), and re-measure.

## 3. Next milestones, in priority order

### 3.1 `jev_extract(state, candidates_json)` — select, don't generate
The strongest follow-on and the pattern that scales to large-scale extraction. Code
proposes candidate spans (`regexp_extract_all`, date/money/id patterns, a
dictionary); Jev picks per field from the candidates plus a `none` option; result is
a STRUCT of `(value, p)` per field. Implement as one fused `jev()` request per row
with one Choice per field. Benchmark on a public receipts or invoices corpus
(field-level exact match and coverage of the candidate builder, reported
separately). In the sibling email-poc repo this pattern scored 50/52 exact when the
candidate set covered the value, so candidate coverage is the metric to design for.

### 3.2 Round two on Banking77 (cheap, ~$0.30)
Criteria tightening on confusable pairs; two-level Choice (division → intent) with
the parent reported below confidence 0.9; fusion of `jev_choice` + a `sem_where`
Noul into one `jev()` request to show the cost lever. Update the results doc with a
rounds table; keep round one as the baseline row.

### 3.3 `sem_join` / `sem_dedup` / `sem_topk`
Blocked pairs from a key-equality join, then `jev_noul(struct_pack(...))` on each pair;
`sem_topk` via `jev_score` over a shortlist. Table macros. Benchmark on a small
entity-matching set.

### 3.4 Tier two: community extension
C++ or Rust extension template with the HTTP client inside the extension so
`INSTALL jev FROM community` works from the CLI, Node, Go, WASM; cost annotations so
the planner orders semantic filters last; `jev_explain(query)` reporting estimated
token spend before execution.

### 3.5 Tier three: maintained judgment columns
Sidecar table + view + `jev_refresh(table, column)` judging only new state hashes
under the pinned model; a view-matching rewrite so a repeated question over a stored
column becomes a column read.

## 4. Environment on the new machine

`uv sync --extra dev`; `uv run pytest -q`; `uv run ruff check .` and
`uv run ruff format --check .`. Live runs need `TYPESAFE_API_KEY` in the environment
and are scripts under `bench/`, never tests. Never print, log, or commit the key.
`bench/data/` is gitignored; the benchmark downloads once.

## 5. Housekeeping

- The merged `spike-banking77` branch still exists on origin; delete it or keep it.
- `docs/results/banking77.md` ends with a hand-written "Reading the numbers" section;
  `bench/banking77.py full` overwrites the file, so move that section into this doc
  or the README before re-running.
- README "further reading" should cite LOTUS (semantic operators), provenance
  semirings (Green, Karvounarakis, Tannen 2007) and probabilistic databases (Dalvi &
  Suciu; MystiQ/Trio/MayBMS): calibrated Jev answers are the input those systems
  never had.
