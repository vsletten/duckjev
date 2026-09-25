# duckjev — state at hand-off and what comes next

Written 2026-09-24 13:20 PDT by the Claude Fable 5.1 session that ran the build;
§1.1 and §3.1 updated the same day by the Claude Opus 5.5 session that built `jev_extract`.
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

Then `jev_extract` (§3.1), in PR #2 on branch `jev-extract`; see §1.1.

Deviations from HANDOFF forced by DuckDB 1.5.5 or the data (all recorded in PR #1):
`numpy` is a runtime dependency; types come from `duckdb.sqltypes`; NULL handling is
`'special'`; the soft group-by is written `UNNEST(...) AS u(e)`; data is
`mteb/banking77` `test.jsonl` (3,080 rows; the parquet has 3,076); Markdown is
excluded from `ruff format`.

## 1.1 `jev_extract` (PR #2, 2026-09-24)

- `jev_extract(state, fields_json)` returns `MAP(VARCHAR, STRUCT(value, p, p_none,
  confidence, n_candidates, probabilities))`. Each row is one fused request with one
  Choice per field over that row's own candidates plus `none`. Candidates are
  stripped, deduped in order and capped at 254 (more raises).
- Fields with no candidates are not asked; a row with no candidates at all sends no
  request.
- Macros: `jev_field(instructions, candidates[, none])`, `jev_money_spans`,
  `jev_date_spans`, `jev_line_windows(lines, k)`.
- 60 offline tests (17 new).
- Benchmark `bench/sroie.py` on ICDAR 2019 SROIE receipts (`rth/sroie-2019-v2`,
  626 train as dev, 347 test held out). Results are in `docs/results/sroie.md`,
  generated from the committed run log `docs/results/sroie_runs.json`.
- Held-out test, 347 receipts, round R3:
  - candidate coverage 95.2%, exact 92.9%, selection given coverage 97.6%;
  - all four fields exact on 73.5% of receipts (baseline 46.4%);
  - $0.25 per 1,000 receipts, 114 receipts/s;
  - ECE 0.020 on covered fields.
- Total live spend for all rounds: $1.11.

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

### 2.1 Reading the round-one numbers


Written by hand after the round-one run (PR #1); moved here from `docs/results/banking77.md` before round two, because `bench/banking77.py report` now generates that file from the run log alone.

**Throughput and cost.** 3,080 rows took 20.9 s at client concurrency 16, with no
429 or 529 responses, so the rate limiter never engaged at roughly 148 requests/s.
Cost came in at $0.28 rather than the handoff's $0.10 estimate because a Choice
request over 77 glossed options is about 2,170 input tokens, not about 800: the
option list is re-sent with every row. Fusing several questions over one state
through `jev()` amortizes the state, but it cannot shrink a 77-option rubric; the
next lever there is shorter glosses (or none) and measuring what accuracy that costs.

**Calibration.** ECE is 0.074 over `confidence` and 0.079 over the top-1
probability. Across the middle bins the model is overconfident by about 0.1 to 0.2
(for example, mean top-1 p of 0.85 in bin 8 against 0.66 accuracy). The top bin
holds 77% of rows and is overconfident by about 0.055 (0.989 against 0.934).

**Soft vs hard.** Summed over intents, the expected counts are only slightly
closer to the truth than argmax counts (593.9 against 602 total absolute error),
and just 29 of 77 intents land within two standard errors. The reason is visible in
the per-intent table: the errors are systematic, not noise. The largest cases, from a confusion query over the cached
answers: `declined_card_payment` absorbs 16 `reverted_card_payment?` and 11
`declined_transfer` messages; `transfer_timing` absorbs 10 `pending_transfer` and 8
`transfer_not_received_by_recipient`; `transfer_into_account` absorbs 12
`topping_up_by_card` and 10 `top_up_by_bank_transfer_charge`; and
`beneficiary_not_allowed` loses 14 rows to `failed_transfer`. In each case the hard
count and the expected count move the same way by about the same amount. The
standard error √Σp(1−p) counts only the Bernoulli sampling variance of
independent, calibrated rows. It carries no term for per-intent model bias, so it
is far too narrow whenever that bias exists. Soft aggregation fixes the
thresholding loss that hard counts suffer on *calibrated* probabilities; it does
not fix a rubric the model reads differently from the annotators. On this corpus
the rubric disagreement dominates.

**`sem_where`.** The top-up Noul question returns 496 rows at p ≥ 0.5 (precision
0.72, recall 0.89) and an expected count of 621 ± 14 against 400 true. The gap is
mostly a definitional mismatch rather than a counting error. The probability mass
outside the ten top-up intents sits on `apple_pay_or_google_pay` (32.8),
`transfer_into_account` (32.3), `balance_not_updated_after_cheque_or_cash_deposit`
(24.3) and `supported_cards_and_currencies` (20.5): messages about adding money by
Apple Pay, bank transfer, cheque or a given card type, which a customer would
reasonably call topping up but the gold labels file under other intents.
Again, the stderr describes sampling noise under the model's own reading of the
question, not disagreement with a different labeling.

Total live spend for this benchmark: $0.018 (sample) + $0.281 (full split) + $0.038
(`sem_where`) = $0.337. The cache re-run and the demo soft group-by cost $0.

## 3. Next milestones, in priority order

### 3.1 `jev_extract` — done (PR #2); what the rounds taught

Shipped as `jev_extract(state, fields_json)`; numbers in §1.1 and
`docs/results/sroie.md`. What the six dev rounds showed, and the next levers:

- **Wording that names the exclusions** lifted company selection from 67% to 90%.
  The plain question picked the printed line with its registration number, and the
  address question picked the address with the company line in front.
- **Candidate order matters for nested spans.** Line windows nest (a 3-line window
  contains 2-line ones). Offering the longest first lifted address selection from 93%
  to 97%, while reversing the regex hit lists moved amounts and dates by 0.2 points at most. Make
  longest-first the default for window builders (a `jev_line_windows` option), and
  check order on any new nested builder.
- **Company has a label ceiling on SROIE, not a reading one.** R3 ("the registered
  name wins") and R5 ("the first-printed name wins") each got about 90%. They swap 44
  dev receipts for 45: Jev follows either rule, and the gold uses both. On a
  customer's corpus, state their convention in the question.
- **`none` never fired.** Every SROIE receipt has all four fields. Measuring the
  `none` option and `p_none` calibration needs a corpus with optional fields
  (invoices with and without PO numbers, due dates, tax ids).
- **Cost lever.** About 6,000 input tokens per request, most of it the ~104 address
  windows. Two stages (pick the first address line, then the block length), or a
  cheap Noul pre-filter on lines, should cut that several-fold. Measure against R3.
- **Coverage lever.** Address coverage is 84.7% strict and 94.2% loose. The gap is
  gold punctuation, not missing lines, so the builders are not the constraint here.
  Next corpus: invoices (FATURA, or a Kleister set), where the builders will matter.

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
