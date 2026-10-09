# duckjev — state at hand-off and what comes next

Written 2026-09-24 13:20 PDT by the Claude Fable 5.1 session that ran the build;
§1.1 and §3.1 updated the same day by the Claude Opus 5.5 session that built `jev_extract`;
§1.2, §2, §3.2 and §5 updated the same day by the Claude Fable 5.1 session that ran
Banking77 round two; §1.3, §2.2, §3 and §3.3 by the same session when it wrapped up tier one;
§1.4, §1.5, §3, §3.6 and §5 on 2026-09-27 by the Claude Opus 5.5 session that built the MAUDE demo.
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

## 1.2 Banking77 round two (PR #3, 2026-09-24)

- `bench/banking77.py` rewritten in the SROIE pattern: `prepare` (downloads, and a
  stratified dev split of train, 20 per intent, 1,540 rows), `run <round> --split dev|test`,
  `confusions`, `rescore` (each run rebuilt from its own answer cache through a transport
  that refuses requests), `report` (renders `docs/results/banking77.md` from the committed
  run log `docs/results/banking77_runs.json` alone). Pre-flight samples (`--limit`) and
  dry runs write their own suffixed files and record nothing; a full run refuses to start
  without a pre-flight of its round.
- Every round is one fused `jev(text, $questions)` request per message; the round decides
  the questions: flat (one 77-option Choice) or two-level (a division Choice plus one intent
  Choice per division, speculative, product distribution, parent reported under 0.9
  confidence), the gloss set (`v1`, structured `v2`, `short`), option order, and whether the
  top-up Noul rides in the same request.
- Six dev rounds, two held-out runs, $2.61 of live spend in total; the dev sample, division
  map and structured criteria come from label names and train only. Tests: 71 offline,
  including the whole pipeline against the fake transport and a check that the README
  headline matches the run log.

## 1.3 Tier one wrap-up (PR #4 and PR #5, 2026-09-25)

- PR #4: the answer-cache key hashes the request in the order it is sent (option, level
  and field order all reach the model), instead of sorted JSON.
- PR #5, version 0.2.0: `jev_pair`, `jev_match`, `sem_match` and the table macros
  `sem_join`, `sem_dups`, `sem_dedup`, `sem_topk` (table and column names as strings via
  `query_table` and the row-as-struct pattern; each blocked pair judged once, under a
  window barrier that keeps the optimizer from copying the pure UDF into the pushed-down
  filter). Benchmark `bench/entity_matching.py` on Abt-Buy and DBLP-ACM in the rounds
  discipline, with the macros themselves run live on a sample; results in
  `docs/results/entity_matching.md` from the committed run log. 91 offline tests.
- With §3.3 done, tier one (Python UDFs + macros, HANDOFF §7) is complete. What remains
  is tier two (§3.4) and tier three (§3.5), plus the follow-ons listed under each
  benchmark.

## 1.4 Plans (PR #6 and PR #7, 2026-09-25 and 2026-09-26)

`docs/TIER2.md`, the tier-two plan (§3.4), and `docs/MAUDE.md`, the build handoff for the
MAUDE demo (§1.5, §3.6).

## 1.5 The MAUDE demo (PR #8, 2026-09-27)

- `bench/maude.py` (`prepare`, `run`, `confusions`, `reading`, `rescore`, `report`, `demo`,
  each with `--dry-run`). `prepare` pulled 41,776 QBJ (the 8th and 22nd of each month),
  33,176 FTR and 24,619 LWS reports received 2025-07-01 to 2026-06-30 from openFDA, 41,748,
  30,832 and 24,521 of them eligible, plus 256 recall records; it wrote FDA's device-problem
  annex (491 terms with definitions and hierarchy) to `bench/maude_terms.json` and the split
  keys and option sets (40, 35 and 40 terms) to `bench/maude_ids.json`. The R1 glosses,
  aliases for FDA's editorial renames, and the R2 `not_for` phrases and examples are in
  `bench/maude_criteria_v2.json`. The phrases and examples were written from the gloss
  slice; the manufacturer conventions they target were also checked against pool-wide term
  counts that include the test reports (issue #10).
- Candidate frequencies and order came from the full eligible pool before the split,
  including the test reports' filed labels. The test reports were held out from round
  selection, but the candidate lists are test-aware.
- Seven dev rounds, the reading recorded before the held-out split (2026-09-27 06:27 UTC),
  held-out runs of R0 and the chosen R3, and the demo on QBJ; the run log is
  `docs/results/maude_runs.json` and `docs/results/maude.md` is generated from it. Total live
  spend $2.76 against the configured $3.00 token guard, every call recorded in the log.
  In-flight requests can overshoot the guard, so it is not a strict spend ceiling.
- Held out, 3,000 reports, R3 against R0: top-1 in set **0.826** ± 0.007 against 0.550
  (QBJ 0.850, FTR 0.797, LWS 0.832); harm accuracy **0.893** (macro 0.929) against 0.810
  (0.869); ECE 0.050 on the problem and 0.030 on the harm (R0 0.177 and 0.079); severity
  AUROC 0.973; $0.174 per 1,000 reports at 5,123 tokens per request, 107 reports/s at
  concurrency 16 (745 retried 429s). The reading had predicted 0.82, 0.90 and 0.05. At
  confidence 0.9, R3 answers 56% of reports at 0.947 top-1 in set.
- The package is unchanged (0.2.0). Tests: `tests/test_maude_bench.py` (15, offline, the
  whole pipeline on a dozen synthetic reports against a keyword fake transport) and the
  README check in `tests/test_results_docs.py`.

## 1.7 Keyless demo (2026-10-09)

- `python -m duckjev.demo` replays 138 recorded answers over 150 openFDA reports for
  continuous glucose monitors (QBJ, received 2026-01-08 and 2026-01-22) shipped in
  `duckjev/demo/`. It needs no key and sends no request. `--live` re-asks under a
  250,000-token cap. `bench/demo_data.py pull|record` rebuilds both files. The recording
  spent $0.0029, and a first attempt that failed on a SQL error after the judging step
  spent about as much again.
- What it shows on that slice: 88 of 150 reports describe a missed alert (87.1 ± 2.6
  expected), mostly Dexcom's G7 app sensor-failure-alert template (§3.6). Reports filed as
  Injury average 0.79 for "got care" against 0.04 for Malfunction.
- `tests/test_demo.py` runs the demo with sockets blocked and checks the recorded numbers,
  so a change to marshalling or the cache key that breaks replay fails offline.

## 2. The numbers and what they mean

Banking77 held-out test split, 3,080 rows, `jev-1.13.0`, concurrency 16, from
`docs/results/banking77_runs.json` (PR #3). R0 is the PR #1 config re-run; R3 is the round
the fixed rule chose on dev (best dev accuracy, ties to fewer tokens).

| metric | R0: one 77-option Choice, one-line glosses | R3: structured criteria on 29 confusable intents |
|---|---|---|
| accuracy (argmax = gold) | 0.841 ± 0.007 | 0.864 ± 0.006 |
| ECE, 10 bins, over `confidence` / top-1 p | 0.073 / 0.078 | 0.057 / 0.062 |
| Σ abs(hard − true) / Σ abs(expected − true) | 602 / 594.8 | 470 / 484.1 |
| intents within 2 SE | 29 / 77 | 34 / 77 |
| input tokens per request; $ per 1,000 rows | 2,172; $0.091 | 5,501; $0.231 |
| throughput at concurrency 16 | 171 rows/s, 0 × 429 | 81 rows/s, 2,459 × 429 retried |
| cache re-run | 0.14 s, $0 | 0.31 s, $0 |

What to say plainly when presenting this:

1. **The rubric was the lever, and it costs tokens.** Rewriting the 29 intents that
   appeared in the dev confusions as structured criteria (what / not_for / three examples
   from train) took held-out accuracy from 0.841 to 0.864 and ECE from 0.073 to 0.057. Those
   entries are 2.5× the request size, because the 77 option descriptions ride on every
   request; the state is a 20-token message. Short glosses (R5) cut tokens by 18% with no
   accuracy change, so the cost lever and the accuracy lever pull in opposite directions
   on this corpus: pay for detail only on the options that get confused.
2. **Fusion is nearly free, as the docs say.** The top-up Noul fused into the intent
   request (R4) cost 22 tokens per request against 294 for a separate `jev_noul` query, and
   changed neither the intent argmax (99.0% of rows identical to R0) nor the top-up
   probability (99.4% within 0.05, correlation 1.000). Reversing the option order (R3 vs R1)
   moved more rows, 3.6%, and 0.1 points of accuracy: order is noise for a flat list, unlike
   the nested spans in SROIE.
3. **The two-level Choice lost here.** Division accuracy of the explicit division question
   was 0.903, below the 0.932 the flat answer reaches by implication, and the product
   distribution over all eight branches recovered only part of that (0.811 against 0.849).
   Summing the flat distribution per division gives the same readout for free: 0.948
   division accuracy on held-out R3, 10.5% of rows deferred at mass 0.9, 0.898 intent
   precision on the rest. The 77-option list is well inside the 255 cap, so the hierarchy
   adds a decision with its own error rate and buys nothing back.
4. **Soft counts still do not beat argmax counts on this corpus**, and with R3 the expected
   counts total slightly more error than the hard counts (484 against 470). Calibration is
   good in aggregate, but per intent the residual probability mass leaks to the same
   neighbours row after row, so Σp accumulates that leak where argmax hides it. The
   Bernoulli standard error has no term for it; 34 of 77 intents are within 2 SE. The
   estimator's advantage needs per-intent calibration, or a cleaner label space.

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

### 2.2 Entity matching (PR #5)

Held-out test pairs, one `jev_match` Noul per pair, R0 the plain question over the JSON
pair and R2 the colleague-style question over labeled text lines:

| corpus | pairs (matches) | R0 F1 at 0.5 | R2 F1 at 0.5 | R2 precision / recall | AUROC | ECE | Σp vs true | $ / 1k pairs |
|---|---|---|---|---|---|---|---|---|
| Abt-Buy | 1,916 (206) | 0.898 | 0.915 | 0.894 / 0.937 | 0.996 | 0.022 | 229.6 ± 7.3 vs 206 | $0.0207 |
| DBLP-ACM | 2,473 (444) | 0.965 | 0.978 | 0.973 / 0.982 | 0.999 | 0.020 | 459.6 ± 7.6 vs 444 | $0.0190 |

The question wording that names the rules of the match is what moved these: F1 up, ECE
halved, and Σp from about 35% over the true count to about 10% over. The rest of the
overshoot is the lowest bin, where Jev leaves about 1.5% on clear non-matches. Order
matters for pairs (the shorter record first cost four points), the opposite of the flat
Banking77 list. Total live spend for the benchmark: $0.74.

## 3. Next milestones, in priority order

Tier one is complete through §3.3, and the MAUDE demo (§3.6) is done. The order of what
remains: the cheap follow-ons under §3.2, §3.3 and §3.6, then §3.4, then §3.5.

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

### 3.2 Round two on Banking77 — done (PR #3); what the rounds taught

Numbers in §2 and `docs/results/banking77.md`. Six dev rounds (1,540 rows, one input
change each), then the baseline and the chosen round on the held-out split:

| dev round | change | accuracy | tokens / req | $ / 1k |
|---|---|---|---|---|
| R0 | baseline, PR #1 config | 0.817 | 2,172 | $0.091 |
| R1 | structured criteria on the 29 confusable intents | 0.849 | 5,501 | $0.231 |
| R2 | R1 as division + per-division Choices, one request | 0.811 | 6,286 | $0.264 |
| R3 | R1 with every option list reversed | 0.850 | 5,501 | $0.231 |
| R4 | R0 with the top-up Noul fused in | 0.822 | 2,194 | $0.092 |
| R5 | R0 with short glosses | 0.819 | 1,772 | $0.074 |

- **Write the rubric for the confusions, with examples.** The dev confusion matrix
  named the pairs (pending vs failed top-up, direct debit vs card payment, transfer
  timing vs a transfer not showing); structured criteria with train examples resolved
  most of them. Some pairs are label noise in Banking77 itself (`beneficiary_not_allowed`
  messages that only say "error message"; `card_not_working` messages that say
  "declined at a restaurant") and no rubric moves them; R3's biggest held-out confusion
  is still `beneficiary_not_allowed → failed_transfer`, 21 rows.
- **The rule picked R3 over R1 by two rows.** Same tokens, 96.4% identical argmax; the
  order check is a check, not a lever, for a flat list.
- **Next levers, in order.** (a) Structured entries for the other 48 intents, and two
  examples instead of three, to find the accuracy-per-token frontier; the pre-flight
  makes each config a $0.01 question. (b) `null` descriptions on the intents whose names
  are unambiguous, the cheapest gloss there is. (c) Per-intent calibration on dev
  (a 77 × 77 confusion-aware correction of the distribution) so that Σp stops
  accumulating systematic leak; that is what would let expected counts beat argmax.
  (d) Concurrency 8 for requests over 5,000 tokens: at 16 the 250k tokens/s limit
  produced 2,459 retried 429s on the held-out run and halved throughput.
- **Two-level, if revisited:** give the division Choice examples of its own, or make the
  division the summed mass of the flat answer (free) and ask only the chosen division's
  intent question in a second, much cheaper request.

### 3.3 `sem_join` / `sem_dedup` / `sem_topk` — done (PR #5); what the rounds taught

Numbers in §2.2 and `docs/results/entity_matching.md`. Five dev rounds per corpus:

| dev round | change | Abt-Buy F1 at 0.5 | DBLP-ACM F1 at 0.5 |
|---|---|---|---|
| R0 | plain question, JSON pair | 0.930 | 0.964 |
| R1 | colleague-style question | 0.928 | 0.974 |
| R2 | R1 over labeled text lines | 0.941 | 0.978 |
| R3 | R2 with the records swapped | 0.897 | 0.970 |
| R4 | R1 with the guidance as Noul criteria | 0.939 | 0.974 |

- **The rules of the match belong in the question.** Naming what still counts as the same
  (wording, a missing field, a different price) and what makes it different (model
  number, variant, accessory) is the round that moved F1, ECE and Σp together.
- **Order is a lever for pairs.** The record with the fuller description goes first.
- **The pure-UDF trap.** DuckDB copies a pure scalar into a pushed-down filter, so a naive
  `SELECT ... WHERE p >= thr` over a judged subquery judges twice; the macros put the
  judged rows under `row_number() OVER ()`. Users writing their own SQL should judge into
  a table (`CREATE TABLE judged AS ...`) and filter that.
- **Next levers.** (a) A cheap pre-filter before the Noul: a string-similarity guard or a
  Noul over names only, then the full pair question on survivors, measured as coverage
  against the gold. (b) Per-intent-style calibration for the bottom bin, so Σp stops
  overshooting by the 1.5% left on clear non-matches. (c) Multi-field matching through
  STRUCT columns is supported but unmeasured; DBLP-ACM with `title` only against all four
  fields would show what each field buys. (d) `sem_dedup` inside large blocks is
  quadratic; a per-block cap or a two-stage dedup (cluster by a cheap key, judge within)
  is the follow-on before anyone runs it on a whole catalogue.

### 3.4 Tier two: community extension
Planned in full in `docs/TIER2.md` (2026-09-25): a C++ extension built beside tier one
under `extension/`, same wire format, cache key and SQL surface, accepted by replaying
the recorded benchmark caches through it with zero requests; the planner rule that runs
semantic predicates last and never twice; `jev_explain(query)`; then the platform matrix
and the community registry. Seven milestones, each one PR with its own gate. The
research it rests on (DuckDB 1.5.5 optimizer internals, the extension template, the
registry, and the LLM-calling precedents) is recorded there with what could not be
verified.

### 3.5 Tier three: maintained judgment columns
Sidecar table + view + `jev_refresh(table, column)` judging only new state hashes
under the pinned model; a view-matching rewrite so a repeated question over a stored
column becomes a column read.

### 3.6 The MAUDE demo — done (PR #8); what the rounds taught

Numbers in §1.5 and `docs/results/maude.md`. Seven dev rounds on 1,500 reports (500 per
code), one input change each; top-1 in set pooled and per code:

| dev round | change | top-1 in set | QBJ | FTR | LWS | harm acc. | tokens / report | $ / 1k |
|---|---|---|---|---|---|---|---|---|
| R0 | baseline: bare term strings, the description as the state | 0.570 | 0.455 | 0.448 | 0.812 | 0.809 | 900 | $0.038 |
| R1 | the official FDA definitions as `what` | 0.640 | 0.443 | 0.677 | 0.804 | 0.809 | 1,965 | $0.083 |
| R2 | R1 plus `not_for` and one gloss-slice example per option, both Choices | 0.803 | 0.858 | 0.717 | 0.833 | 0.899 | 3,786 | $0.159 |
| R3 | R2 with the state as an object (device names, manufacturer narrative) | 0.820 | 0.860 | 0.770 | 0.831 | 0.901 | 4,239 | $0.178 |
| R4 | R3 with the option order reversed (check) | 0.826 | 0.908 | 0.737 | 0.833 | 0.903 | 4,239 | $0.178 |
| R5 | R3 over the full 201-term vocabulary (check) | 0.787 | 0.840 | 0.724 | 0.798 | 0.902 | 10,127 | $0.425 |
| R6 | R3 as three separate requests (check) | 0.815 | 0.858 | 0.764 | 0.825 | 0.905 | 5,538 | $0.233 |

- **The label is the filer's convention; write it down.** Almost every R0 miss was a report
  read correctly and coded where its manufacturer does not file it: Dexcom files a missed
  sensor-failure alert under `Protective Measures Problem`, Mentor files capsular
  contracture under `Adverse Event Without Identified Device or Use Problem` where Allergan
  uses `Device Appears to Trigger Rejection`, Establishment Labs files a rupture as `Break`.
  Definitions alone moved FTR; `not_for` naming each option's confusable neighbours with one
  example from the gloss slice moved QBJ from 0.443 to 0.858; the brand name and the
  manufacturer's narrative in the state moved FTR another five points.
- **Candidates in code pay twice.** The full vocabulary cost 3.3 points at 2.4 times the
  tokens and a quarter of the throughput (1,647 retried 429s at concurrency 8).
- **Fusion is the cost lever.** Three requests instead of one: 31% more tokens per report,
  2.7 times the wall time, answers within half a point.
- **Order is per code.** Reversing the option lists moved QBJ up 4.8 points and FTR down 3.3.
- **Harm follows 803.3 once the rule is stated.** Saying that a lead capped, replaced or
  explanted with no complication is a serious injury took LWS Injury from 0.262 to 0.631 on
  dev; held out it is 0.570, and the rest are reports where the filer counted reprogramming
  or a recommended replacement as the intervention.
- **The demo.** On 200 QBJ test reports the trend flagged January 2026 for `Protective
  Measures Problem` (30.5 ± 0.7 expected against a trailing mean near zero): Dexcom filed its
  G7 app sensor-failure-alert reports that month, and `sem_join` matched them to the G7 and
  ONE+ app recalls (Z-2446-2025 to Z-2450-2025). Reports whose `remedial_action` says Recall
  matched at the same rate as the rest, because most are Abbott reports citing field action
  FA1002-2025, which is not among the 23 recall records openFDA holds for QBJ. `sem_dedup`
  judged 900 of 2,598 blocked pairs the same incident, most of them Abbott MedWatch reports
  whose narratives repeat verbatim on one event date.
- **Next levers, in order.** (a) The manufacturer's name in the state, or option
  descriptions per filer, for the conventions the narrative does not carry (Mentor against
  Allergan, Abbott US against UK, Dexcom's direction-free inaccuracy template). (b) Option
  order set per code. (c) A harm wording round for the reprogramming and replacement
  conventions, selected on the harm question. (d) Dedup pair states with identifiers (lot,
  source, device serial where published), since templated narratives cannot separate
  incidents. (e) A volume term in the signal rule's standard error, and the trend over a
  whole split rather than 200 reports. (f) The accuracy-per-token frontier: `not_for` trimmed
  to the pairs each code confuses, since R3's 5,123 tokens are mostly option descriptions.
  (g) A native object state: `jev()` takes a VARCHAR state, so R3 sent its object as JSON
  text; an object-valued state needs a JSON-typed state argument in the package and a new run.

## 4. Environment on the new machine

`uv sync --extra dev`; `uv run pytest -q`; `uv run ruff check .` and
`uv run ruff format --check .`. Live runs need `TYPESAFE_API_KEY` in the environment
and are scripts under `bench/`, never tests. Never print, log, or commit the key.
`bench/data/` is gitignored; the benchmark downloads once.

## 5. Housekeeping

- The merged `spike-banking77` branch still exists on origin; delete it or keep it.
- The answer-cache key is order-preserving since PR #4: reordering the options of a
  Choice, the levels of a Score or the fields of an object state is a new key. Entries
  written before that change whose insertion order differed from sorted-key order are
  never hit again (they were keyed on sorted JSON); deleting the default cache file
  reclaims only those orphaned entries.
- The entity-matching answer caches (`bench/data/cache_abt_*.duckdb`,
  `cache_dblp_*.duckdb`) and the demo cache live only in the `sem-join` worktree, like the
  Banking77 caches in `banking77-round-two`; `rescore` needs them.
- The MAUDE answer caches (`bench/data/cache_{dev,test}_R*.duckdb`, `cache_demo_QBJ.duckdb`),
  the raw openFDA pages, the pools and the FDA annex workbook are archived outside Git at
  `/mnt/data/vsletten/artifacts/duckjev/maude-pr8/data/`. Restore that directory to
  `bench/data/` for `rescore`; a re-pull may not reproduce the pools exactly.
- The PR #1 held-out run lives in `docs/results/banking77_runs.json` as the `pr1/R0`
  entry (flagged `legacy`, no answer cache), imported from its headline file
  `docs/results/banking77.json`, which PR #3 removed so that `report` reads one file.
- README "further reading" should cite LOTUS (semantic operators), provenance
  semirings (Green, Karvounarakis, Tannen 2007) and probabilistic databases (Dalvi &
  Suciu; MystiQ/Trio/MayBMS): calibrated Jev answers are the input those systems
  never had.
