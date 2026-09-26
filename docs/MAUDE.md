# MAUDE demo: coded complaint surveillance on FDA device adverse events

Written 2026-09-26 by the Claude Fable 5.1 session that planned it, for the Claude Opus 5.5
session that builds it. Read `docs/HANDOFF.md` §1 to §4 (what duckjev is, the Jev API,
the package) and `bench/entity_matching.py` first: this benchmark has the same shape as
that one (a `prepare`, rounds on a dev split, one held-out run, a run log, a report
generated from the log, a `rescore`, a `demo` that runs the macros live on a sample) and
should reuse its code where the shapes match. This file says what the demo is, what data
it runs on, which questions it asks, how it is measured, and what is delivered.

## 0. What we are building and why

Every medical-device manufacturer must take each adverse event report, code it to the
FDA/IMDRF device-problem vocabulary, decide the harm, decide reportability inside 30 days,
deduplicate it against what user facilities and distributors filed about the same event,
and trend the counts per device and problem to catch a signal (21 CFR 820 as QMSR since
February 2026; EU MDR Articles 83 to 88 with the Article 88 trend-reporting duty). FDA
receives over two million such reports a year and publishes them, with the manufacturer's
own narratives and codes, as MAUDE, queryable through openFDA for free. A MAUDE row is
what a manufacturer's own complaint file looks like, so a pipeline that works on MAUDE
works on the buyer's data unchanged.

The demo is that pipeline in SQL, on tier-one duckjev, over a table pulled from openFDA:

1. **Code it.** One fused `jev()` request per report asks a Choice over the device-problem
   terms that occur for that device's product code (candidates in code), a Choice over the
   harm categories FDA uses, and a Score for severity. Typed, calibrated answers land in
   columns.
2. **Count it with error bars.** The soft group-by gives expected reports per device,
   problem and month with `expected_count_stderr`, which is the Article 88 trend number
   with the uncertainty a QMS record needs; a month is flagged when its expected count
   sits more than two standard errors above the trailing mean.
3. **Deduplicate it.** `sem_dedup` over reports in the same product code and event date
   asks whether two reports describe the same event, which is how manufacturer and
   user-facility filings of one incident collapse to one.
4. **Link it to action.** `sem_join` from reports to the recalls on file for the same
   product code asks whether the report describes the defect the recall addresses; the
   complement is the problem clusters with no recall.
5. **Rank it.** `sem_topk` by severity score gives the twenty reports a reviewer reads
   first.

The measured claim is the first step: accuracy and calibration of the problem code and the
harm category against the codes the manufacturers filed, tuned in rounds on a dev split
and reported once on a held-out split. Steps 2 to 5 are demo queries run live on a sample
and recorded with their usage, as the entity-matching `demo` command does. The cost line
writes itself: a report is a few hundred characters and the fused request between about
750 and 2,200 tokens depending on how much the options say, so a device family costs
cents and the whole 2026 stream to date, 2.5 million reports, costs between about $80 and
$230. `report` computes that projection from the chosen round's tokens per request.

The reason this beats a generic LLM call over the same rows: a published benchmark on
this exact coding task (MADE, arXiv 2604.15203) found LLM self-reported confidence poorly
correlated with accuracy. Calibration is what a System One judgment has by construction,
and the demo shows it as an ECE number next to the accuracy, plus expected counts whose
error bars mean something. Spice already exposes `ai()` for prose; this is the judgment
layer under it, and the README says so in one sentence without disparaging either.

## 1. Decisions

1. **Source: openFDA `device/event`** through the API, with the free key from
   `OPENFDA_API_KEY` when present (120,000 requests a day) and without it otherwise
   (1,000 a day, which the pull in §1.3 fits with room to spare). The bulk partitions
   are not needed for the benchmark; §2.4 records where they are for the "whole stream"
   number and for Spice later.
2. **Corpus: three product codes**, from the 2026 counts in §2.3: `QBJ` (continuous
   glucose monitor: the largest narrative stream, malfunction-heavy, software and
   readings problems), `FTR` (silicone gel-filled breast implant: injury-heavy, a 60-term
   vocabulary, rupture and rejection), `LWS` (implantable cardioverter defibrillator:
   malfunction and injury in balance, 134 deaths in 2026, sensing and shock problems).
   The corpus window is closed and fixed, `date_received` from 2025-07-01 to 2026-06-30,
   so late-arriving reports do not move the splits. The `prepare` command commits the
   sampled report keys to `bench/maude_ids.json` so the splits are reproducible even if
   openFDA re-serves the window differently.
3. **Pull and splits.** `prepare` pulls only reports that carry a "Description of Event
   or Problem" text (the search adds `mdr_text.text_type_code.exact:"Description of
   Event or Problem"`), with `limit=999` pages under a `date_received:asc` sort and the
   `Link: rel="next"` cursor (§2.1). For `FTR` and `LWS` it pulls the whole window
   (about 30,000 and 20,000 reports, under 60 requests each). For `QBJ`, whose window
   holds about 600,000, it pulls two fixed days per month (the 8th and the 22nd), about
   35,000 reports in about 40 requests. Reports whose description is under 100 characters or
   whose `event_type` is empty or "No answer provided" are dropped before sampling.
   Per code, dev is 500 reports and test is 1,000, drawn by a fixed rule (hash of
   `mdr_report_key`, ordered, taken) stratified by `event_type` with a floor of 50 per
   harm category where the code has that many, so Death and Injury are measurable where
   they exist (`FTR` has three deaths in 2026; the floor does not apply there). A gloss
   slice of 200 reports per code, disjoint from both, is the only place examples for
   the structured criteria may be drawn from. Test is run only with the baseline and
   the round chosen on dev.
4. **State.** The report's "Description of Event or Problem" text as published: upper
   case where it is upper case, redaction tokens like `(B)(4)` and `(B)(6)` kept,
   retractions ("submitted in error") kept, nothing cleaned. Rounds decide whether the
   device names and the "Additional Manufacturer Narrative" text ride along as an object
   state (§3.2, R3).
5. **Questions**, all in one fused `jev()` request per report:
   - `problem`: a Choice over the option set for the report's product code (§1.6), plus a
     catch-all option `some other problem` (bare in R0, glossed from R1 on). Measured.
   - `harm`: a Choice over FDA's `event_type` categories `Death`, `Injury`,
     `Malfunction`, `Other`, with glosses written from the 21 CFR 803.3 definitions
     (serious injury, malfunction) and the yaml's own words for Other. Measured.
   - `severity`: a Score over five ordered levels from no patient harm to death. Readout,
     validated only as AUROC for Death-or-Injury versus Malfunction.
6. **Candidates in code.** The option set for a product code is its 40 most frequent
   device-problem terms in the corpus window, dropping any with fewer than five reports,
   in descending frequency order, plus the catch-all. That is roughly the whole
   vocabulary for `FTR`, most of it for `LWS`, and the head of a long tail for `QBJ`.
   Rows whose filed terms all fall outside the set are kept in the split but excluded
   from problem accuracy and reported as the coverage of the option set. The full
   vocabulary (485 terms seen in 2026) is the ablation R5, not the default. Two legacy
   FDA strings that occur as labels but are not in the published vocabulary, "Adverse
   Event Without Identified Device or Use Problem" and "Appropriate Device Problem
   Term/Code Not Available", are options like any other when they make the top 40, with
   hand-written glosses in the criteria file since the spreadsheet has none.
7. **Ground truth and the multi-label rule.** `product_problems` is a list. The primary
   metric, `top-1 in set`, counts a row correct when the argmax is one of its filed terms;
   `strict` accuracy is exact match on rows with exactly one filed term, reported beside
   it; `set mass` is Σp over the filed terms, the calibration readout (for a calibrated
   answer its mean tracks `top-1 in set`). Harm accuracy is plain argmax equals
   `event_type`.
8. **Deduplication and the join are demo queries, not measured**: MAUDE carries no
   duplicate labels and no report-to-recall labels. They run live on a sample, with a cap
   on the block sizes as the entity-matching demo does, and their usage is recorded.
9. **Budget.** A fused request is about 750 tokens with bare terms and about 2,200 with
   definitions on every option, so a 1,500-row dev round costs five to fourteen cents.
   The discipline is kept for its own sake: the 40-row pre-flight before each round's
   first full run, `--max-usd` (default $0.50) as `max_input_tokens` on every run, a
   hard stop at $3 for the whole benchmark including the demo. Report spend in every PR.
10. **Nothing hedged about the model** in anything committed. A poor number is an input
    problem first (wording, option set, state layout, order, split), fixed in the next
    round; the held-out number is published once, from the chosen round.

## 2. Ground truth (verified 2026-09-26)

Every number below came from a live call to `api.fda.gov` on 2026-09-26 (index
`meta.last_updated` 2026-09-08) or from the linked documentation page. "2026" means
`date_received` from 2026-01-01 to 2026-09-26, 2,503,728 reports.

### 2.1 Query mechanics

- Endpoint `https://api.fda.gov/device/event.json`. Docs: https://open.fda.gov/apis/
  (`query-syntax`, `paging`, `query-parameters`, `advanced-syntax`, `dates-and-ranges`,
  `authentication`, `device/event`).
- `limit` is at most 1,000, and **1,000 itself returns 403 `API_KEY_MISSING` without a
  key**; 999 works. `skip` is at most 25,000. Beyond that, page with `search_after`: sort
  (`sort=date_received:asc`), omit `skip`, follow the response's `Link: <...>;
  rel="next"` header until it is absent. The cursor value is the sort key in epoch
  milliseconds plus `mdr_report_key` as tiebreak. Adding `skip` to a cursor URL is a 400.
- `count=<field>.exact` returns 100 buckets by default; `limit=999` returns all of them
  (485 distinct `product_problems` terms in 2026). `count` honors the `search` filter.
- Dates: `date_received:[20250701+TO+20260630]`, inclusive, either date format. Exact
  match: `device.device_report_product_code:QBJ` (case-sensitive) and
  `product_problems.exact:"Low Readings"` (the whole string; a substring is a 404).
  `+AND+` combines; a bare `+` is OR; parentheses group.
- No rate-limit headers are sent. Documented limits: 240 requests a minute; 1,000 a day
  per IP without a key, 120,000 with one (free, instant, from open.fda.gov). 98 requests
  over 25 minutes drew no 429.
- A query that returns no records is a **404**, not an empty list; `prepare` treats 404
  as zero rows.

### 2.2 Fields (https://open.fda.gov/fields/deviceevent.yaml)

- `mdr_report_key`: the report id. `report_number`: varies by source, empty for user
  facilities. `event_key`: undocumented ("documentation forthcoming"). Dates are
  `YYYYMMDD` strings: `date_received`, `date_of_event`, `date_report`.
- `event_type`: `Death`, `Injury`, `Malfunction`, `Other`, plus `No answer provided` and
  empty (26 rows in 2026; dropped). 2026: Malfunction 1,661,107; Injury 831,654; Death
  9,350; Other 1,591.
- `product_problems`: a list of vocabulary strings (§2.5). In the 100 most recent `QBJ`
  reports every row had exactly one term; the multi-label rule of §1.7 still stands for
  the other codes.
- `mdr_text[]`: `text_type_code` is `Description of Event or Problem` (2,103,363 rows in
  2026, the state) or `Additional Manufacturer Narrative` (1,657,089, the R3 addition);
  the yaml also lists `Manufacturer Evaluation Summary`, unseen live. Description length
  in a 100-row `QBJ` sample: min 203, median 327, max 543 characters; an earlier 20-row
  sample across codes had median 587. No redaction tokens in that sample; keep them if
  they appear.
- `device[]`: `device_report_product_code` (the three-letter code), `generic_name`,
  `brand_name` (`NA` for reprocessed single-use devices), `manufacturer_d_name`, and
  `openfda.device_name`, `openfda.device_class`, `openfda.medical_specialty_description`.
  `generic_name` has case variants (`ENDOSSEOUS DENTAL IMPLANT` and `Endosseous Dental
  Implant` are separate buckets); the product code is the clean key.
- `report_source_code`: Manufacturer report (2,363,201 in 2026), Distributor report
  (125,433), Voluntary report (11,321), User facility report (3,773). `type_of_report`:
  Initial submission, Followup, Extra copy received, Other information submitted.
  `source_type`: who told the manufacturer (Consumer, Health Professional, User facility,
  Literature, Study, ...).
- `patient[].sequence_number_outcome`: the yaml enumerates words (Required Intervention,
  Hospitalization, Life Threatening, Death, Disability, Congenital Anomaly, Other, ...),
  live data also carries single letters with a leading space (` R`, ` H`, ` L`, ` D`,
  ` S`, ` O`, ` C`), and 1,692,282 rows in 2026 are empty. A possible second severity
  ground truth; not used in this benchmark.
- `remedial_action`: `Recall` on 226,944 reports in 2026, `Notification` on 84,978, empty
  on 2,203,415. A weak but free check on the join demo: reports flagged `Recall` should
  match a recall more often than the rest (§3.4).
- `adverse_event_flag` (Y on 844,570) and `product_problem_flag` (Y on 1,775,398) exist;
  not used.

### 2.3 The three codes, 2026 counts

| code | device | reports | with Description | Malfunction / Injury / Death / Other | source | distinct problem terms; head of the list |
|---|---|---|---|---|---|---|
| QBJ | Continuous Glucose Monitor | 462,012 | 370,132 | 448,721 / 13,004 / 103 / 184 | Manufacturer 460,583; Voluntary 1,429 | many; Low Readings 128,514; Incorrect, Inadequate or Imprecise Result or Readings 115,768; Wireless Communication Problem 74,887; Protective Measures Problem 57,309; Detachment of Device or Device Component 12,476; Appropriate Device Problem Term/Code Not Available 10,583; Device Alarm System 9,796; High Readings 9,132; No Device Output 8,311; Unintended Application Program Shut Down 7,253 |
| FTR | Prosthesis, Breast, Silicone Gel-Filled | 22,231 | 21,257 | 315 / 21,896 / 3 / 17 | Manufacturer 21,946; Voluntary 278; User facility 7 | 60; Material Rupture 9,822; Device Appears to Trigger Rejection 6,520; Adverse Event Without Identified Device or Use Problem 4,451; Migration 984; Patient Device Interaction Problem 698; Malposition of Device 688; Gel Leak 563; Break 352 |
| LWS | Implantable Cardioverter Defibrillator (Non-CRT) | 15,554 | 14,131 | 6,928 / 8,490 / 134 / 2 | Manufacturer 15,377; Voluntary 173; User facility 4 | Over-Sensing 4,836; Inappropriate/Inadequate Shock/Stimulation 3,293; Signal Artifact/Noise 2,576; High impedance 2,295; Adverse Event Without Identified Device or Use Problem 2,109; Failure to Read Input Signal 1,298; Pacing Problem 1,097; Device Sensing Problem 1,006; Fracture 907; Impedance Problem 893; Under-Sensing 858 |

The 2026 top ten product codes by volume, for the record: DZE dental implants 511,733;
QBJ 462,012; QFG insulin pumps 274,209; FPA IV administration sets 146,000; OZP automated
insulin dosing 80,224; QLG flash glucose monitors 68,751; OSR pacemaker/ICD non-implanted
components 48,151; BZD ventilators 47,447; BRY automated dispensing cabinets 38,863; NAY
bipolar forceps 36,466. Dental implants were passed over because two terms (Failure to
Osseointegrate, Loss of Osseointegration) cover almost everything.

### 2.4 Bulk partitions

`https://api.fda.gov/download.json` lists `device.event`: export date 2026-09-25,
26,136,889 records, 371 zip partitions named by quarter
(`https://download.open.fda.gov/device/event/{yyyy}q{n}/device-event-NNNN-of-NNNN.json.zip`,
1991q4 to 2026q3, about 18 GB in total, 125 partitions under 50 MB). Each holds
`{"meta": ..., "results": [...]}`. This is the "whole stream" path for Spice or for a
full-year run later; the benchmark does not use it.

### 2.5 The problem vocabulary

FDA publishes the coding vocabulary as one spreadsheet, `FDA Annexes A-G August 11
2026.xlsx`, at https://www.fda.gov/media/192166/download?attachment (linked from
https://www.fda.gov/medical-devices/mdr-adverse-event-codes/coding-resources-medical-device-reports).
Sheet `A` is the device-problem annex: a preamble, then a header at row 8 with `Level 1
Term | Level 2 Term | Level 3 Term | IMDRF Code | FDA Code | NCIt Code | Definition |
Non-IMDRF Code | Status | Status Description | CodeHierarchy`; 491 rows (27 level-one, 180
level-two, 284 level-three), **every row with a definition**, hierarchy explicit
(`A01|A0102|A010201`), 4 rows retired and 43 modified. The term strings match the
`product_problems` strings openFDA returns exactly (checked: Wireless Communication
Problem A1305, Failure to Osseointegrate A010201, Break A0401, Low Readings A090808).
Two openFDA label strings are not in the file: `Adverse Event Without Identified Device
or Use Problem` (136,231 uses in 2026) and `Appropriate Device Problem Term/Code Not
Available` (21,303). `prepare` reads sheet A with DuckDB's `read_xlsx` (the `excel`
extension, autoloaded; `header` and `range` options skip the preamble) and writes
`bench/maude_terms.json` (term, level, parent, IMDRF and FDA codes, definition, status),
which is committed because the criteria rounds and the tests read it.

### 2.6 Recalls for the join

`https://api.fda.gov/device/recall.json` (fields
https://open.fda.gov/fields/devicerecall.yaml) carries `product_code`,
`product_description`, `reason_for_recall`, `root_cause_description`,
`event_date_initiated` (ISO date), `recalling_firm`, `k_numbers`, `pma_numbers`,
`product_res_number`. `device/enforcement.json` has no usable product code (the field is
in its yaml but absent from live records) and is not used. `QBJ` has 23 recall records
in total, 4 initiated in 2026; an example: Dexcom G6, "A software defect in version
v1.15.0 of the G6 Android app...", root cause "Software Design Change". On this endpoint
`count=product_code` works and `count=product_code.exact` is a 404.

## 3. The benchmark

### 3.1 Files

```
bench/maude.py                 # prepare | run | confusions | rescore | report | demo
bench/maude_ids.json           # sampled report keys per code and split (committed by prepare)
bench/maude_terms.json         # the FDA device-problem vocabulary with definitions (committed by prepare)
bench/maude_criteria_v2.json   # the R2 overlay: not_for and examples per term, plus glosses for the
                               # legacy strings and the catch-all; written from the gloss slice only
bench/README.md                # a MAUDE section, same shape as the others
docs/results/maude_runs.json   # the run log; every full live run and every demo
docs/results/maude.md          # generated by `report` from the run log alone
tests/test_maude_bench.py      # offline: configs, metrics, tiny-corpus pipeline
tests/test_results_docs.py     # gains the README headline check for MAUDE
bench/data/                    # gitignored: raw JSON, parquet slices, caches, judged tables
```

### 3.2 Rounds

One input change per round. `selectable=False` marks checks that are not candidate
configs. The chosen round is the selectable round with the best dev `top-1 in set` on the
problem question, ties to fewer tokens per request; the harm number is reported for the
same round.

| Round | Change | Selectable |
|---|---|---|
| R0 | baseline: per-code option set with bare term strings, narrative-only state, three questions fused | yes |
| R1 | the official term definitions as structured criteria (`what` only) on every option | yes |
| R2 | R1 plus `not_for` and one example per option: `not_for` names the option's confusable siblings under the same parent term in the hierarchy, the example is one sentence written from the gloss slice; both in `bench/maude_criteria_v2.json` | yes |
| R3 | the best of R0 to R2 with the state as an object: narrative, device brand and generic name, the manufacturer's additional narrative when present | yes |
| R4 | the best round with the option order reversed: the order check | no |
| R5 | the best round with the full vocabulary (every term seen on the corpus, up to 255) instead of the per-code set: the candidates-in-code ablation | no |
| R6 | the best round as three separate requests instead of one fused: the fusion cost check | no |

The `Round` dataclass carries `note, criteria (none | what | full), state (text | object),
order (given | reversed), vocabulary (code | global), fused (bool), selectable`. Questions
are built by one function from the round and the product code, as `questions_for` does in
`bench/banking77.py`.

### 3.3 Metrics per run

Per corpus code and pooled: `top-1 in set`, `strict`, `set mass`, harm accuracy, harm
accuracy per category and its macro average (Malfunction dominates `QBJ` and Injury
dominates `FTR`, so the plain number alone would flatter), severity AUROC, ECE over
`confidence` and over top-1 p for both
Choices (10 equal-width bins, reliability table), deferral curve (share of rows and
accuracy above confidence thresholds 0.5 to 0.95), option-set coverage, the top ten
confusions for the problem question (`confusions` command, offline), tokens per request,
$ per 1,000 reports, reports per second, 429 count, cache re-run seconds and identical
rows. Same run-log entry shape as the entity-matching bench (`corpus, round, split, limit,
dry_run, round_config, model, duckjev, duckdb, python, concurrency, timestamp, seconds,
usage, ...`).

### 3.4 The demo command

`demo --code <code> --n 200`: on 200 sampled test reports of one code, run the demo
queries of §4 live with the chosen round's questions, record each query's usage, rows in
and out, and the trend table, under `demo/<code>` in the run log. Block caps: dedup pairs
within the same code and `date_of_event`, at most 5 other reports per block; the join
uses every recall on file for the code (`QBJ` has 23), so 200 reports make at most
4,600 pairs, about $0.12. The join section also reports the match rate for reports whose
`remedial_action` includes `Recall` against the rest, the one free sanity check the data
offers. Dry-run works through the fake transport.

### 3.5 Offline tests

`tests/test_maude_bench.py`, mirroring `tests/test_entity_matching_bench.py`: round and
corpus configs well formed; `questions_for` for every round (option set, catch-all, order
reversal, structured criteria shape, object state); the multi-label metric rule on a
hand-built judged table; the chosen-round rule; a tiny synthetic corpus of a dozen
reports through `prepare --from-fixture`, dry-run, pre-flight gate, `demo --dry-run`,
`rescore`, `report`. The fake transport answers by keyword (a problem term named in the
narrative gets 0.7) so the tiny pipeline has deterministic numbers. Tests never touch the
network; `prepare` is the only code that calls openFDA and it is not exercised by tests
except through the fixture path.

## 4. The demo queries

All over `judged` (one row per report; `problem`, `harm` and `severity` are struct
columns unpacked from the fused `jev()` JSON answer the way `bench/banking77.py`
flattens its answers; `prepare` has already cast the `YYYYMMDD` strings to DATE columns)
and `recalls` (the product code's recall records).

```sql
-- 1. Expected reports per problem and month, with error bars: the trend table.
SELECT product_code, e.key AS problem, date_trunc('month', date_received) AS month,
       expected_count(e.value) AS expected, expected_count_stderr(e.value) AS se,
       count(*) FILTER (WHERE jev_argmax(problem) = e.key) AS argmax_count
FROM judged, UNNEST(map_entries(problem.probabilities)) AS e
GROUP BY ALL ORDER BY product_code, problem, month;

-- 2. Signal months: expected count more than two SE above the trailing three-month mean.
WITH t AS (SELECT * FROM trend)
SELECT * FROM (
  SELECT *, avg(expected) OVER w AS trailing, sqrt(sum(se * se) OVER w) / 3 AS trailing_se
  FROM t WINDOW w AS (PARTITION BY product_code, problem ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING))
WHERE expected > trailing + 2 * sqrt(se * se + trailing_se * trailing_se);

-- 3. One event, one row: manufacturer and user-facility filings of the same incident collapse.
SELECT * FROM sem_dedup('reports', 'mdr_report_key', 'dedup_block', 'narrative',
  'Do these two adverse event reports describe the same incident with the same device on the same occasion?', 0.5);

-- 4. Recall coverage: which reports describe a defect a recall already addresses, and which do not.
SELECT left_row.mdr_report_key, right_row.product_res_number, p
FROM sem_join('reports', 'recalls', 'product_code', 'narrative', 'recall_text',
  'Does this adverse event report describe the defect that this recall addresses?', 0.5);
-- and the calibrated size of the covered set: expected_count(p) over the unfiltered pairs (thr = 0).

-- 5. What a reviewer reads first.
SELECT mdr_report_key, brand_name, score, confidence
FROM sem_topk('reports', 'narrative', $severity_instructions, $severity_levels, 20);
```

`dedup_block` is `product_code || '/' || date_of_event`; `recall_text` is the recall's
`product_description`, `reason_for_recall` and `root_cause_description` as labeled lines
(the entity-matching rounds found labeled text lines beat a JSON object for pair states
by a point). Column names follow §2.2 and §2.6.

## 5. Deliverables and acceptance

- [ ] `bench/maude.py` with the commands of §3.1; `prepare` writes the raw pull, the
      parquet slices and `bench/maude_ids.json`; every command has `--dry-run`.
- [ ] Dev rounds R0 to R6 recorded in `docs/results/maude_runs.json`; test run with R0
      and the chosen round; `demo` recorded for at least one code.
- [ ] `docs/results/maude.md` generated from the run log: the rounds table, the held-out
      section per code and pooled (accuracy, ECE, reliability, deferral, coverage,
      confusions, cost, throughput), the demo section (trend table with a flagged month if
      one exists, dedup and join counts with usage, top-20), and a "Reading the numbers"
      section written by hand before the held-out run.
- [ ] README: a "MAUDE numbers" section with the headline (held-out `top-1 in set`, harm
      accuracy, ECE, $ per 1,000, and the whole-stream projection) and the one-sentence
      positioning against `ai()`; `bench/README.md` section; NEXT.md §1.5 (what
      shipped, with the numbers) and §3.6 marked done with what the rounds taught.
- [ ] `tests/test_maude_bench.py` green, `tests/test_results_docs.py` headline check,
      `uv run pytest -q`, `ruff check`, `ruff format --check` clean.
- [ ] Total live spend under $3, reported in the PR. No key value anywhere.

## 6. Rules that carry over

- Worktree per branch under `/mnt/data/vsletten/src/vsletten/duckjev/<branch>` (suggested
  `maude-bench`); `main/` is read-only; never change an existing worktree's branch.
- TypeSafe key: `source ~/.bash_secrets >/dev/null 2>&1; export TYPESAFE_API_KEY`. The
  file sets it without exporting it. `OPENFDA_API_KEY` the same way if one is added.
  Never print, log or commit either value; grep the tree and the diff before pushing.
- Tests never touch the network. Live runs are `bench/` scripts.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the PR body
  has `## Summary`, `## Test plan`, `## Breaking changes`, then the Claude Code line.
  After opening the PR, address every Sourcery comment (Sourcery's review budget may be
  exhausted until about 2026-09-27; if the check shows "skipped", wait and comment
  `@sourcery-ai review`), squash-merge when green with no unresolved thread, confirm the
  merged tree equals the reviewed head, fast-forward `main/`.
- The answer caches under `bench/data/` are the only way to `rescore`; do not delete the
  worktree without copying them (see `docs/NEXT.md` §5).
