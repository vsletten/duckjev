# Benchmarks

Live benchmarks live here. They are scripts, not tests, and they need
`TYPESAFE_API_KEY` in the environment (on the workstation: `source ~/.bash_secrets`).

## Banking77

```bash
uv run python bench/banking77.py prepare                        # downloads, dev sample of train, no key
uv run python bench/banking77.py run R0 --split dev --limit 40  # pre-flight sample, not recorded
uv run python bench/banking77.py run R1 --split dev             # one tuning round on the dev split
uv run python bench/banking77.py run R3 --split test            # the held-out split
uv run python bench/banking77.py confusions R0 --split dev --show 5  # top confusions, offline
uv run python bench/banking77.py rescore                        # rebuild runs from their answer caches
uv run python bench/banking77.py report                         # writes docs/results/banking77.md
uv run python bench/banking77.py run R2 --split dev --dry-run   # fake transport, no key
```

The dev split is a stratified sample of `mteb/banking77` `train.jsonl`, 20 messages per
intent (1,540 rows), drawn by a fixed rule in `prepare`; the test split (3,080 rows) is
held out and runs only with the baseline and the round chosen on dev. Every round sends
one fused `jev(text, $questions)` request per message; the round in `ROUNDS` decides the
questions: flat (one 77-option Choice) or two-level (a division Choice plus one intent
Choice per division in the same request, consumed as the product distribution), the gloss
set (`banking77_criteria.json`, the structured `banking77_criteria_v2.json` overlay, or
`banking77_criteria_short.json`), option order, and whether the top-up Noul rides in the
same request. Glosses, examples and `banking77_divisions.json` were written from the label
names and the train split only.

Every full live run records its metrics in `docs/results/banking77_runs.json`, which is
committed; `report` renders the results doc from that file alone and picks the reported
round by a fixed rule: the selectable round with the best dev accuracy, ties to fewer tokens
per request. `--max-usd` (default $0.30) becomes the run's `max_input_tokens` budget, each
run starts from a fresh cache file so the timed pass pays for every row, and a full run
refuses to start until its round has a `--limit` pre-flight (which prints the projected
cost of the split). Pre-flights and `--dry-run` runs write only `_nN` / `_dry` suffixed
files under `bench/data/` and record nothing. `rescore` rebuilds each recorded run from
that run's own answer cache (`bench/data/cache_<split>_<round>.duckdb`) with a transport
that refuses every request. A test checks the README headline against the run log.

## SROIE receipts (`jev_extract`)

```bash
uv run python bench/sroie.py prepare                         # text columns of rth/sroie-2019-v2, no key
uv run python bench/sroie.py coverage                        # candidate coverage per field, offline
uv run python bench/sroie.py run R0 --split train --limit 40 # pre-flight sample, not recorded
uv run python bench/sroie.py run R3 --split train            # one round on the dev split
uv run python bench/sroie.py run R3 --split test             # the held-out split
uv run python bench/sroie.py rescore                         # rebuild runs from their answer caches
uv run python bench/sroie.py report                          # writes docs/results/sroie.md
```

The candidate builders are fixed SQL, the same in every round, so coverage is a
property of the builders only. They were designed on the train split before any
Jev call, and their test-split coverage was first computed after they were frozen.
Each round in `ROUNDS` changes one input: question wording, state layout, or
candidate order. Every full run records its metrics in
`docs/results/sroie_runs.json`, which is committed. `report` renders the results doc
from that file alone and picks the reported round by a fixed rule: the selectable
round with the best all-fields exact match on train. `--max-usd` (default $0.50)
becomes the run's `max_input_tokens` budget, and each run starts from a fresh cache
file so the timed pass pays for every receipt.

Per-row results (`bench/data/scored_<split>_<round>.parquet`) and the prepared data
are gitignored. `rescore` rebuilds each recorded run from that run's own answer
cache (`bench/data/cache_<split>_<round>.duckdb`) with a transport that refuses every
request, so it costs nothing and cannot read another run's data. `--dry-run` swaps
in a fake transport, writes only `_dry` files, and records nothing. A test checks
the README headline table against `docs/results/sroie_runs.json`.

## Entity matching (`sem_join`, `sem_dedup`, `sem_topk`)

```bash
uv run python bench/entity_matching.py prepare                                   # downloads, no key
uv run python bench/entity_matching.py coverage --record                         # blocking coverage, offline
uv run python bench/entity_matching.py run R0 --corpus abt --split dev --limit 40 # pre-flight, not recorded
uv run python bench/entity_matching.py run R2 --corpus abt --split dev           # one round on the dev split
uv run python bench/entity_matching.py run R2 --corpus dblp --split test         # the held-out split
uv run python bench/entity_matching.py demo --corpus abt                         # the macros live, on a sample
uv run python bench/entity_matching.py rescore                                   # rebuild runs from their caches
uv run python bench/entity_matching.py report                                    # writes docs/results/entity_matching.md
```

Two DeepMatcher / Magellan sets, Abt-Buy (textual products) and DBLP-ACM (structured
citations), each with two record tables and labeled candidate pairs split train / valid /
test by their authors. The valid split is the dev set; the test split is held out and runs
only with the baseline and the round chosen on dev (best dev F1 at the default threshold
0.5, ties to fewer tokens). Every labeled pair is one `jev_match` Noul; a round in
`ROUNDS` changes the question wording, the pair-state layout or the record order.
`coverage` measures what a key-equality block keeps without Jev; `demo` runs `sem_join`,
`sem_dedup` and `sem_topk` themselves on a sample of left rows (the dedup demo caps the
right rows per block, because every pair inside a block is judged). The same conventions
as the other benchmarks apply: a committed run log, `rescore` through a refusing
transport, suffixed files for pre-flights and dry runs, a full run that refuses to start
without a pre-flight, and a test that checks the README table against the run log.


## MAUDE (coded complaint surveillance)

Historical run sequence. The committed spend ledger leaves too little of its configured
budget to replay these paid commands; a new run needs its own ledger, pre-flights,
pre-test reading, and spend approval.

```bash
uv run python bench/maude.py prepare                              # openFDA pull and FDA vocabulary, no Jev key
uv run python bench/maude.py run R0 --split dev --limit 40        # pre-flight, spend recorded, rows not
uv run python bench/maude.py run R3 --split dev                   # one round on the dev split
uv run python bench/maude.py confusions R3 --split dev --code FTR --show 3  # top confusions, offline
uv run python bench/maude.py reading --file reading.md            # the hand-written reading, before test
uv run python bench/maude.py run R3 --split test                  # the held-out split
uv run python bench/maude.py demo --code QBJ                      # trend, sem_dedup, sem_join, sem_topk live
uv run python bench/maude.py rescore                              # rebuild runs from their answer caches
uv run python bench/maude.py report                               # writes docs/results/maude.md
```

`prepare` pulls every report of three FDA product codes over a closed `date_received`
window from openFDA `device/event` (999 a page, following the `search_after` cursor; QBJ
only on the 8th and 22nd of each month), keeps the reports whose Description of Event or
Problem has at least 100 characters and whose event type is filled, and writes the raw
pages, one parquet pool per code, the split slices and the recalls on file under
`bench/data/`. It draws dev (500), test (1,000) and a gloss slice (200) per code by a fixed
hash rule stratified by event type (a floor of 50 per type in dev and test where that many
exist), and commits the keys and each code's option set to `bench/maude_ids.json`; a later
`prepare` reuses those keys unless `--resample` and refuses a refreshed pull that omits
one. The per-code option sets were counted from the entire eligible pool, including test
labels, before the split; the test reports were held out from round selection but their
candidate lists are test-aware. It also reads sheet A of FDA's annexes
workbook with DuckDB's `read_xlsx` into `bench/maude_terms.json`, the vocabulary with its
definitions and hierarchy. `OPENFDA_API_KEY` is optional; the pull fits the keyless limit.

Every round sends one fused `jev()` request per report (R6 sends three): a Choice over the
code's option set plus a catch-all, a Choice over the event types and a severity Score. The
round in `ROUNDS` changes one input: the option descriptions (bare, official definitions,
or those plus `not_for` and examples where available from `bench/maude_criteria_v2.json`, whose
phrases and examples come from the gloss slice), the state (the description, or an object with the device names and
the manufacturer's narrative), the option order, the vocabulary (the code's set or every
term seen) and fusion. The chosen round is the selectable round with the best pooled dev
top-1 in set, ties to fewer tokens per request. The held-out split refuses to run before the
reading is recorded and for any round but R0 and the chosen one.

Every live call appends its spend to the run log, pre-flights and failed runs included,
and `--max-usd` (default $0.50) derives an input-token guard from the unspent part of the
configured $3.00 budget. Concurrent in-flight requests can overshoot that guard. The
other conventions are those of the benchmarks above: a committed run
log `docs/results/maude_runs.json`, a report rendered from it with the committed IDs and
criteria, a fresh answer cache
per run, `rescore` through a refusing transport, `_nN` / `_dry` files for pre-flights and
dry runs, a full run that refuses to start without a live pre-flight of its round, and a
test that checks the README table against the run log. The PR #8 answer caches,
raw pages, pools, and FDA annex workbook were archived outside Git at
`/mnt/data/vsletten/artifacts/duckjev/maude-pr8/data/` before worktree cleanup.
Restore that directory to `bench/data/` to run `rescore` without paid calls.
