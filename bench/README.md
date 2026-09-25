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
