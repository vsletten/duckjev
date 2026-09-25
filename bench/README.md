# Benchmarks

Live benchmarks live here. They are scripts, not tests, and they need
`TYPESAFE_API_KEY` in the environment (on the workstation: `source ~/.bash_secrets`).

## Banking77

```bash
uv run python bench/banking77.py sample   # 200-row reservoir sample; projects full-split cost
uv run python bench/banking77.py full     # full 3,080-row test split; writes docs/results/banking77.md
```

`full` refuses to start unless the last `sample` projected the full split at or
under `--max-usd` (default $0.50), and it also passes that amount to
`duckjev.register(max_input_tokens=...)` as a hard budget. Each phase starts from
a fresh cache file under `bench/data/` so the timed run pays for every row; the
cache re-run then reuses that file and must cost $0.

`--dry-run` runs the same pipeline against a fake transport (no key, no Jev calls)
and writes to `bench/data/banking77_dryrun.md` instead of the results doc.

The data (`mteb/banking77` `test.jsonl`, 3,080 rows) is downloaded once into
`bench/data/`, which is gitignored. The 77 option glosses in
`banking77_criteria.json` were written from the label names and train-split
examples only; the test split was never used to tune them.

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
