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
