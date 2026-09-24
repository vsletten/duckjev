# duckjev — handoff for the implementing agent

Written 2026-09-23 by a Claude Fable 5.1 design session for Victor Sletten.
You are picking this up cold. Everything you need is in this file plus the
live docs it cites. Verified facts are marked **(verified)**; everything else
is a design decision or an instruction. Do not re-litigate decided items.

Move this file into the repo as `docs/HANDOFF.md` in the scaffold commit.

---

## 1. What we are building and why

**duckjev** is a Python package that installs *semantic operators* into DuckDB,
backed by TypeSafe's **Jev** System One model. Jev takes text `state` plus one
or more typed questions and returns **typed answers with calibrated
probabilities**: a `noul` (probability of yes), a `choice` (one option from a
closed set, with a full probability distribution and a confidence), or a
`score` (position on an ordered scale). It never generates text.

That output shape is why this works where LLM-backed `ai_classify`-style
functions are awkward: a Jev answer is a STRUCT with a schema known at
registration time, so it composes with the rest of SQL. Calibrated
probabilities also **aggregate**: `SUM(p)` over a table is an unbiased
estimate of how many rows have the property, with a variance. That is the
demo. A hard-label classifier gives you a count; a calibrated one gives you an
estimator with error bars, and DuckDB can express it in one query.

The public API is one call:

```python
import duckdb, duckjev
con = duckdb.connect()
duckjev.register(con)            # installs functions, macros, cache
```

after which SQL has `jev(...)`, `jev_noul(...)`, `jev_choice(...)`,
`jev_score(...)` and a set of macros (`sem_where`, `expected_count`, ...).

**Deliverable:** the package, its tests, and a benchmark on a public labeled
corpus (Banking77) with three numbers on the record: rows per second,
dollars per thousand rows, and a soft-vs-hard group-by comparison against
ground truth. Plus a short results doc.

**Time box:** one day. Tier one only (Python UDFs + macros). The community
extension (C++/Rust) and maintained judgment columns are documented in §10 as
follow-ons and are explicitly out of scope.

---

## 2. Ground truth (verified 2026-09-23 against live docs)

### 2.1 Jev HTTP API **(verified)**

Docs index: https://docs.typesafe.ai/llms.txt (append `.md` to any page path
for Markdown). Read https://docs.typesafe.ai/api.md and
https://docs.typesafe.ai/sdk/python.md before writing the client.

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json
```

Request:

```json
{
  "state": "string | object | array",
  "model": "jev-1.13.0",
  "questions": {
    "qid": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "qid2": {"type": "choice", "instructions": "...", "criteria": {"option_key": "description or null"}},
    "qid3": {"type": "score",  "instructions": "...", "criteria": ["level_0", "level_1", "level_2"]}
  }
}
```

`criteria` is optional for noul. Choice supports up to 255 options; score
accepts 2 to 10 levels. Option descriptions may be strings, objects
(`what` / `not_for` / `examples`), arrays, or null.

Response:

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "qid":  {"type": "noul",   "noul": 0.95},
    "qid2": {"type": "choice", "choice": "option_key", "probabilities": {"option_key": 0.88, "...": 0.12}, "confidence": 0.81},
    "qid3": {"type": "score",  "score": 1.43, "legend": {"0": "level_0", "1": "level_1", "2": "level_2"}, "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}, "confidence": 0.35}
  },
  "usage": {"input_tokens": 360, "output_tokens": 39}
}
```

- `noul` has no confidence field; the probability is the whole distribution.
- `choice.probabilities` sums to 1.0 over all options; `confidence` in [0,1] is
  derived from distribution concentration.
- `score` is the expectation over level indices (Σ index × p). `legend` maps
  index strings back to level descriptions.
- Status codes: 401 bad key, 422 validation, 429 rate limit, 529 overloaded.
  Retry 429/529 with exponential backoff.
- All questions in one request are evaluated independently against the same
  state; batching does not change answers. The cookbook measured 13 questions
  on one 54k-char doc: 12.2x cheaper and 10x faster than 13 separate calls.
  State is processed once per request. **Fusion is the cost lever.**

### 2.2 Model, limits, pricing **(verified)**

- Model id `jev-1.13.0`; aliases `jev-latest`, `jev-preview`. **Pin
  `jev-1.13.0`** and put the model id in every cache key.
- Context 64k tokens per request; 32k for state plus the longest question.
- Text input only (string, JSON object, or array of text).
- Pricing: **$42 per billion input tokens ($0.042 / M)**. Output tokens free.
- Rate limits: 250,000 tokens/sec, 1,200 requests/min, "adjusting
  dynamically". Design concurrency around ~20 req/s and expect 429s.
- Not trained on customer requests; enterprise ZDR available. English primary.
- Python SDK exists (`uv add typesafe-sdk`, `TypeSafeClient` /
  `AsyncTypeSafeClient`, `client.system_one(state=..., questions={...})`).
  **Decision: do not depend on it.** Use `httpx` directly against the contract
  above; fewer moving parts, and we need our own batching, backoff and usage
  accounting anyway.

### 2.3 DuckDB Python UDF facts **(verified against duckdb.org/docs/current/clients/python/function.html)**

- `con.create_function(name, fn, [param_types], return_type, type='arrow',
  side_effects=False, null_handling='default')`.
- `type='arrow'` gives vectorized UDFs: inputs arrive as PyArrow arrays (treat
  as `pa.Array` or `pa.ChunkedArray`; call `.combine_chunks()` defensively),
  return a `pa.Array` of the declared return type and the same length.
- DuckDB calls the UDF once per vector (~2048 rows). **Each call is a batch.**
- STRUCT / MAP / LIST return types are built from `duckdb.typing`
  (`duckdb.struct_type({...})`, `duckdb.map_type(k, v)`, `duckdb.list_type(t)`).
  Verify exact constructor names against the installed version.
- `side_effects=False` declares the function pure (same inputs, same output).
  We keep it False: a judgment is a pure function of (model, state, question).
- Default NULL handling is null-in null-out, which is what we want.

### 2.4 Local environment **(verified)**

- Workspace root `/mnt/data/vsletten/src/`. This repo lives at
  `/mnt/data/vsletten/src/vsletten/duckjev/`. The directory exists and is
  empty except for this file. **GitHub repo `vsletten/duckjev` does not exist
  yet.** `gh` is authenticated as `vsletten`.
- `uv 0.5.24`; CPython 3.13.1 installed under uv. Use Python 3.13.
- `TYPESAFE_API_KEY` is defined in `~/.bash_secrets` on this workstation
  (`source ~/.bash_secrets` before live runs). **Never copy its value into
  any file, test, log, commit, or PR.** It exists only on this workstation,
  so live runs are workstation-only.
- `duckdb`, `pyarrow`, `httpx` are not installed in any shared env; they are
  this project's own dependencies.

---

## 3. Repo conventions (Victor's rules, non-negotiable)

- Layout: `/mnt/data/vsletten/src/vsletten/duckjev/main` is the `main`
  worktree and is treated as read-only reference after the scaffold. All work
  happens in a branch-named worktree:
  `git -C .../duckjev/main worktree add .../duckjev/<branch> -b <branch>`.
- Bootstrapping exception: `git init` in `main/`, one scaffold commit on
  `main` (pyproject, README stub, this file at `docs/HANDOFF.md`, `.gitignore`,
  ruff config), `gh repo create vsletten/duckjev --private --source=main --push`.
  Then branch `spike-banking77` in its own worktree for everything else.
- One branch = one worktree = one PR. Open the PR into `main`, leave it open
  for Victor. This repo is not webhook-wired to the cloud PR watcher.
- Commit messages end with `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
  PR bodies: `## Summary`, `## Test plan`, `## Breaking changes`, then
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Tooling: `uv` for env, `ruff check` and `ruff format --check` clean,
  `pytest -q` green. **Tests never touch the network.** Live runs are scripts
  under `bench/`, not tests.
- Python ≥ 3.13, type hints throughout, pydantic not required (keep deps to
  `duckdb`, `pyarrow`, `httpx`; dev: `pytest`, `ruff`).

---

## 4. Package design (decided)

```
duckjev/
├── __init__.py       # register(), flush(), usage(), __version__
├── client.py         # JevClient: batching, concurrency, backoff, usage accounting
├── cache.py          # content-addressed answer cache
├── marshal.py        # answer dicts <-> Arrow struct/map arrays; question builders
├── functions.py      # UDF bodies + registration
└── macros.sql        # SQL macros installed by register()
bench/
├── banking77.py      # live benchmark, writes docs/results/banking77.md
└── README.md
tests/
docs/
├── HANDOFF.md        # this file
└── results/
```

### 4.1 `register(con, *, model="jev-1.13.0", concurrency=16, cache=True, cache_path=None, max_input_tokens=None, api_key=None, base_url=None)`

Installs everything below on `con`. `api_key` defaults to
`os.environ["TYPESAFE_API_KEY"]`; missing key raises at first call, not at
register (so tests and offline macro use work). `max_input_tokens` is a job
budget: the client raises `JevBudgetExceeded` once cumulative billed input
tokens pass it.

### 4.2 SQL functions

All text arguments are VARCHAR. Question specs are passed as **JSON strings**
(DuckDB JSON literals are VARCHAR-compatible), which keeps the UDF signatures
simple and lets users write criteria inline.

| Function | Signature | Returns |
|---|---|---|
| `jev(state, questions_json)` | VARCHAR, VARCHAR | VARCHAR (JSON of the `answers` map, verbatim from the API) |
| `jev_noul(state, instructions)` | VARCHAR, VARCHAR | DOUBLE |
| `jev_noul(state, instructions, criteria_json)` | VARCHAR, VARCHAR, VARCHAR | DOUBLE |
| `jev_choice(state, instructions, criteria_json)` | VARCHAR, VARCHAR, VARCHAR | `STRUCT(choice VARCHAR, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE))` |
| `jev_score(state, instructions, levels_json)` | VARCHAR, VARCHAR, VARCHAR | `STRUCT(score DOUBLE, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE), legend MAP(VARCHAR, VARCHAR))` |

`jev()` is the **fused primitive**: many questions over one state in one
request. The typed functions are sugar that build a one-question request. If
DuckDB overload resolution on Python UDFs is awkward, register
`jev_noul2` / `jev_noul3` internally and expose the arity variants via macros.

### 4.3 Vector execution model (the part that matters)

Inside each Arrow UDF call:

1. Combine chunks; build the list of `(state, question_spec)` pairs for the
   vector; NULL state → NULL result (null_handling default does this for you
   when *any* arg is NULL; make sure that is acceptable, else use `'special'`).
2. **Dedupe within the vector** on the cache key (§4.4). Many rows share a
   state (repeated product descriptions, boilerplate); pay once.
3. Consult the cache; collect misses.
4. Fan out misses concurrently, bounded by `concurrency`, with backoff on
   429/529 (base 0.5s, factor 2, jitter, max 6 attempts; then raise
   `JevTransportError` for the whole vector, do not partially fill).
5. Write hits back into the cache; marshal all answers to the declared Arrow
   type; return an array of the vector's original length and order.

Run the async fan-out on a **dedicated event loop in a helper thread**
(`asyncio.run` inside a UDF is fine unless the host process already has a
running loop, e.g. Jupyter; a thread with its own loop works in both cases).
Keep one `httpx.AsyncClient` per client instance with connection pooling.

### 4.4 Cache

Key: `sha256(canonical_json({"model": model, "state": state, "questions": questions}))`
where canonical JSON is `sort_keys=True, separators=(",", ":")`.
Value: the `answers` object plus `usage` and a timestamp.

Constraint: **do not write to the query's own DuckDB connection from inside a
UDF while a query is running.** Decision: the cache is a separate DuckDB file
(`~/.cache/duckjev/cache.duckdb` by default, `cache_path` to override,
`cache=False` to disable) opened on its own connection, guarded by a lock,
written by the UDF thread. Provide `duckjev.flush(con)` as a no-op-safe
explicit sync, and `duckjev.cache_table(con, name="jev_cache")` which imports
the cache into a table on the user's connection for SQL inspection.

Re-running any query is free after the first run. Bumping `model` invalidates
by construction because the model id is in the key.

### 4.5 Usage and cost

`duckjev.usage()` returns `{requests, input_tokens, output_tokens,
cache_hits, cache_misses, est_usd}` with `est_usd = input_tokens * 42 / 1e9`.
Reset with `duckjev.usage(reset=True)`. The benchmark reports from this.

### 4.6 Macros (installed from `macros.sql` by `register()`)

```sql
-- filter
CREATE OR REPLACE MACRO sem_where(state, q, thr) AS jev_noul(state, q) >= thr;

-- calibrated aggregation over a probability column p
CREATE OR REPLACE MACRO expected_count(p) AS SUM(p);
CREATE OR REPLACE MACRO expected_count_var(p) AS SUM(p * (1 - p));
CREATE OR REPLACE MACRO expected_count_stderr(p) AS sqrt(SUM(p * (1 - p)));

-- distribution helpers over a jev_choice struct c
CREATE OR REPLACE MACRO jev_argmax(c) AS c.choice;
CREATE OR REPLACE MACRO jev_p(c, k) AS coalesce(c.probabilities[k], 0.0);
CREATE OR REPLACE MACRO jev_runner_up(c) AS (
  SELECT key FROM (SELECT unnest(map_keys(c.probabilities)) AS key,
                          unnest(map_values(c.probabilities)) AS p)
  WHERE key <> c.choice ORDER BY p DESC LIMIT 1);
```

`SUM(p*(1-p))` is the variance of a sum of independent Bernoullis, which is
the tuple-independent assumption. Say so in the README.

The **soft group-by** needs no macro; it is native DuckDB and it is the demo:

```sql
SELECT e.key AS intent, SUM(e.value) AS expected_rows
FROM tickets t,
     UNNEST(map_entries(jev_choice(t.text, $instr, $criteria).probabilities)) AS e
GROUP BY intent ORDER BY expected_rows DESC;
```

(Check `map_entries` / `unnest` semantics on the installed DuckDB; the intent
is to explode each row's distribution into (key, p) rows and sum p per key.)

Optional, only if time remains: a table macro `sem_join(a, b, block_key, q, thr)`
that blocks on key equality then filters on `jev_noul(struct_pack(...))`.

---

## 5. Benchmark: Banking77 (decided)

Why this corpus: public, short customer-support utterances, **77 gold intent
labels** (fits Choice's 255-option cap), so we can measure accuracy *and*
calibration, and compare soft expected counts per intent against true counts.
That comparison is the point of the whole exercise.

Source: the `mteb/banking77` parquet on Hugging Face has `text`, `label`,
`label_text` columns (verify; fall back to `PolyAI/banking77` via its parquet
conversion branch or the `datasets` library if needed). DuckDB can read
`hf://datasets/<org>/<name>/<path>.parquet` with the `httpfs` extension, or
download once with `httpx` to `bench/data/` (gitignored).

Criteria: build the 77-option `criteria_json` from the distinct `label_text`
values, key = label_text, description = a short human gloss (write them once
into `bench/banking77_criteria.json`; the label names are mostly
self-describing, e.g. `card_arrival`, `lost_or_stolen_card`; a one-line gloss
per label is enough and is worth the 20 minutes). Add no `none` option; the
corpus is closed-world.

Instructions string: `"Which banking-support intent does this customer message express?"`

Procedure (in `bench/banking77.py`):

1. Load the **test split (3,080 rows)**. First run on a 200-row sample to
   validate the pipeline and check cost (expect well under $0.05), then the
   full split. Full split cost estimate: ~800 tokens/request × 3,080 ≈ 2.5M
   tokens ≈ **$0.10**. Wall time bounded by 1,200 req/min ≈ 3 min.
2. Materialize `CREATE TABLE judged AS SELECT text, label_text,
   jev_choice(text, $instr, $criteria) AS j FROM banking77_test`.
   Time it; record `duckjev.usage()`.
3. Report, all from SQL over `judged`:
   - **Throughput**: rows/s, requests/s, tokens/s.
   - **Cost**: est_usd and usd per 1,000 rows.
   - **Accuracy**: `AVG(j.choice = label_text)`.
   - **Calibration**: expected calibration error over `j.confidence` in 10
     bins, and a reliability table (bin, mean confidence, accuracy, n).
   - **Soft vs hard group-by**: per intent, `true_count`, `hard_count`
     (argmax), `expected_count` (Σp), `stderr` (√Σp(1−p)); plus totals of
     |hard − true| and |expected − true| across intents. Include a column
     `within_2se = abs(expected - true) <= 2*stderr`.
   - **Cache re-run**: run step 2 again and show wall time and $0.
4. Write `docs/results/banking77.md` with the tables and the exact query
   text, and paste the headline numbers into the README.

Also run one `sem_where` example for the README, e.g. Noul "Is this message
about a physical card?" with `expected_count` and `expected_count_stderr`
compared against the true count of card-related intents.

---

## 6. Tests (no network)

Provide a `FakeTransport` (an `httpx.MockTransport` or a stub client method)
that returns canned answers keyed by question type, records requests, and can
inject 429/529 sequences. Cover:

- `marshal`: each answer type → Arrow array of the declared type; MAP
  construction; NULL passthrough; legend mapping for score.
- `client`: one request per unique (state, questions) within a batch; backoff
  retries then raises; usage accounting; budget exceeded raises.
- `cache`: hit/miss; key includes model; disabled cache path; persistence
  across two client instances on the same cache file (tmp path).
- `functions`: end-to-end `register()` on an in-memory DuckDB with the fake
  transport, run each SQL function and each macro, check shapes and values;
  soft group-by query returns expected sums; `jev_runner_up` correct.
- `register()` works with no API key present; the first live call without a
  key raises a clear error.

---

## 7. Deliverables and acceptance

- [ ] `vsletten/duckjev` private repo; scaffold on `main`; work on
      `spike-banking77` worktree; PR open into `main`.
- [ ] `uv run pytest -q` green, `uv run ruff check .` and
      `uv run ruff format --check .` clean.
- [ ] `duckjev.register(con)` installs all functions in §4.2 and macros in §4.6.
- [ ] Benchmark ran live on the full Banking77 test split on this
      workstation. `docs/results/banking77.md` contains: rows/s, $/1k rows,
      accuracy, ECE and reliability table, the soft-vs-hard per-intent table
      with totals, and the cache re-run timing.
- [ ] README: what it is (two paragraphs, use §1), install, the three SQL
      examples (choice column, soft group-by, `sem_where` +
      `expected_count_stderr`), the headline numbers, and a "what is
      calibrated aggregation" paragraph with the Bernoulli-variance note.
- [ ] No secret value anywhere in the tree or PR.

---

## 8. Suggested subagent split

- **A. client + cache + marshal** with the fake transport and their tests.
  Pure Python, no DuckDB.
- **B. functions + macros + register()** against A's interfaces (agree the
  `JevClient.judge(batch) -> list[answers]` signature first), with the
  DuckDB end-to-end tests.
- **C. bench** script and the criteria gloss file, developed against the
  fake transport, then run live by the integrator.
- **Integrator**: scaffold, worktree, wire A/B/C, live sample run, full run,
  results doc, README, PR.

A and C can start immediately; B needs A's interface only.

---

## 9. Known gotchas

- Arrow UDF inputs may be `ChunkedArray`; normalize before indexing.
- DuckDB may evaluate a UDF on constant-folded arguments at bind time; with a
  missing API key that would raise at bind, so guard: if all states in the
  vector are NULL or empty, return NULLs without touching the network.
- Build MAP arrays with `pa.MapArray.from_arrays(offsets, keys, items)`; the
  offsets must be int32 and monotone; empty maps are fine.
- `probabilities` keys for choice are option keys; for score they are level
  index strings. Keep both as VARCHAR keys.
- The vector may contain many rows with the same state but the UDF is called
  per function, not per column set. Cross-column fusion is only available via
  `jev()`; document that plainly.
- Rate limits are dynamic; log 429 counts in `usage()` so the benchmark
  reports them.
- `httpx` timeouts: 30s per request is plenty; Jev is sub-second.

---

## 10. Out of scope, recorded for the future

- **Tier two**: a DuckDB community extension (C++ or Rust template) with the
  HTTP client inside the extension so the functions are `INSTALL jev FROM
  community` and work from CLI/Node/Go/WASM; cost annotations so the planner
  orders semantic filters last; a `jev_explain(query)` table function that
  reports estimated token spend before execution.
- **Tier three**: maintained judgment columns (sidecar table + view +
  `jev_refresh(table, column)` judging only new state hashes under the pinned
  model), and a view-matching rewrite so a repeated question over a stored
  column becomes a column read.
- `jev_extract(state, candidates_json)`: code proposes candidate spans via
  `regexp_extract_all`, Jev selects per field with a `none` option, returns a
  struct of (value, p) per field. This is the large-scale extraction pattern
  (select, don't generate) and the strongest follow-on.
- `sem_join` / `sem_dedup` over blocked pairs; `sem_topk` via `jev_score`.
- Theory note for the README's "further reading": semantic operators (LOTUS,
  Stanford), provenance semirings (Green, Karvounarakis, Tannen 2007) and
  probabilistic databases (Dalvi & Suciu; MystiQ/Trio/MayBMS). Calibrated
  Jev answers are the missing input those systems never had.
