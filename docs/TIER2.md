# duckjev tier two: the native extension, built beside tier one

Written 2026-09-25 by the Claude Fable 5.1 session that wrapped up tier one. This is the
cold-start handoff for the coding session that builds tier two. Read `docs/HANDOFF.md` §1
and §2 first (what duckjev is, the verified Jev API contract) and `docs/NEXT.md` §1 and §2
(what shipped, what the numbers mean). This file says what tier two is, what it must
reproduce from tier one byte for byte, which DuckDB facts it rests on, and the milestones
in order with the gate each one has to pass.

## 0. What tier two is

Tier one is a Python package: Arrow UDFs plus SQL macros, registered per connection by
`duckjev.register(con)`, with the HTTP client, the answer cache and the usage counters in
the Python process. It works only where Python is the host.

Tier two is the same operators as a native DuckDB extension. The HTTP client, the cache,
the usage accounting and the budget cap move into compiled code that DuckDB loads, so the
functions work in every DuckDB host: the CLI, Python, Node, Go, Java, R, and later WASM.
The end state is `INSTALL duckjev FROM community; LOAD duckjev;`. The first milestone is
`LOAD 'build/release/extension/duckjev/duckjev.duckdb_extension'` on this machine.

Two things only a native extension can do come with it:

- the planner stops treating a Jev call as a cheap predicate: semantic filters run last,
  after the cheap predicates have cut the rows, and are not duplicated by filter pushdown;
- `jev_explain(query)` reports, before anything runs, how many requests a query will make
  and roughly what they cost.

Tier two is a side-by-side rebuild, not a replacement. The Python package stays as it is:
it is the reference implementation, the benchmark harness runs on it, and its answer
caches are the offline oracle the extension is tested against (§3). Nothing tier one
returns changes in this tier. The extension lives under `extension/` in this repo, with
its own build, its own tests and its own CI; the Python package gains one optional
switch, `register(con, backend="extension")`, which loads the extension instead of
installing Python UDFs, and the bench harness gains the matching `--backend` flag so the
same runs can be replayed through either.

## 1. Decisions

These are made. Revisit only with a reason written next to the change.

1. **Language: C++ on `duckdb/extension-template`.** The Rust template sits on the C API,
   which registers scalar and table functions but has no optimizer hook, no macro
   registration, and no access to the planner's cardinality estimates. Both planner
   features of this tier (§0) need the C++ API. See §2.2.
2. **Extension name `duckjev`, function names unchanged.** `jev` is TypeSafe's product
   name and the community registry is public; the extension carries the project's name and
   the functions keep theirs (`jev_noul`, `sem_join`, and so on). Renaming to `jev` later is
   a one-line change in `description.yml` and the CMake target.
3. **Layout: `extension/` in this repo**, one directory, self-contained (the template's
   `Makefile`, `CMakeLists.txt`, `extension_config.cmake`, `src/`, `test/sql/`, and the two
   submodules `duckdb` and `extension-ci-tools` under it). The registry's build workflow
   has no subdirectory input and needs a public GitHub repository (§2.2), so submission
   uses a subtree split of `extension/` into a public `vsletten/duckjev-extension` (M6).
   Until then nothing about the layout leaks out, and this repo stays private.
4. **HTTP through DuckDB's own `HTTPUtil` with httpfs loaded**, not a bundled TLS client.
   Core's util does GET only; httpfs replaces it with a curl-based one that does POST
   over TLS, and duckdb-wasm replaces it with the browser's fetch (§2.3, §2.2). So one
   code path covers native and WASM, there is no vcpkg port, and the platform list is
   httpfs's list. The extension autoloads httpfs at the first live call. Every precedent
   in §2.4 bundles OpenSSL or curl instead and excludes platforms for it; that route
   (`duckdb_httplib_openssl` plus vcpkg `openssl`, as `open_prompt` does) is the named
   fallback if M1 finds the httpfs POST path wanting, and switching is one file.
5. **Same wire format, same cache key, same SQL surface as tier one.** §3 is the
   contract. Where tier one's behavior is a wart (the `jev_noul2`/`jev_noul3` split), the
   extension fixes it natively and the macro overload is dropped; where it is a decision
   (order-preserving key, NULL rule, no partial fill), the extension reproduces it.
6. **The answer cache is a DuckDB file the extension opens on its own,** not a table in the
   user's database. Same schema as tier one, so the Banking77 and entity-matching caches
   written by tier one are read by tier two unchanged. See §3.4.
7. **The API key never becomes a setting.** DuckDB settings print in `duckdb_settings()`.
   The key comes from `CREATE SECRET` (type `duckjev`) or `$TYPESAFE_API_KEY`, in that
   order, and is read at the first call that needs the network. Never printed, logged,
   returned by any function, or written to the cache file.
8. **Tests never touch the network.** The extension gets an offline switch
   (`SET duckjev_offline = true`) under which a cache miss is an error, plus a replay
   transport for the sqllogictests that answers from a fixture file with no socket at all.
   The only sockets any test opens are loopback, in the pytest suite that exercises the
   HTTP layer against a local fixture server, and that suite says so at the top.
9. **Live spend in this tier: at most $1 total,** and only for the wire-format smoke in
   milestone 2 and the final end-to-end demo. Everything else is offline against the
   recorded caches. The 40-row pre-flight rule and `duckjev_max_input_tokens` cap apply to
   any live call, as in tier one.
10. **CI builds Linux amd64 on every PR and the full platform matrix only on a tag or a
    manual dispatch.** This repo has no CI today (Sourcery only). The full matrix is five
    to eight heavy jobs per run and this is a private repo on metered minutes.

## 2. Ground truth (verified 2026-09-25)

Facts the plan rests on, with where they were checked. Anything marked "could not verify"
is a question for the first milestone, not an assumption to build on.

### 2.1 Local environment

- DuckDB Python 1.5.5 on `linux_amd64` (checked with `pragma_platform()`); installed
  extensions `core_functions`, `icu`, `json`, `parquet` loaded, `httpfs` installed but not
  loaded. No DuckDB CLI on the path. Node v25.2.1 on the path; no Go.
- Toolchain: cmake 4.4.3 (snap), GNU make, g++ 13.3 (Ubuntu 24.04). No ninja, no clang++,
  no vcpkg, no git-lfs. 32 cores, 61 GB RAM, 1.4 TB free on `/mnt/data`. A release build
  of DuckDB from the submodule is a 20 to 40 minute job here the first time; `ccache` is
  worth installing before milestone 0.
- The repo: `vsletten/duckjev`, private, `main` protected, no `.github/workflows`. Sourcery
  reviews PRs. Worktrees live beside `main/` under `/mnt/data/vsletten/src/vsletten/duckjev/`.

### 2.2 DuckDB extension tooling

Read from `duckdb/extension-template`, `duckdb/extension-template-rs`,
`duckdb/extension-ci-tools` (tag `v1.5-variegata`), `duckdb/community-extensions` and
duckdb.org on 2026-09-25.

**Releases.** Latest stable is v1.5.5 (2026-07-22), the version the local Python package
runs. The release calendar lists 1.5.6 for 2026-09-28 and 2.0.1 for 2026-11-16. Every
other minor is long-term support; 1.4.x is the LTS line, 1.5.x is not. The registry builds
every extension against the latest stable and, when a descriptor gives a `repo.andium`
ref, also against 1.4.5. This plan pins v1.5.5 and does not claim the LTS line.

**C++ template.** `Makefile` is three lines that include
`extension-ci-tools/makefiles/duckdb_extension.Makefile`; targets are `release` (the
default), `debug`, `test` (sqllogictests under `test/sql/`, release build), `test_debug`,
`format-check`, `format-fix`, `tidy-check`, `wasm_mvp`, `wasm_eh`, `wasm_threads`. The
build compiles DuckDB itself from the `duckdb` submodule; the docs recommend `GEN=ninja
make` with ccache. `scripts/bootstrap-template.py <name>` renames the demo extension
(`waddle`). `vcpkg.json` lists `openssl` "for demo purposes" and vcpkg is needed only
when that file has dependencies, so removing the dependency removes vcpkg from the
toolchain. Loading a local build: `LOAD '/path/x.duckdb_extension'` under
`allow_unsigned_extensions` (CLI `duckdb -unsigned`; Python
`duckdb.connect(config={"allow_unsigned_extensions": "true"})`; Node the same option).
CI: `MainDistributionPipeline.yml` calls
`duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml@v1.5-variegata`
(the template still pins `duckdb_version: v1.5.4`; use `v1.5.5`) plus
`_extension_code_quality.yml` for format and tidy. Default platform matrix: linux_amd64,
linux_arm64, osx_amd64, osx_arm64, windows_amd64, windows_amd64_mingw, wasm_mvp, wasm_eh,
wasm_threads; opt-in: the musl variants and windows_arm64; `exclude_archs` is a
semicolon list. `docs/UPDATING.md` says the C++ API "is not guaranteed to be stable" and
that each DuckDB release means bumping both submodules and the workflow refs.

**Rust template.** Experimental, on the C extension API through `duckdb-rs` with
`loadable-extension`; builds with `USE_UNSTABLE_C_API=1`, so a binary works only on the
exact DuckDB version it was built for. Registers scalar (`VScalar`) and table (`VTab`)
functions; no aggregate wrapper in the crate; the C API has no macro registration and no
optimizer, planner or parser hook at all (the v1.5.5 `duckdb.h` mentions neither). That
rules it out for §0's planner features.

**Registry.** Submission is one pull request adding `extensions/<name>/description.yml`;
the directory name must equal `extension.name` (lowercase letters, digits, `-`, `_`);
fields used by the build script are `repo.github` (a public GitHub repository),
`repo.ref`, `repo.ref_next`, `repo.andium`, `extension.excluded_platforms`,
`opt_in_platforms`, `requires_toolchains`, `custom_toolchain_script`, `vcpkg_url`,
`vcpkg_commit`, `test_config`, plus `version`, `language`, `build`, `license`,
`maintainers`, `description` and the docs fields. The reusable workflow checks the
repository out at the workspace root and runs `make configure_ci`, `make <build_type>`,
`make test_<build_type>` there; it has no path or working-directory input, so the
extension's Makefile must be at the root of the repository the registry builds. Names must
be unique and the Foundation may refuse or rename on trademark or core-extension clashes.
Binaries are signed; `INSTALL <name> FROM community; LOAD <name>;` works in every client
because `community` is an engine constant (`COMMUNITY_REPOSITORY_URL`); Python also has
`con.install_extension(name, repository="community")`. All extensions are rebuilt when a
new DuckDB ships and whenever their descriptor changes. Census of the 351 descriptors:
275 C++, 51 Rust; 172 exclude all three WASM targets.

**WASM.** DuckDB-Wasm loads community extensions (`LOAD` fetches the `.wasm` build from
the registry and `dlopen`s it). There are no sockets; HTTP goes through
`config.http_util`, which duckdb-wasm sets to its own `HTTPWasmUtil` (a synchronous
`XMLHttpRequest`), forced to HTTPS and bound by the browser's CORS rules. httpfs's curl
client is compiled out under Emscripten. Whether `HTTPWasmUtil` implements `Post` could
not be verified from the docs; it is the first thing the M6 spike checks, together with
whether `api.typesafe.ai` answers a browser preflight with permissive CORS headers, which
no extension can work around.

### 2.3 Optimizer hooks, HTTP, secrets, macros from C++

Read from the `v1.5.5` tag of `duckdb/duckdb` (the release the local Python package runs)
and from `duckdb/duckdb-httpfs` main. Paths are under `src/` unless said otherwise.

**Filter ordering.** `optimizer/expression_heuristics.cpp` reorders the conjuncts of every
`LOGICAL_FILTER` by a cost heuristic, ascending. A function's cost is looked up by name in
a fixed table (`+` 5, `*` 10, `round` 100, `~~` and `regexp_matches` 200); a function not
in the table costs `1000 + children`. A comparison costs `left + 5 + right`, a column
reference 8 to 40. So a Jev predicate already sorts after every cheap predicate. There is
no way for a function to declare its own cost: `BaseScalarFunction` carries only
`stability`, `null_handling` and `errors`. Two catches: the pass returns without
reordering at all if any conjunct `CanThrow()`, that is, if any function in the filter is
registered with `FunctionErrors::CAN_THROW_RUNTIME_ERROR`, which is the default; and the
pass (`OptimizerType::REORDER_FILTER`) runs second to last in the built-in pipeline, after
`FILTER_PUSHDOWN` and `JOIN_ORDER`.

**Duplication by pushdown.** `optimizer/pushdown/pushdown_projection.cpp` pushes a filter
on a projected alias below the projection by copying the projection's expression into the
filter. That is the tier-one double judge (`SELECT jev_match(...) AS p ... WHERE p >= thr`
evaluates the call in the filter and again in the projection). The copy is skipped, and
the filter stays above the projection, only when the projected expression `IsVolatile()`
or the filter `CanThrow()`. `IsVolatile()` is true only for
`FunctionStability::VOLATILE`; `CONSISTENT_WITHIN_QUERY` changes nothing in the optimizer
(it only disables the executor's per-dictionary result cache). VOLATILE also switches off
constant folding, common-subexpression elimination, pushdown into scans, late
materialization and CTE inlining for that expression.

**Optimizer extension.** `optimizer/optimizer_extension.hpp`: an `OptimizerExtension` has
`pre_optimize_function` (runs on the bound plan before every built-in pass) and
`optimize_function` (runs after all of them, so after `REORDER_FILTER`), both
`void(OptimizerExtensionInput &, unique_ptr<LogicalOperator> &)`. Register with
`OptimizerExtension::Register(DBConfig::GetConfig(db), ext)`; the old
`config.optimizer_extensions` vector no longer exists. There is no hook between built-in
passes.

**Planning a query string inside a function.** `ClientContext::ExtractPlan(query)` returns
the optimized plan but takes the context lock, which the running query already holds, so
calling it from a bind function deadlocks. The in-tree precedent for `jev_explain` is
`extension/json/json_functions/json_serialize_plan.cpp`: construct `Planner
planner(context); planner.CreatePlan(statement)` and `Optimizer optimizer(*planner.binder,
context)` directly. `LogicalOperator::estimated_cardinality` is set by the join-order
optimizer on scans and joins; other operators fall back to the maximum of their children.

**Extension API.** `ExtensionUtil` is gone in 1.5 (its header is a `static_assert`
pointing at PR #17772). Extensions implement `Load(ExtensionLoader &loader)` and register
through the loader: `loader.RegisterFunction(ScalarFunctionSet)`,
`loader.RegisterFunction(CreateMacroInfo &)`, `loader.RegisterSecretType(...)`,
`loader.RegisterFunction(CreateSecretFunction)`.

**HTTP.** Core bundles cpp-httplib 0.27 (`third_party/httplib/httplib.hpp`) but without
OpenSSL, and core's `HTTPLibClient` implements only `Get`; `Post` throws
`NotImplementedException`. So core alone can do neither HTTPS nor POST. The abstraction
is `common/http_util.hpp` (`HTTPUtil::Get(db)`, `HTTPParams`, `PostRequestInfo` with an
input buffer and an output buffer, `HTTPClient::Post`), and httpfs replaces it at load
with a curl-based implementation that does both (`duckdb-httpfs/src/http/http_settings.cpp`
calls `config.SetHTTPUtil(make_shared_ptr<HTTPFSCurlUtil>(...))`). Conclusion: an HTTPS
JSON POST needs httpfs loaded, or the extension linking OpenSSL or curl itself through
vcpkg.

**Secrets.** `main/secret/secret_manager.hpp`: register a `SecretType {name, deserializer,
default_provider}` and a `CreateSecretFunction {secret_type, provider, function,
named_parameters}`; the create function returns a `KeyValueSecret` with `secret_map` and
`redact_keys`. Worked example: httpfs's bearer secret in
`duckdb-httpfs/src/create_secret_functions.cpp`. Read at call time with
`KeyValueSecretReader(context, "duckjev", path).GetSecretKey("api_key")`.
`duckdb_secrets()` prints `redacted` for keys in `redact_keys`; the unredacted form is
refused unless `allow_unredacted_secrets` is set.

**Macros from C++.** Scalar macros: a `DefaultMacro {schema, name, parameters[8],
named_parameters[8], macro}` table, `DefaultFunctionGenerator::CreateInternalMacroInfo`,
then `loader.RegisterFunction(*info)`; live example `extension/json/json_extension.cpp`.
Table macros: core has `DefaultTableMacro` and `CreateTableMacroInfo`, but the helper is
private and no in-tree extension registers a table macro; the extension replicates the
ten lines (a `CreateMacroInfo(CatalogType::TABLE_MACRO_ENTRY)` holding a
`TableMacroFunction(query_node)` parsed from the SELECT, `internal = true`). Parameters
are capped at eight per macro; `sem_join` has seven.

What this settles for the design:

1. Register every `jev_*` function `FunctionStability::VOLATILE`. It is the one switch
   that stops the pushdown copy at its source, and the client-level dedupe (§3.6) makes
   the lost common-subexpression elimination cost CPU, not requests. The `sem_join` and
   `sem_dups` window barrier becomes unnecessary (milestone 3 proves it).
2. Do not rely on `REORDER_FILTER` for the ordering guarantee, since one throwing
   function in the conjunction disables it and the functions do throw on transport
   failure. The extension's own `optimize_function` walks every filter and every
   conjunction and stable-partitions the conjuncts so those containing a `jev_*` call
   come last. Fifty lines, independent of the `errors` flag.
3. The same `optimize_function` pulls a filter whose conjuncts all contain `jev_*` calls
   up above an inner join when every column it references comes from one side, and
   leaves it where it is otherwise (outer joins, both sides, projections that drop the
   columns). This is the "judge after the join has cut the rows" behavior of §5.1, done as
   a rewrite of the final plan because there is no hook before `FILTER_PUSHDOWN`.
4. `jev_explain` plans through `Planner` and `Optimizer` directly, never `ExtractPlan`.
5. HTTPS POST goes through `HTTPUtil::Get(db)` and a `PostRequestInfo` with httpfs
   loaded: at the first call that needs the network the extension autoloads httpfs
   (`ExtensionHelper::AutoLoadExtension`, which honors `autoload_known_extensions`) and,
   if the util still cannot POST, raises `JevTransportError: HTTPS needs the httpfs
   extension (INSTALL httpfs; LOAD httpfs)`. Retries stay in the extension
   (`HTTPParams.retries = 0`) so the tier-one backoff and 429/529 accounting are
   reproduced exactly rather than delegated to httpfs's retry loop. Under WASM the util
   is the browser's and httpfs is not involved.
6. Leave `FunctionErrors` at its default. The functions do throw (transport, budget),
   the flag is honest, and the ordering guarantee comes from item 2, not from
   `REORDER_FILTER`.

### 2.4 Precedents: extensions that call an LLM from SQL

Read from the repositories and registry descriptors on 2026-09-25.

| Extension | HTTP | Platforms excluded | Secrets | Batching and cache | Offline tests |
|---|---|---|---|---|---|
| `flock` (was FlockMTL, dais-polymtl, C++) | libcurl via vcpkg, `curl_multi` for parallel batches; a sync-XHR path under Emscripten that is declared but not deployed | windows_amd64_rtools | custom types `openai`, `azure_llm`, `ollama`, `anthropic` | `max_batch_size` rows per prompt, async batches, requests/minute and token quotas; no cache | gtest with a mock provider; no sqllogictests; live pytest suite apart |
| `open_prompt` (quackscience, C++) | bundled `duckdb_httplib_openssl` + vcpkg openssl | windows_amd64_rtools | `SET VARIABLE` → env → secret `open_prompt` | one request per row, no cache | the one SQL test creates a secret and never calls the API |
| `http_client` (query-farm, C++) | `duckdb_httplib_openssl` + vcpkg openssl | windows_amd64_mingw | none | none | SQL tests hit live hosts |
| `http_request` (C++) | `duckdb_httplib_openssl` | windows_amd64_mingw | none | per-chunk thread pool capped at 32; 1 s TTL cache keyed on method, url, headers, body | |
| `llm` (C++) | runs SQL against `http_request` on a fresh `Connection` "to avoid deadlock" | all WASM, mingw | type `llm` | `llm_and_cache` | |
| `ai` (leonardovida, C++) | libcurl via vcpkg | all WASM | env or type `duckdb_ai` | in-memory LRU (1,024 entries), token bucket, egress allowlist | SQL test makes no network call |
| `gsheets` (C++) | `duckdb_httplib_openssl`; a `HTTPUtil` client stub that throws "Waiting for HTTPUtil POST support" | all WASM, rtools, mingw | type `gsheet` | | unit tests with a mock client; SQL tests need a token |
| `quackformers` (Rust, cargo) | `ureq` with rustls | all WASM, mingw, musl | none | local models, no API | |

What the precedents settle:

- Nobody calls an LLM through `HTTPUtil`; everyone links OpenSSL or curl through vcpkg
  and pays for it in excluded platforms (mingw always, WASM usually). The one attempt at
  HTTP from WASM (flock's synchronous XHR) is not deployed. Rusty Conover's WASM test of
  124 extensions passed 58 and recommends `HTTPUtil` precisely because duckdb-wasm wires
  it to the browser's HTTP stack. `HTTPUtil` with httpfs loaded is therefore the
  untraveled but right path, and §1.4 keeps the bundled-OpenSSL route as the named
  fallback if M1 finds `HTTPUtil::Post` through httpfs wanting.
- Nobody has a persistent content-addressed cache or in-vector deduplication; `ai` has an
  LRU and `http_request` a one-second TTL. The tier-one cache design carries over
  unchanged and is the reason the differential test in §4 costs nothing.
- Running SQL against the user's database from inside a function is a documented hazard:
  flock wraps its `ATTACH` in a retry-and-sleep guard, `llm` comments "separate
  connection to avoid deadlock", ducklake has an open "Resource deadlock avoided" issue
  from a `Connection` opened in a table function. §1.6 keeps the cache on a private
  `DuckDB` instance opened on the cache file, never a connection to the user's database
  and never an `ATTACH` into it.
- Custom secret types are the norm (`flock`, `open_prompt`, `llm`, `ai`, `gsheets`), with
  `CREATE SECRET (TYPE <type>, API_KEY '...')` and a config provider. Same here.

## 3. The parity contract: what tier two reproduces from tier one

Everything in this section is checked by the differential test in §4, not by reading.

### 3.1 Wire format

One request per unique `(state, questions)` pair, `POST {base_url}/v1/systemone`, header
`Authorization: Bearer <key>`, body exactly:

```json
{"state": <state>, "model": "jev-1.13.0", "questions": {<qid>: <question>, ...}}
```

Key order in the body is `state`, `model`, `questions`; the questions map keeps the order
the caller built; each question keeps `type`, `instructions`, `criteria` in that order. The
state is sent as the VARCHAR the SQL passed, parsed as JSON when it is valid JSON (a
`jev_pair` state is an object, not a string containing an object), else as a string.
Compact JSON, no spaces, UTF-8 not escaped (`ensure_ascii=False`). This is the same string
the cache key hashes.

Question builders (tier one `duckjev/marshal.py`, reproduce exactly):

- `jev_noul(state, instructions)` → `{"type":"noul","instructions":i}`; the three-argument
  form adds `"criteria"` parsed from the JSON argument, which must be an object.
- `jev_choice(state, instructions, criteria_json)` → `criteria` as given when it is an
  object; a JSON array of options becomes `{option: null, ...}` in array order; empty or
  more than 255 options is an error before any request.
- `jev_score(state, instructions, levels_json)` → `criteria` is the array of 2 to 10
  level strings, in order.
- `jev(state, questions_json)` → the map as given, after checking every entry has a type
  in `noul`, `choice`, `score`.
- `jev_extract(state, spec_json)` → one Choice per field that has candidates, options are
  the candidate strings themselves (stripped, deduplicated, empties dropped, order kept)
  plus a `none` option (key `none`, or `none of these` when a candidate is literally
  `none`; description `None of these candidates is the requested value.` unless the spec
  gives one; `"none": false` omits it); more than 255 options counting `none` is an error
  naming the field.

Question id for every one-question function is `q`.

Response handling: `answers` must be an object whose key set equals the request's question
ids, else an API error. `usage.input_tokens` and `usage.output_tokens` are integers,
default 0.

Status handling: 200 accept; 401 and 403 raise an auth error and are not retried; 429,
500, 502, 503, 504, 529 and transport failures retry with exponential backoff (base 0.5 s,
factor 2, jitter uniform in [0.5×, 1.5×), `Retry-After` honored as a floor, six attempts)
then raise a transport error; any other status raises an API error with the first 500
bytes of the body. A failed request fails the whole vector: no partial fill, but every
answer that did arrive is written to the cache first so the retry does not pay for it
again.

### 3.2 Cache key

```
key = sha256( compact_json({"model": model, "state": state, "questions": questions}) )
```

hex-encoded, over the same compact JSON as §3.1 with the keys in that order and no
sorting anywhere. Test vector to pin in both implementations (tier one produces it today;
the extension's first test asserts the same string):

```sql
SELECT jev_cache_key('jev-1.13.0', '"hello"', '{"q":{"type":"noul","instructions":"Is it a greeting?"}}');
```

The extension exposes `jev_cache_key(model, state_json, questions_json)` as a scalar
function so the two implementations can be compared row by row from SQL.

### 3.3 SQL surface

Functions, all VARCHAR arguments, same return types as tier one (`duckjev/functions.py`):

| Function | Returns |
|---|---|
| `jev(state, questions_json)` | VARCHAR: the `answers` object as compact JSON, verbatim |
| `jev_noul(state, instructions)` and `jev_noul(state, instructions, criteria_json)` | DOUBLE, native overload (no `jev_noul2`/`jev_noul3`) |
| `jev_choice(state, instructions, criteria_json)` | `STRUCT(choice VARCHAR, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE))` |
| `jev_score(state, instructions, levels_json)` | `STRUCT(score DOUBLE, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE), legend MAP(VARCHAR, VARCHAR))` |
| `jev_extract(state, spec_json)` | `MAP(VARCHAR, STRUCT(value VARCHAR, p DOUBLE, p_none DOUBLE, confidence DOUBLE, n_candidates INTEGER, probabilities MAP(VARCHAR, DOUBLE)))` |
| `jev_cache_key(model, state, questions_json)` | VARCHAR (new, §3.2) |
| `jev_usage()` | table: one row of the counters in §3.5 (new; tier one has `duckjev.usage()` in Python) |
| `jev_usage_reset()` | BOOLEAN (new) |
| `jev_flush()` | BOOLEAN (new; checkpoints the cache file, like `duckjev.flush()`) |
| `jev_version()` | VARCHAR (new) |
| `jev_explain(query)` | table (new, §5.2) |

NULL rule: a NULL argument, or a state that is empty or whitespace, gives NULL with no
request. A vector whose rows are all NULL or empty makes no request and needs no key, so
constant folding at bind time never touches the network.

Per-row marshalling (tier one `marshal.py`, reproduce exactly): `probabilities` values
cast to DOUBLE, keys to VARCHAR, entry order as the API returned it; `score.legend` keys
are the index strings; `jev_extract` gives `value` NULL when `none` wins or the field had
no candidates, `p` is the probability of what was returned, `p_none` is 0.0 when there was
no `none` option, `n_candidates` counts the offered candidates, and `probabilities` lists
the candidates in offered order with 0.0 for any the API omitted.

Macros: every macro in `duckjev/macros.sql` is registered by the extension at load, from
the same file (the build copies `duckjev/macros.sql` into the extension's sources; a test
fails if the copy is stale). The one macro that goes away is the `jev_noul` overload
wrapper, because the function is overloaded natively. The window barrier inside `sem_join`
and `sem_dups` stays until milestone 4 proves the optimizer rule makes it unnecessary,
then it is removed from the shared file and tier one keeps working because the rule only
matters for cost, not results.

Settings (extension-scoped, all optional, all visible in `duckdb_settings()`, none holds
a secret):

| Setting | Default | Meaning |
|---|---|---|
| `duckjev_model` | `jev-1.13.0` | model id; part of every cache key |
| `duckjev_base_url` | `https://api.typesafe.ai` | the fixture server in tests |
| `duckjev_concurrency` | 16 | in-flight requests, process-wide |
| `duckjev_cache_path` | `~/.cache/duckjev/cache.duckdb` | `''` disables the cache |
| `duckjev_max_input_tokens` | unset | job budget; passing it raises `JevBudgetExceeded` |
| `duckjev_offline` | false | a cache miss is an error instead of a request |
| `duckjev_timeout_ms` | 30000 | per request |

Errors surface as DuckDB exceptions whose message starts with the tier-one class name
(`JevAuthError: ...`, `JevAPIError: ...`, `JevTransportError: ...`,
`JevBudgetExceeded: ...`, `JevQuestionError: ...`) so the bench harness and the tests can
match on it from either backend.

### 3.4 Cache file

Same file, same schema, same semantics as tier one `duckjev/cache.py`:

```sql
CREATE TABLE IF NOT EXISTS jev_cache (
    key VARCHAR PRIMARY KEY, model VARCHAR, answers VARCHAR,
    input_tokens BIGINT, output_tokens BIGINT, created_at DOUBLE)
```

`answers` is the compact JSON of the `answers` object. The extension opens the file on
its own `DuckDB` instance (not the user's), loads the table into an in-memory map on first
use, serves reads from the map, and writes misses to both the map and the file under one
lock, as `INSERT OR REPLACE` in one statement per judged vector. A file that cannot be
opened is a warning and a memory-only cache, never an error. Deduplication within a
vector happens on the key before the cache is consulted, so a vector of identical states
costs one request.

The point of keeping the file identical: the recorded caches of every tier-one benchmark
run (`bench/data/cache_*.duckdb` in the `banking77-round-two` and `sem-join` worktrees,
about 30 MB, gitignored) are the extension's offline test oracle (§4).

### 3.5 Usage counters

The same ten counters as tier one's `Usage`, process-wide, plus `est_usd`:
`rows, deduped, cache_hits, cache_misses, requests, input_tokens, output_tokens, retries,
rate_limited, overloaded`, `est_usd = input_tokens × 42 / 1e9`. `jev_usage()` returns
them as one row; `jev_usage_reset()` zeroes them. The budget check compares billed
`input_tokens` since the last reset against `duckjev_max_input_tokens` before every
request and after every response.

### 3.6 Execution model

DuckDB calls a scalar function once per vector (up to 2,048 rows) on each of its worker
threads. Per call: build `(state, questions)` per usable row, key them, deduplicate,
look up the cache, fan the misses out over a process-wide pool bounded by
`duckjev_concurrency` (the bound is global, not per thread, because DuckDB may run the
same function on many threads at once), block until every miss is answered or one has
failed, write the cache, marshal in the vector's order. Requests in flight are tracked in a
process-wide map keyed by cache key so two threads judging the same state at the same
time send it once. That in-flight map plus the synchronous cache is what makes a
double-evaluated expression cost one request, whatever the planner does (§5.1).

## 4. Acceptance: the differential test

The extension is correct when it reproduces tier one's recorded results without a single
request. Concretely:

1. `bench/data/` from the two worktrees named in §3.4 is copied into the tier-two
   worktree (it is gitignored there too).
2. `bench/banking77.py rescore --backend extension` and
   `bench/entity_matching.py rescore --backend extension` rebuild every recorded run's rows
   and metrics through the extension, with `duckjev_offline = true`, and write the run log
   to a suffixed file (`_ext`). The report generated from that file must be identical to
   `docs/results/banking77.md` and `docs/results/entity_matching.md` as committed (a diff
   with zero lines, checked by a test in `tests/`, skipped when the extension binary or
   the caches are absent).
3. The offline pytest suite of tier one runs unchanged against the extension backend:
   `DUCKJEV_BACKEND=extension uv run pytest -q` swaps `register()` for
   `LOAD` plus settings, and the fake transport for the replay transport fed from the same
   canned answers. Tests that reach into Python internals (`client_for`, the `httpx`
   transport) are marked `python_backend_only`.

Identical metrics through identical cache keys is the whole proof: it covers the wire
format (the key is the request), the question builders, the marshalling and the macros at
once, on 4,620 Banking77 rows and 8,778 labeled entity-matching pairs across every
recorded round, for $0.

## 5. What is new in tier two

### 5.1 Planner: semantic predicates last, and never duplicated

Mechanism (settled in §2.3): the functions are VOLATILE, which stops the pushdown copy;
the extension's `optimize_function` orders conjuncts and pulls all-Jev filters back above
inner joins; the client's dedupe and in-flight map make any evaluation the planner still
repeats cost CPU, not requests. The pull-up never increases requests: a Jev predicate
over one side's columns has the same set of distinct states above the join as below it,
or fewer when the join is selective, and distinct states are what is billed.

Behavior:

- In a conjunction, every predicate that contains a `jev_*` call is evaluated after every
  predicate that does not. Test: `EXPLAIN SELECT ... WHERE cheap AND jev_noul(...) >= 0.5
  AND cheap2` shows the Jev predicate last in the FILTER; the replay transport records
  only the rows that passed the cheap predicates.
- A `jev_*` expression that appears once in the query is evaluated once per row, whatever
  the planner rewrites. Test: the `sem_join` query with the window barrier removed makes
  exactly one request per blocked pair (tier one needed the barrier; PR #5 measured nine
  requests for five pairs without it).
- A filter with a `jev_*` predicate is not pushed below a join it could stay above. Test:
  a lookup join that keeps 1% of rows followed by a semantic filter judges 1% of the rows,
  not all of them.

### 5.2 `jev_explain(query)`

A table function that takes a query string, plans it with the current settings, and
returns one row per `jev_*` call site in the plan:

| column | meaning |
|---|---|
| `call` | the expression text |
| `operator` | where it sits (FILTER, PROJECTION, ...) |
| `est_rows` | the planner's cardinality estimate for that operator's input |
| `est_requests` | `est_rows` less the in-vector duplicate and cache-hit rates observed so far in this process (0 for both when nothing has run) |
| `est_input_tokens` | `est_requests × (state bytes ÷ 4 + question tokens)`, question tokens from the spec's byte length ÷ 4; a calibration constant (`duckjev_tokens_per_byte`) that the usage counters refine after the first live vector |
| `est_usd` | `est_input_tokens × 42 / 1e9` |

It never sends a request. It is an estimate and the column names say so. The number the
user cares about is the total row, which the function appends with `call = 'total'`.

## 6. Milestones

Each milestone is one PR from its own worktree, in the rounds discipline: a gate that is
observable, no live spend unless the gate needs it, and NEXT.md updated when it merges.
Milestones 0 to 2 are sequential. Milestones 3 and 4 are independent of each other and
can be two parallel worktrees once 2 has merged. Milestone 5 needs 2; 6 needs everything.

### M0: skeleton (`extension/`, one function, loads in Python and the CLI)

- `extension/` created from `duckdb/extension-template`, renamed with
  `scripts/bootstrap-template.py duckjev`, the `duckdb` submodule pinned at `v1.5.5` and
  `extension-ci-tools` at `v1.5-variegata`, `openssl` removed from `vcpkg.json` so no
  vcpkg toolchain is needed. Submodules are added from the `extension/` directory of this
  repo (`git submodule add` records paths relative to the repo root; that is fine, the
  subtree split in M6 carries `.gitmodules` entries under `extension/`; verify the split
  builds before relying on it).
- One scalar function, `jev_version()` → `'0.3.0-dev'`, and the settings of §3.3
  registered with their defaults (no behavior yet).
- `ninja-build` and `ccache` installed (`apt`), then `GEN=ninja make` from `extension/`;
  the first build compiles DuckDB and is the slow one. `make test` runs the
  sqllogictests; `make format-check` and `make tidy-check` pass.
- A pytest test under `tests/` that loads the built extension into the Python `duckdb`
  package with `allow_unsigned_extensions` and calls `jev_version()`, skipped when the
  binary is absent, so the Python suite stays green without a C++ build.
- The DuckDB CLI downloaded to `extension/.cli/` (gitignored) and the same call made
  from it; the command is recorded in `extension/README.md`.
- `.github/workflows/extension.yml`: on every PR touching `extension/`, the reusable
  `_extension_distribution.yml@v1.5-variegata` with `duckdb_version: v1.5.5` and
  `exclude_archs` listing every platform but `linux_amd64`, plus
  `_extension_code_quality.yml`; the same workflow with an empty `exclude_archs` behind
  `workflow_dispatch` and version tags. The workflow runs from the repo root (the
  reusable job checks the repo out at the root and runs `make` there), so the root
  gets a three-line `Makefile` that forwards to `extension/`: `%: ; $(MAKE) -C
  extension $@`. That is the only file tier two adds outside `extension/`, `tests/`,
  `bench/` and the docs. This repo has no CI today and is private; the PR job is one
  Linux build with ccache, about 30 minutes cold, and `concurrency:
  cancel-in-progress` keeps stacked pushes from queueing.
- Gate: `make test` green locally and in CI; `jev_version()` from Python and from the
  CLI; `uv run pytest -q` green with and without the binary.

### M1: client core, offline

- `client.cpp`: request body builder (§3.1), cache key (§3.2, with the pinned test
  vector), response parsing, error classes, retry loop with the tier-one backoff, the
  process-wide concurrency pool and in-flight map (§3.6), the usage counters and
  `jev_usage()` / `jev_usage_reset()` (§3.5), the budget check.
- `cache.cpp`: the cache file (§3.4) opened on a private `DuckDB` instance, loaded into
  the map on first use, written through on every judged vector.
- The transport is an interface with three implementations: `HTTPUtil` (live; not
  exercised by tests), `replay` (answers from a JSON fixture of request-key → response,
  records every request it saw, injects a status sequence; the C++ twin of
  `tests/fake.py`), and `refuse` (every request is an error; what `duckjev_offline`
  installs). Selected by `SET duckjev_transport = 'replay:/path/to/fixture.json'` for
  tests only; the setting exists in release builds because the bench harness uses it.
- The secret type `duckjev` with `api_key`, and the key resolution order of §1.7.
- The live transport's one integration check: a pytest test starts Python's
  `http.server` on `127.0.0.1` with a handler that answers like `tests/fake.py`, sets
  `duckjev_base_url` to it, `LOAD httpfs`, and runs one `jev_noul` through the real
  `HTTPUtil` path over plain HTTP. It is the only test in the repo that opens a socket
  (loopback; the conftest socket block is Python-side and cannot see DuckDB's curl
  anyway), it is marked `loopback`, and its docstring says so. If this test cannot be
  made to pass with httpfs's util (no POST, no request headers, no body), the fallback
  of §1.4 is taken in this milestone and recorded in NEXT.md.
- Gate: sqllogictests for key parity (the §3.2 vector), body shape, retry then failure,
  budget exceeded, offline miss, cache round trip across two `LOAD`s on one file. A
  pytest test computes fifty keys through tier one and through `jev_cache_key` over the
  Banking77 criteria files and asserts equality. The loopback test passes. No request
  leaves the machine.

### M2: the SQL surface

- `jev`, `jev_noul` (both arities), `jev_choice`, `jev_score`, `jev_extract`, with the
  marshalling of §3.3, all registered VOLATILE.
- `extension/scripts/gen_macros.py` turns `duckjev/macros.sql` into
  `src/macros_generated.hpp` (scalar and table macros, overloads grouped); a pytest test
  regenerates and diffs, so the two backends cannot drift. The macros are registered
  internal, in the system catalog, never in the user's database.
- httpfs autoload at the first live call (§2.3, item 5).
- The minimal backend switch in the Python package: `register(con, backend="extension",
  extension_path=None)` runs `LOAD` (the built binary, or `duckjev` once it is
  installed) and `SET`s the settings from its keyword arguments; the connection must
  have been opened with `allow_unsigned_extensions` until M6, and the error message
  says so. `usage()` and `flush()` route to `jev_usage()` and `jev_flush()` on that
  backend. Docs and polish wait for M5.
- The bench harness gains `--backend {python,extension}` and the `rescore` commands the
  `_ext` suffix (§4). `DUCKJEV_BACKEND=extension` for the pytest suite.
- Gate: the differential test of §4 passes on both benchmarks with zero requests; the
  offline pytest suite passes on the extension backend; then one live smoke of at most
  40 rows through the extension against the real endpoint, from the CLI, with
  `duckjev_max_input_tokens = 100000`, and its usage row recorded in NEXT.md. This is the
  only live spend before milestone 6.

### M3: the planner rule

- The `optimize_function` of §2.3 (items 2 and 3): conjunct ordering and the inner-join
  pull-up, behind `SET duckjev_planner = true` (default true) so a query can be run
  without it for comparison.
- Remove the window barrier from `sem_join` and `sem_dups` in `duckjev/macros.sql`; the
  tier-one Python backend keeps its `side_effects=False` UDFs and therefore still needs
  it, so the Python `register()` re-adds the barrier variant of those two macros after
  the shared file (one function, ten lines, with a comment saying why).
- Gate: the three `EXPLAIN` and request-count tests of §5.1, through the replay
  transport, on both backends; the sqllogictests assert the plan shape.

### M4: `jev_explain`

- The table function of §5.2 through `Planner` and `Optimizer`, walking the optimized
  plan for `BoundFunctionExpression`s whose name starts with `jev_`.
- Gate: on the Banking77 dev query it predicts the request count within the dedupe rate,
  and its `est_input_tokens` is within 25% of the recorded `input_tokens` of the same
  run after the calibration constant has seen one vector. Both from the run log, no
  network.

### M5: hosts and docs

- The CLI and Node examples (Node: `@duckdb/node-api`, `allow_unsigned_extensions`, one
  `LOAD`, one `sem_where` query) as scripts under `extension/examples/` that a test runs
  when the binary exists. Node is installed here; Go, Java and R are not and are not
  claimed until M6 makes `INSTALL ... FROM community` work.
- README: an "Install the extension" section beside the Python install, the settings
  table, the secret, the two examples, and the planner behavior in two sentences. The
  Python package version becomes 0.3.0 and `jev_version()` says the same. NEXT.md: §1.4
  what shipped, §3.4 marked done with what it taught.
- Gate: the README examples run as written from a fresh directory.

### M6: platforms and the registry

- The full distribution pipeline on a manual dispatch: linux amd64 and arm64, macOS,
  Windows (amd64 and mingw), and the three WASM targets. Platforms that fail to build
  are excluded, named in `description.yml`, and the reason recorded in NEXT.md.
- The WASM spike, before any WASM build is shipped: (1) does duckdb-wasm's
  `HTTPWasmUtil` implement `Post` (read `lib/src/http_wasm.cc` in `duckdb/duckdb-wasm`);
  (2) does `api.typesafe.ai` answer a browser preflight (`OPTIONS /v1/systemone` with
  `Origin` and `Access-Control-Request-Headers: authorization, content-type`) with
  `Access-Control-Allow-Origin`; one `curl -i` from a shell, no key, no cost. If either
  answer is no, WASM is excluded and NEXT.md says which one and what TypeSafe would have
  to change. A WASM build that loads but cannot reach the API is excluded, not shipped.
- `git subtree split --prefix=extension` pushed to `vsletten/duckjev-extension` (public,
  because the registry builds from a public repository), `description.yml` written from
  the field list in §2.2 (`build: cmake`, `language: C++`, `repo.ref` at the split
  commit, `excluded_platforms` from the matrix run), the submission PR opened against
  `duckdb/community-extensions`. The split is a script (`extension/scripts/publish.sh`)
  so it repeats on every release, and the first thing it checks is that the split tree
  builds on its own with its submodules. By then 1.5.6 has shipped (§2.2): bump the
  submodules and `duckdb_version` first, since the registry builds against the latest
  stable.
- Gate: the registry's CI builds green for every platform not excluded; `INSTALL duckjev
  FROM community; LOAD duckjev;` works from the CLI on this machine; the end-to-end demo
  (one `sem_where`, one soft group-by, one `sem_join` on the entity-matching demo rows)
  runs live from the CLI within the remaining budget of §1.9 and is recorded in NEXT.md.

## 7. How to start (for the coding session)

1. Read `docs/HANDOFF.md` §1 to §4 and this file. Skim `duckjev/client.py`,
   `duckjev/cache.py`, `duckjev/marshal.py`, `duckjev/functions.py`, `tests/fake.py`:
   together they are the specification of §3, and the C++ mirrors their structure file
   for file (`client`, `cache`, `marshal`, `functions`, plus `transport`, `planner`,
   `explain`, `secret`, `settings`).
2. Worktree `tier2-m0` from `main`; `apt install ninja-build ccache`; clone the template
   into `extension/`, bootstrap, pin the submodules, remove `openssl` from `vcpkg.json`,
   `GEN=ninja make`, `make test`. Do not start M1 until the M0 PR has merged: the build
   and CI shape is what every later PR sits on.
3. Copy `bench/data/` from the `banking77-round-two` and `sem-join` worktrees into the M2
   worktree before writing marshalling code; the differential test is the feedback loop,
   run it early and often (`rescore` takes seconds through the cache).
4. Budget: `duckjev_max_input_tokens` set on every live command, the 40-row pre-flight
   before the first live call of each milestone, $1 total for the tier. Report spend in
   every PR's test plan.
5. When a fact in §2 turns out wrong, fix it here in the same PR that discovers it, with
   what was found instead. This file is the record.

Out of scope for tier two, recorded so nobody drifts into it: tier three (§3.5 of
NEXT.md: maintained judgment columns, `jev_refresh`, the view-matching rewrite); any
change to what the tier-one Python UDFs return; publishing the Python package to PyPI;
the LTS (`andium`) build line; aggregate functions in the extension (the calibrated
aggregates stay SQL macros over `SUM`).

## 8. Rules that carry over

- Worktree per branch under `/mnt/data/vsletten/src/vsletten/duckjev/<branch>`; `main/` is
  read-only. Never change an existing worktree's branch.
- Key: `source ~/.bash_secrets >/dev/null 2>&1; export TYPESAFE_API_KEY`. The file sets
  the key without exporting it. Never print, log or commit the value. Before every push,
  grep the tree and the diff for it. The extension's own code never echoes the
  `Authorization` header, not even in a debug log.
- `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` stay green;
  the extension adds `make test` (sqllogictests) and `clang-format` on its sources.
- Commits end with `Co-Authored-By: <model name> <noreply@anthropic.com>`; PR bodies have
  `## Summary`, `## Test plan`, `## Breaking changes`, then the Claude Code line. After
  opening the PR, address every Sourcery comment, squash-merge when green with no
  unresolved thread, confirm the merged tree equals the reviewed head, fast-forward
  `main/`.
- Never publish a verdict from an untuned run; never hedged or condescending language
  about System One models in anything the user or others will read.
