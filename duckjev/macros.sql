-- Installed on the user's connection by duckjev.register(). One statement per ';'.

-- typed noul with optional criteria (DuckDB does not overload Python UDFs; macros overload)
CREATE OR REPLACE MACRO jev_noul(state, q) AS jev_noul2(state, q),
                                (state, q, criteria) AS jev_noul3(state, q, criteria);

-- filter
CREATE OR REPLACE MACRO sem_where(state, q, thr) AS jev_noul(state, q) >= thr;

-- calibrated aggregation over a probability column p
CREATE OR REPLACE MACRO expected_count(p) AS SUM(p);
CREATE OR REPLACE MACRO expected_count_var(p) AS SUM(p * (1 - p));
CREATE OR REPLACE MACRO expected_count_stderr(p) AS sqrt(SUM(p * (1 - p)));

-- distribution helpers over a jev_choice struct c
CREATE OR REPLACE MACRO jev_argmax(c) AS c.choice;
CREATE OR REPLACE MACRO jev_p(c, k) AS coalesce(c.probabilities[k], 0.0);
-- The UNNEST-subquery form fails with "UNNEST() for correlated expressions is not
-- supported" in some plans on DuckDB 1.5, so this uses list lambdas instead.
CREATE OR REPLACE MACRO jev_runner_up(c) AS
  list_reverse_sort(
    list_transform(
      list_filter(map_entries(c.probabilities), lambda e: e.key <> c.choice),
      lambda e: {'p': e.value, 'key': e.key}))[1].key;
