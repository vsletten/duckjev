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

-- jev_extract (select, don't generate): code proposes candidates, Jev picks one per field.
-- jev_field builds one field's spec; json_object(name, jev_field(..), ...) builds the whole spec.
CREATE OR REPLACE MACRO jev_field(instructions, candidates) AS
    json_object('instructions', instructions, 'candidates', candidates),
  (instructions, candidates, none_option) AS
    json_object('instructions', instructions, 'candidates', candidates, 'none', none_option);

-- candidate builders; each over-finds on purpose, since Jev can only pick what is offered.
-- amounts with two decimals, optionally comma-grouped: 9.00, 1,315.50 (no currency symbol)
CREATE OR REPLACE MACRO jev_money_spans(text) AS
  regexp_extract_all(text, '\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2}');
-- numeric and month-name dates: 25/12/2018, 12-01-19, 2018-12-25, 25 DEC 2018,
-- 02/JAN/2017, DEC 25, 2018, and compact 20181225 / 25122018
CREATE OR REPLACE MACRO jev_date_spans(text) AS
  regexp_extract_all(text, '(?i)\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|\d{1,2}[ /-]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[ ,/-]*\d{2,4}|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]* \d{1,2},? \d{4}|20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])|(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])20\d{2})\b');
-- every run of 1..k consecutive lines, joined with a space (multi-line names, addresses)
CREATE OR REPLACE MACRO jev_line_windows(lines, k) AS
  flatten(list_transform(range(1, k + 1), lambda w:
    list_transform(range(1, len(lines) - w + 2),
      lambda i: array_to_string(lines[i:i + w - 1], ' '))));
