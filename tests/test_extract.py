from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from fake import FakeTransport

import duckjev
from duckjev import marshal
from duckjev.marshal import JevQuestionError


def starred(state: str, options: list[str]) -> dict[str, float]:
    """Fake Jev: 0.8 on the first option written as ``*option`` in the state, else on none."""
    top = next((o for o in options if f"*{o}" in state), "none")
    rest = 0.2 / (len(options) - 1)
    return {o: 0.8 if o == top else rest for o in options}


RECEIPTS = [
    (1, "SHOP A\nTOTAL *9.00\nCASH 10.00\nCHANGE 1.00\nDATE *25/12/2018"),
    (2, "SHOP B\nSUBTOTAL 3.00\nTOTAL *3.50\n2018-01-02 12-01-19"),
    (3, "no amounts and no dates here"),
    (4, None),
    (5, "   "),
]
SPEC_SQL = """json_object(
    'total', jev_field('Which amount is the total paid?', jev_money_spans(text)),
    'date',  jev_field('On what date was the receipt issued?', jev_date_spans(text)))"""


@pytest.fixture
def fake() -> FakeTransport:
    return FakeTransport(choice=starred)


@pytest.fixture
def con(fake: FakeTransport) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    duckjev.register(c, cache=False, api_key="test-key", transport=fake)
    c.execute("CREATE TABLE r(id INTEGER, text VARCHAR)")
    c.executemany("INSERT INTO r VALUES (?, ?)", RECEIPTS)
    return c


# --------------------------------------------------------------------------- marshal


def spec(**fields: dict) -> str:
    return json.dumps(fields)


def test_extract_questions_one_choice_per_field_with_none() -> None:
    questions, fields = marshal.extract_questions(
        spec(
            total={"instructions": "total?", "candidates": [" 9.00", "10.00", "9.00", "", None]},
            date={"instructions": "date?", "candidates": ["25/12/2018"]},
        )
    )
    assert set(questions) == {"total", "date"}
    q = questions["total"]
    assert q["type"] == "choice" and q["instructions"] == "total?"
    assert list(q["criteria"]) == ["9.00", "10.00", "none"]  # stripped, deduped, order kept
    assert q["criteria"]["9.00"] is None
    assert q["criteria"]["none"] == marshal.NONE_DESCRIPTION
    assert [f.name for f in fields] == ["total", "date"]
    assert fields[0].candidates == ("9.00", "10.00") and fields[0].none_key == "none"


def test_extract_questions_descriptions_and_none_options() -> None:
    questions, fields = marshal.extract_questions(
        spec(
            a={"instructions": "a?", "candidates": {"x": "on the TOTAL line", "y": None}},
            b={"instructions": "b?", "candidates": ["x"], "none": False},
            c={"instructions": "c?", "candidates": ["x"], "none": "No amount is the tip."},
            d={"instructions": "d?", "candidates": ["none", "x"]},
            e={"instructions": "e?", "candidates": ["x"], "none": None},
        )
    )
    assert questions["a"]["criteria"] == {
        "x": "on the TOTAL line",
        "y": None,
        "none": marshal.NONE_DESCRIPTION,
    }
    assert list(questions["b"]["criteria"]) == ["x"] and fields[1].none_key is None
    assert questions["c"]["criteria"]["none"] == "No amount is the tip."
    assert list(questions["d"]["criteria"]) == ["none", "x", "none of these"]
    assert fields[3].none_key == "none of these"
    assert questions["e"]["criteria"]["none"] == marshal.NONE_DESCRIPTION


def test_extract_questions_field_without_candidates_is_not_asked() -> None:
    questions, fields = marshal.extract_questions(
        spec(
            total={"instructions": "total?", "candidates": []},
            date={"instructions": "date?", "candidates": None},
        )
    )
    assert questions == {}
    assert [(f.name, f.candidates) for f in fields] == [("total", ()), ("date", ())]


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("nope", "not valid JSON"),
        ("[]", "non-empty JSON object"),
        ("{}", "non-empty JSON object"),
        ('{"total": {"candidates": ["1.00"]}}', "needs an object with instructions"),
        ('{"total": {"instructions": "t?", "candidates": "1.00"}}', "array or object"),
    ],
)
def test_extract_questions_rejects_bad_specs(bad: str, match: str) -> None:
    with pytest.raises(JevQuestionError, match=match):
        marshal.extract_questions(bad)


def test_extract_questions_option_cap() -> None:
    many = [f"{i}.00" for i in range(254)]
    questions, _ = marshal.extract_questions(spec(t={"instructions": "t?", "candidates": many}))
    assert len(questions["t"]["criteria"]) == 255
    with pytest.raises(JevQuestionError, match="more than the 254 a Choice can take beside none"):
        marshal.extract_questions(spec(t={"instructions": "t?", "candidates": [*many, "x"]}))
    questions, _ = marshal.extract_questions(
        spec(t={"instructions": "t?", "candidates": [*many, "x"], "none": False})
    )
    assert len(questions["t"]["criteria"]) == 255


def test_extract_array_maps_answers() -> None:
    _, fields = marshal.extract_questions(
        spec(
            total={"instructions": "t?", "candidates": ["9.00", "10.00"]},
            date={"instructions": "d?", "candidates": ["25/12/2018"]},
            tip={"instructions": "tip?", "candidates": []},
        )
    )
    answers = {
        "total": {
            "type": "choice",
            "choice": "9.00",
            "probabilities": {"9.00": 0.7, "10.00": 0.2, "none": 0.1},
            "confidence": 0.6,
        },
        "date": {
            "type": "choice",
            "choice": "none",
            "probabilities": {"25/12/2018": 0.3, "none": 0.7},
            "confidence": 0.4,
        },
    }
    arr = marshal.extract_array([fields, None], [answers, None])
    assert arr.type == marshal.EXTRACT_TYPE
    assert isinstance(arr, pa.MapArray)
    row, null_row = arr.to_pylist()
    assert null_row is None
    got = dict(row)
    assert got["total"]["value"] == "9.00"
    assert got["total"]["p"] == pytest.approx(0.7)
    assert got["total"]["p_none"] == pytest.approx(0.1)
    assert got["total"]["n_candidates"] == 2
    assert got["total"]["probabilities"] == [("9.00", 0.7), ("10.00", 0.2)]
    assert got["date"]["value"] is None  # none won
    assert got["date"]["p"] == pytest.approx(0.7) and got["date"]["p_none"] == pytest.approx(0.7)
    assert got["tip"] == {
        "value": None,
        "p": None,
        "p_none": None,
        "confidence": None,
        "n_candidates": 0,
        "probabilities": [],
    }


# --------------------------------------------------------------------------- SQL


def test_candidate_builder_macros(con: duckdb.DuckDBPyConnection) -> None:
    money, dates, windows = con.execute(
        """SELECT jev_money_spans('TOTAL RM 1,315.50 TAX 115.50 QTY 2 -5.59'),
                  jev_date_spans('25/12/2018 8:13 PM, 12-01-19, 2018-12-25, 25 DEC 2018, '
                                 || '02/JAN/2017, DEC 25, 2018, 20181225, TEL 07-3507405'),
                  jev_line_windows(['NO 1, JALAN A', 'TAMAN B', '81100 JOHOR'], 2)"""
    ).fetchone()
    assert money == ["1,315.50", "115.50", "5.59"]
    assert dates == [
        "25/12/2018",
        "12-01-19",
        "2018-12-25",
        "25 DEC 2018",
        "02/JAN/2017",
        "DEC 25, 2018",
        "20181225",
    ]
    assert windows == [
        "NO 1, JALAN A",
        "TAMAN B",
        "81100 JOHOR",
        "NO 1, JALAN A TAMAN B",
        "TAMAN B 81100 JOHOR",
    ]


def test_jev_extract_end_to_end(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    rel = con.execute(f"SELECT id, jev_extract(text, {SPEC_SQL}) AS x FROM r ORDER BY id")
    assert "MAP(VARCHAR, STRUCT" in str(rel.description[1][1])
    rows = dict(rel.fetchall())
    one = dict(rows[1])
    assert one["total"]["value"] == "9.00" and one["total"]["p"] == pytest.approx(0.8)
    assert list(one["total"]["probabilities"]) == ["9.00", "10.00", "1.00"]  # candidate order
    assert one["date"]["value"] == "25/12/2018"
    two = dict(rows[2])
    assert two["total"]["value"] == "3.50"
    assert two["date"]["value"] is None and two["date"]["p_none"] == pytest.approx(0.8)
    three = dict(rows[3])  # no candidates for any field: no request, nothing asked
    assert three["total"]["n_candidates"] == 0 and three["total"]["value"] is None
    assert three["date"]["p"] is None
    assert rows[4] is None and rows[5] is None
    # one fused request per judged row, one Choice per field
    assert len(fake.requests) == 2
    assert all(set(r["questions"]) == {"total", "date"} for r in fake.requests)
    assert all(q["type"] == "choice" for r in fake.requests for q in r["questions"].values())


def test_jev_extract_field_access_in_sql(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        f"""SELECT id, x['total'].value AS total, round(x['total'].p, 2) AS p,
                   x['date'].value AS date, x['total'].n_candidates AS n
            FROM (SELECT id, jev_extract(text, {SPEC_SQL}) AS x FROM r WHERE id <= 2)
            ORDER BY id"""
    ).fetchall()
    assert rows == [(1, "9.00", 0.8, "25/12/2018", 3), (2, "3.50", 0.8, None, 2)]


def test_jev_extract_dedupes_identical_rows(fake: FakeTransport) -> None:
    c = duckdb.connect()
    duckjev.register(c, cache=False, api_key="k", transport=fake)
    c.execute("CREATE TABLE r AS SELECT 'TOTAL *4.20 CASH 5.00' AS text FROM range(3000)")
    n = c.execute(
        "SELECT count(*) FROM r WHERE jev_extract(text, json_object('t', "
        "jev_field('total?', jev_money_spans(text))))['t'].value = '4.20'"
    ).fetchone()[0]
    assert n == 3000
    assert len(fake.requests) <= 2  # at most one per vector; DuckDB splits 3000 rows in two


def test_jev_extract_errors_surface_in_sql(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.Error, match="more than the 254"):
        con.execute(
            "SELECT jev_extract('x', json_object('t', jev_field('t?', "
            "list_transform(range(300), lambda i: i::VARCHAR))))"
        ).fetchall()
    with pytest.raises(duckdb.Error, match="needs an object with instructions"):
        con.execute("""SELECT jev_extract('x', '{"t": {"candidates": ["1"]}}')""").fetchall()


def test_jev_extract_without_key_offline_paths() -> None:
    fake = FakeTransport(choice=starred)
    c = duckdb.connect()
    duckjev.register(c, cache=False, transport=fake)
    # NULL / empty state, and a row with no candidates at all, never need the key
    row = c.execute(
        "SELECT jev_extract(NULL::VARCHAR, '{}'), "
        "jev_extract('no numbers', json_object('t', jev_field('t?', jev_money_spans('none'))))"
    ).fetchone()
    assert row[0] is None and dict(row[1])["t"]["n_candidates"] == 0
    with pytest.raises(duckdb.Error, match="TYPESAFE_API_KEY is not set"):
        c.execute(
            "SELECT jev_extract('TOTAL 1.00', json_object('t', jev_field('t?', ['1.00'])))"
        ).fetchall()
    assert fake.attempts == 0


def test_jev_extract_cache_rerun_is_free(tmp_path: Path) -> None:
    fake = FakeTransport(choice=starred)
    c = duckdb.connect()
    duckjev.register(c, api_key="k", transport=fake, cache_path=tmp_path / "cache.duckdb")
    c.execute("CREATE TABLE r(id INTEGER, text VARCHAR)")
    c.executemany("INSERT INTO r VALUES (?, ?)", RECEIPTS)
    q = f"SELECT id, jev_extract(text, {SPEC_SQL}) FROM r ORDER BY id"
    first = c.execute(q).fetchall()
    duckjev.usage(reset=True)
    assert c.execute(q).fetchall() == first
    u = duckjev.usage()
    assert u["requests"] == 0 and u["cache_hits"] == 2 and u["est_usd"] == 0
