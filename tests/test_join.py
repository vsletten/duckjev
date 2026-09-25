"""sem_join, sem_dedup, sem_topk and the pair macros, end to end against the fake transport.

The fake Noul answers 0.8 when the state mentions "card" and 0.1 otherwise; a pair state is
the JSON of both records, so any pair with "card" on either side matches at 0.5.
"""

from __future__ import annotations

import json

import duckdb
import pytest
from fake import FakeTransport

import duckjev

Q = "Do these two listings describe the same product?"
LEVELS = json.dumps(["budget", "mid-range", "premium"])


@pytest.fixture
def fake() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def con(fake: FakeTransport) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    duckjev.register(c, cache=False, api_key="test-key", transport=fake)
    c.execute(
        "CREATE TABLE a AS SELECT * FROM (VALUES "
        "(1, 'sony', 'sony card reader'), (2, 'sony', 'sony walkman'), "
        "(3, 'bose', 'bose speaker')) v(id, brand, name)"
    )
    c.execute(
        "CREATE TABLE b AS SELECT * FROM (VALUES "
        "(10, 'sony', 'sony reader for cards'), (11, 'sony', 'sony headphones'), "
        "(12, 'bose', 'bose card')) v(id, brand, name)"
    )
    return c


def test_pair_state_is_json_of_both_sides(con: duckdb.DuckDBPyConnection) -> None:
    (s,) = con.execute("SELECT jev_pair('x', struct_pack(name := 'y', price := 2))").fetchone()
    assert json.loads(s) == {"a": "x", "b": {"name": "y", "price": 2}}


def test_jev_match_and_sem_match(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    rows = con.execute(
        "SELECT jev_match('sony card', 'sony reader', $q), sem_match('a', 'b', $q, 0.5)",
        {"q": Q},
    ).fetchone()
    assert rows == (0.8, False)
    body = fake.requests[0]
    assert json.loads(body["state"]) == {"a": "sony card", "b": "sony reader"}
    assert body["questions"]["q"] == {"type": "noul", "instructions": Q}


def test_sem_join_blocks_then_judges(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    rows = con.execute(
        "SELECT left_row.id, right_row.id, p "
        "FROM sem_join('a', 'b', 'brand', 'name', 'name', $q, 0.5) ORDER BY 1, 2",
        {"q": Q},
    ).fetchall()
    # blocked pairs: (1,10) (1,11) (2,10) (2,11) (3,12); "card" on either side matches
    assert rows == [(1, 10, 0.8), (1, 11, 0.8), (2, 10, 0.8), (3, 12, 0.8)]
    assert len(fake.requests) == 5  # one request per blocked pair, judged once, none across blocks
    (n, expected, se) = con.execute(
        "SELECT count(*), expected_count(p), expected_count_stderr(p) "
        "FROM sem_join('a', 'b', 'brand', 'name', 'name', $q, 0.0)",
        {"q": Q},
    ).fetchone()
    assert n == 5 and expected == pytest.approx(4 * 0.8 + 0.1)
    assert len(fake.requests) == 10  # the cache is off in this fixture; a second query pays again
    assert se == pytest.approx((4 * 0.8 * 0.2 + 0.1 * 0.9) ** 0.5)


def test_sem_join_works_on_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE VIEW a2 AS SELECT id, brand, struct_pack(name := name) AS rec FROM a")
    con.execute("CREATE VIEW b2 AS SELECT id, brand, struct_pack(name := name) AS rec FROM b")
    rows = con.execute(
        "SELECT left_row.rec.name, right_row.id "
        "FROM sem_join('a2', 'b2', 'brand', 'rec', 'rec', $q, 0.5) ORDER BY 2, 1",
        {"q": Q},
    ).fetchall()
    assert rows[0] == ("sony card reader", 10)


def test_sem_dups_and_dedup(con: duckdb.DuckDBPyConnection, fake: FakeTransport) -> None:
    con.execute(
        "CREATE TABLE u AS SELECT id, brand, name FROM a UNION ALL SELECT id, brand, name FROM b"
    )
    dups = con.execute(
        "SELECT id, duplicate_id FROM sem_dups('u', 'id', 'brand', 'name', $q, 0.5) ORDER BY 1, 2",
        {"q": Q},
    ).fetchall()
    # sony block pairs (1,2) (1,10) (1,11) (2,10) (2,11) (10,11) match when a side says card
    # (rows 1 and 10); the bose block's (3,12) matches through "bose card"
    assert dups == [(1, 2), (1, 10), (1, 11), (2, 10), (3, 12), (10, 11)]
    kept = con.execute(
        "SELECT id FROM sem_dedup('u', 'id', 'brand', 'name', $q, 0.5) ORDER BY 1", {"q": Q}
    ).fetchall()
    assert kept == [(1,), (3,)]  # each block keeps its earliest row; 12 duplicates 3
    assert all(json.loads(r["state"])["a"] != json.loads(r["state"])["b"] for r in fake.requests)


def test_sem_topk_orders_by_score(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT id, name, score, confidence "
        "FROM sem_topk('a', 'name', 'How premium is this product?', $levels, 2)",
        {"levels": LEVELS},
    ).fetchall()
    assert len(rows) == 2
    assert all(r[2] == 2.0 and r[3] == 1.0 for r in rows)  # the fake scores the top level
    cols = [
        d[0]
        for d in con.execute(
            "SELECT * FROM sem_topk('a', 'name', 'q', $levels, 1)", {"levels": LEVELS}
        ).description
    ]
    assert cols == ["id", "brand", "name", "score", "confidence"]


def test_new_macros_are_registered(con: duckdb.DuckDBPyConnection) -> None:
    names = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT function_name FROM duckdb_functions() "
            "WHERE function_name LIKE 'jev_%' OR function_name LIKE 'sem_%'"
        ).fetchall()
    }
    assert {
        "jev_pair",
        "jev_match",
        "sem_match",
        "sem_join",
        "sem_dups",
        "sem_dedup",
        "sem_topk",
    } <= names
