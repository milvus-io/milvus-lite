"""Array field edge-case tests ported from Milvus test suite.

Source: milvus/tests/python_client/milvus_client_v2/test_milvus_client_search_array.py
        milvus/tests/python_client/milvus_client_v2/test_milvus_client_search_json.py
        milvus/tests/python_client/milvus_client_v2/test_milvus_client_search_invalid.py

Each test is self-contained: create -> insert -> index -> load -> test -> drop.
Uses BRUTE_FORCE index with COSINE metric.  10-20 rows, dim=4.

Covers edge cases NOT already in test_array_field.py:
  - array_contains on INT64 / FLOAT / BOOL element types
  - array_contains_all / array_contains_any on INT64 arrays
  - NOT array_contains negation
  - array_length with various comparison operators
  - array index access at non-zero positions
  - combined array filters with OR
  - array filter returning empty results
  - array output fields roundtrip for all four element types
  - search (vector + array filter) across all element types
  - arrays of varying lengths within the same collection
"""

import pytest
from pymilvus import DataType, MilvusClient

# ---- constants ----
DIM = 4
COLLECTION_PREFIX = "arr_milvus_"


# ---- helpers ----

def _unit_vec(idx: int) -> list:
    """Return a simple dim=4 unit-ish vector distinguishable by index."""
    v = [0.0] * DIM
    v[idx % DIM] = 1.0
    return v


def _setup(client: MilvusClient, name: str, rows: list[dict],
           extra_fields: list[tuple] | None = None):
    """Create collection, insert rows, build BRUTE_FORCE index, load."""
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("int_arr", DataType.ARRAY,
                     element_type=DataType.INT64, max_capacity=20)
    schema.add_field("str_arr", DataType.ARRAY,
                     element_type=DataType.VARCHAR, max_capacity=20,
                     max_length=64)
    schema.add_field("flt_arr", DataType.ARRAY,
                     element_type=DataType.FLOAT, max_capacity=20)
    schema.add_field("bool_arr", DataType.ARRAY,
                     element_type=DataType.BOOL, max_capacity=20)
    if extra_fields:
        for args in extra_fields:
            schema.add_field(*args)
    client.create_collection(name, schema=schema)
    client.insert(name, rows)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vec", index_type="BRUTE_FORCE",
                  metric_type="COSINE", params={})
    client.create_index(name, idx)
    client.load_collection(name)


# ---- deterministic data (20 rows) ----

def _make_rows(n: int = 20) -> list[dict]:
    """Generate n deterministic rows.

    int_arr  = [i%10, (i+1)%10, (i+2)%10]
    str_arr  = [f"s_{i%5}", f"s_{(i+1)%5}"]
    flt_arr  = [float(i%10), float((i*3)%10)]
    bool_arr = [i%2==0, i%3==0]
    """
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "vec": _unit_vec(i),
            "int_arr": [i % 10, (i + 1) % 10, (i + 2) % 10],
            "str_arr": [f"s_{i % 5}", f"s_{(i + 1) % 5}"],
            "flt_arr": [float(i % 10), float((i * 3) % 10)],
            "bool_arr": [i % 2 == 0, i % 3 == 0],
        })
    return rows


# ======================================================================
# Tests
# ======================================================================


class TestArrayContainsIntTypes:
    """Ported from TestSearchArrayShared.test_search_array_contains.

    Original validates array_contains on int64_array with deterministic
    data.  Here we test INT64, FLOAT, and BOOL element types via query
    (pure scalar) to isolate filter logic.
    """

    def test_array_contains_int64(self, milvus_client):
        """array_contains on INT64 array field.

        Ref: TestSearchArrayShared.test_search_array_contains
        Every row whose int_arr includes 5 should be returned.
        """
        name = COLLECTION_PREFIX + "contains_int64"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = 5
        expected_ids = {i for i in range(20)
                        if target in [i % 10, (i + 1) % 10, (i + 2) % 10]}

        res = milvus_client.query(name, filter=f"array_contains(int_arr, {target})",
                                  output_fields=["int_arr"], limit=100)
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert target in r["int_arr"]

        milvus_client.drop_collection(name)

    def test_array_contains_float(self, milvus_client):
        """array_contains on FLOAT array field.

        Ref: test_search_expr_array_contains_any_with_float_field (search_json.py)
        Rows whose flt_arr includes 3.0 should be returned.
        """
        name = COLLECTION_PREFIX + "contains_float"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = 3.0
        expected_ids = {i for i in range(20)
                        if target in [float(i % 10), float((i * 3) % 10)]}

        res = milvus_client.query(name, filter="array_contains(flt_arr, 3.0)",
                                  output_fields=["flt_arr"], limit=100)
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert target in r["flt_arr"]

        milvus_client.drop_collection(name)

    def test_array_contains_bool(self, milvus_client):
        """array_contains on BOOL array field.

        Ref: TestSearchArrayShared schema includes bool_array.
        Every row has bool_arr = [i%2==0, i%3==0].  Searching for true
        should match rows where either element is true.
        """
        name = COLLECTION_PREFIX + "contains_bool"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        expected_ids = {i for i in range(20)
                        if (i % 2 == 0) or (i % 3 == 0)}

        res = milvus_client.query(name, filter="array_contains(bool_arr, true)",
                                  output_fields=["bool_arr"], limit=100)
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert True in r["bool_arr"]

        milvus_client.drop_collection(name)


class TestArrayContainsAllAny:
    """Ported from TestSearchArrayShared.test_search_array_contains_all
    and test_search_array_contains_any.

    Original uses int64_array with mod-50 data.  Here we use mod-10
    with 20 rows.
    """

    def test_array_contains_all_int64(self, milvus_client):
        """array_contains_all: every hit must contain ALL specified values.

        Ref: TestSearchArrayShared.test_search_array_contains_all
        With int_arr = [i%10, (i+1)%10, (i+2)%10], rows containing
        both 0 and 1 are those where {0,1} is a subset of the triple.
        """
        name = COLLECTION_PREFIX + "all_int64"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target_vals = [0, 1]
        expected_ids = {i for i in range(20)
                        if set(target_vals) <= {i % 10, (i + 1) % 10, (i + 2) % 10}}

        res = milvus_client.query(
            name,
            filter=f"array_contains_all(int_arr, {target_vals})",
            output_fields=["int_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            for v in target_vals:
                assert v in r["int_arr"]

        milvus_client.drop_collection(name)

    def test_array_contains_any_int64(self, milvus_client):
        """array_contains_any: every hit must contain at least one value.

        Ref: TestSearchArrayShared.test_search_array_contains_any
        """
        name = COLLECTION_PREFIX + "any_int64"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target_vals = [8, 9]
        expected_ids = {i for i in range(20)
                        if set(target_vals) & {i % 10, (i + 1) % 10, (i + 2) % 10}}

        res = milvus_client.query(
            name,
            filter=f"array_contains_any(int_arr, {target_vals})",
            output_fields=["int_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert any(v in r["int_arr"] for v in target_vals)

        milvus_client.drop_collection(name)

    def test_array_contains_any_varchar(self, milvus_client):
        """array_contains_any on VARCHAR array.

        Ref: test_search_expr_array_contains_any (search_json.py)
        """
        name = COLLECTION_PREFIX + "any_varchar"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target_vals = ["s_0", "s_4"]
        expected_ids = {i for i in range(20)
                        if set(target_vals) & {f"s_{i % 5}", f"s_{(i + 1) % 5}"}}

        res = milvus_client.query(
            name,
            filter='array_contains_any(str_arr, ["s_0", "s_4"])',
            output_fields=["str_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids

        milvus_client.drop_collection(name)


class TestNotArrayContains:
    """Ported from test_search_expr_not_array_contains (search_json.py).

    NOT array_contains should return the complement set.
    """

    def test_not_array_contains_int(self, milvus_client):
        """NOT array_contains returns rows that do NOT contain the value.

        Ref: test_search_expr_not_array_contains
        """
        name = COLLECTION_PREFIX + "not_contains"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = 5
        match_ids = {i for i in range(20)
                     if target in [i % 10, (i + 1) % 10, (i + 2) % 10]}
        expected_ids = set(range(20)) - match_ids

        res = milvus_client.query(
            name,
            filter=f"not array_contains(int_arr, {target})",
            output_fields=["int_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert target not in r["int_arr"]

        milvus_client.drop_collection(name)

    def test_not_array_contains_varchar(self, milvus_client):
        """NOT array_contains on VARCHAR.

        Ref: test_search_expr_not_array_contains (search_json.py)
        """
        name = COLLECTION_PREFIX + "not_contains_str"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = "s_0"
        match_ids = {i for i in range(20)
                     if target in [f"s_{i % 5}", f"s_{(i + 1) % 5}"]}
        expected_ids = set(range(20)) - match_ids

        res = milvus_client.query(
            name,
            filter='not array_contains(str_arr, "s_0")',
            output_fields=["str_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids

        milvus_client.drop_collection(name)


class TestArrayLength:
    """Ported from TestSearchArrayShared.test_search_array_length.

    Original checks array_length == 3.  Here we test multiple comparison
    operators and arrays with varying lengths.
    """

    def test_array_length_equality(self, milvus_client):
        """array_length(field) == N exact match.

        Ref: TestSearchArrayShared.test_search_array_length
        """
        name = COLLECTION_PREFIX + "len_eq"
        # Create rows with varying int_arr lengths
        rows = []
        for i in range(15):
            length = (i % 3) + 1  # lengths cycle: 1, 2, 3
            rows.append({
                "id": i, "vec": _unit_vec(i),
                "int_arr": list(range(length)),
                "str_arr": [f"x_{j}" for j in range(length)],
                "flt_arr": [float(j) for j in range(length)],
                "bool_arr": [j % 2 == 0 for j in range(length)],
            })
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="array_length(int_arr) == 2",
            output_fields=["int_arr"], limit=100,
        )
        for r in res:
            assert len(r["int_arr"]) == 2
        expected_ids = {i for i in range(15) if (i % 3) + 1 == 2}
        assert {r["id"] for r in res} == expected_ids

        milvus_client.drop_collection(name)

    def test_array_length_greater_than(self, milvus_client):
        """array_length(field) > N.

        Ref: TestSearchArrayShared.test_search_array_length (variation)
        """
        name = COLLECTION_PREFIX + "len_gt"
        rows = []
        for i in range(12):
            length = (i % 4) + 1  # lengths cycle: 1, 2, 3, 4
            rows.append({
                "id": i, "vec": _unit_vec(i),
                "int_arr": list(range(length)),
                "str_arr": [f"x_{j}" for j in range(length)],
                "flt_arr": [float(j) for j in range(length)],
                "bool_arr": [j % 2 == 0 for j in range(length)],
            })
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="array_length(int_arr) > 2",
            output_fields=["int_arr"], limit=100,
        )
        for r in res:
            assert len(r["int_arr"]) > 2
        expected_ids = {i for i in range(12) if (i % 4) + 1 > 2}
        assert {r["id"] for r in res} == expected_ids

        milvus_client.drop_collection(name)

    def test_array_length_not_equal(self, milvus_client):
        """array_length(field) != N.

        Ref: TestSearchArrayShared.test_search_array_length (variation)
        """
        name = COLLECTION_PREFIX + "len_ne"
        rows = []
        for i in range(12):
            length = (i % 4) + 1
            rows.append({
                "id": i, "vec": _unit_vec(i),
                "int_arr": list(range(length)),
                "str_arr": [f"x"],
                "flt_arr": [0.0],
                "bool_arr": [True],
            })
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="array_length(int_arr) != 1",
            output_fields=["int_arr"], limit=100,
        )
        for r in res:
            assert len(r["int_arr"]) != 1
        expected_ids = {i for i in range(12) if (i % 4) + 1 != 1}
        assert {r["id"] for r in res} == expected_ids

        milvus_client.drop_collection(name)


class TestArrayIndexAccess:
    """Ported from TestSearchArrayShared.test_search_array_access and
    test_search_with_expression_invalid_array_one (search_invalid.py).

    Original checks int64_array[0] == 10.  Here we test non-zero
    indices and different comparison operators.
    """

    def test_array_access_non_zero_index(self, milvus_client):
        """field[1] and field[2] access.

        Ref: TestSearchArrayShared.test_search_array_access (variation)
        int_arr = [i%10, (i+1)%10, (i+2)%10], so int_arr[1] == (i+1)%10.
        """
        name = COLLECTION_PREFIX + "access_idx1"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = 7
        # int_arr[1] == (i+1)%10, so (i+1)%10 == 7 => i%10 == 6
        expected_ids = {i for i in range(20) if (i + 1) % 10 == target}

        res = milvus_client.query(
            name, filter=f"int_arr[1] == {target}",
            output_fields=["int_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert r["int_arr"][1] == target

        milvus_client.drop_collection(name)

    def test_array_access_with_comparison(self, milvus_client):
        """field[0] >= value comparison.

        Ref: TestSearchArrayShared.test_search_array_access (variation)
        """
        name = COLLECTION_PREFIX + "access_cmp"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        # int_arr[0] >= 8  => i%10 >= 8
        expected_ids = {i for i in range(20) if i % 10 >= 8}

        res = milvus_client.query(
            name, filter="int_arr[0] >= 8",
            output_fields=["int_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids

        milvus_client.drop_collection(name)

    def test_array_access_varchar(self, milvus_client):
        """field[0] == "value" on VARCHAR array.

        Ref: TestSearchArrayShared.test_search_array_access (varchar variant)
        str_arr = [f"s_{i%5}", f"s_{(i+1)%5}"], so str_arr[0] == "s_2".
        """
        name = COLLECTION_PREFIX + "access_str"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        expected_ids = {i for i in range(20) if f"s_{i % 5}" == "s_2"}

        res = milvus_client.query(
            name, filter='str_arr[0] == "s_2"',
            output_fields=["str_arr"], limit=100,
        )
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids
        for r in res:
            assert r["str_arr"][0] == "s_2"

        milvus_client.drop_collection(name)


class TestSearchWithArrayFilter:
    """Ported from TestSearchArrayShared.test_search_array_contains /
    test_search_array_contains_all / test_search_array_contains_any.

    Original tests combine vector search with array filters via the
    search API.  Here we validate the same through pymilvus client.search().
    """

    def test_search_with_array_contains_int(self, milvus_client):
        """Vector search + array_contains filter on INT64 array.

        Ref: TestSearchArrayShared.test_search_array_contains
        """
        name = COLLECTION_PREFIX + "search_int"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target = 3
        expected_ids = {i for i in range(20)
                        if target in [i % 10, (i + 1) % 10, (i + 2) % 10]}

        results = milvus_client.search(
            name, data=[_unit_vec(0)], limit=20,
            filter=f"array_contains(int_arr, {target})",
            output_fields=["int_arr"],
        )
        hit_ids = {h["id"] for h in results[0]}
        # All returned hits must match the filter
        for h in results[0]:
            assert target in h["entity"]["int_arr"]
        # All returned hits must be valid IDs
        assert hit_ids <= expected_ids

        milvus_client.drop_collection(name)

    def test_search_with_array_contains_all(self, milvus_client):
        """Vector search + array_contains_all filter.

        Ref: TestSearchArrayShared.test_search_array_contains_all
        """
        name = COLLECTION_PREFIX + "search_all"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        target_vals = [0, 1, 2]
        expected_ids = {i for i in range(20)
                        if set(target_vals) <= {i % 10, (i + 1) % 10, (i + 2) % 10}}

        results = milvus_client.search(
            name, data=[_unit_vec(0)], limit=20,
            filter=f"array_contains_all(int_arr, {target_vals})",
            output_fields=["int_arr"],
        )
        for h in results[0]:
            arr = h["entity"]["int_arr"]
            for v in target_vals:
                assert v in arr

        milvus_client.drop_collection(name)

    def test_search_with_array_length(self, milvus_client):
        """Vector search + array_length filter.

        Ref: TestSearchArrayShared.test_search_array_length
        """
        name = COLLECTION_PREFIX + "search_len"
        rows = []
        for i in range(15):
            length = (i % 3) + 1
            rows.append({
                "id": i, "vec": _unit_vec(i),
                "int_arr": list(range(length)),
                "str_arr": [f"x"],
                "flt_arr": [0.0],
                "bool_arr": [True],
            })
        _setup(milvus_client, name, rows)

        results = milvus_client.search(
            name, data=[_unit_vec(0)], limit=15,
            filter="array_length(int_arr) == 3",
            output_fields=["int_arr"],
        )
        for h in results[0]:
            assert len(h["entity"]["int_arr"]) == 3

        milvus_client.drop_collection(name)


class TestArrayCombinedFilters:
    """Combined filters ported from multiple Milvus test classes.

    Tests array filters composed with AND/OR and scalar comparisons.
    """

    def test_array_contains_or_array_contains(self, milvus_client):
        """Two array_contains joined with OR.

        Ref: Composite of test_search_array_contains patterns.
        """
        name = COLLECTION_PREFIX + "or_contains"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        expr = "array_contains(int_arr, 0) or array_contains(int_arr, 9)"
        match_0 = {i for i in range(20)
                    if 0 in [i % 10, (i + 1) % 10, (i + 2) % 10]}
        match_9 = {i for i in range(20)
                    if 9 in [i % 10, (i + 1) % 10, (i + 2) % 10]}
        expected_ids = match_0 | match_9

        res = milvus_client.query(name, filter=expr,
                                  output_fields=["int_arr"], limit=100)
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids

        milvus_client.drop_collection(name)

    def test_array_access_and_scalar(self, milvus_client):
        """Array index access combined with primary key filter via AND.

        Ref: TestSearchArrayShared.test_search_array_access + scalar
        """
        name = COLLECTION_PREFIX + "access_and_scalar"
        rows = _make_rows()
        _setup(milvus_client, name, rows)

        # int_arr[0] < 5  =>  i%10 < 5
        # id >= 10
        expr = "int_arr[0] < 5 and id >= 10"
        expected_ids = {i for i in range(20) if i % 10 < 5 and i >= 10}

        res = milvus_client.query(name, filter=expr,
                                  output_fields=["int_arr"], limit=100)
        got_ids = {r["id"] for r in res}
        assert got_ids == expected_ids

        milvus_client.drop_collection(name)


class TestArrayOutputRoundtrip:
    """Verify that array values survive insert -> flush -> query for
    all four element types via the gRPC adapter.

    Ref: TestSearchArrayShared fixture + test_search_array_contains
    (validates output_fields return actual array data).
    """

    def test_all_element_types_roundtrip(self, milvus_client):
        """Insert arrays of INT64, VARCHAR, FLOAT, BOOL and read them back."""
        name = COLLECTION_PREFIX + "roundtrip"
        rows = [
            {"id": 0, "vec": _unit_vec(0),
             "int_arr": [10, 20, 30],
             "str_arr": ["hello", "world"],
             "flt_arr": [1.5, 2.5, 3.5],
             "bool_arr": [True, False, True]},
            {"id": 1, "vec": _unit_vec(1),
             "int_arr": [],
             "str_arr": [],
             "flt_arr": [],
             "bool_arr": []},
            {"id": 2, "vec": _unit_vec(2),
             "int_arr": [99],
             "str_arr": ["only"],
             "flt_arr": [0.0],
             "bool_arr": [False]},
        ]
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="id >= 0",
            output_fields=["int_arr", "str_arr", "flt_arr", "bool_arr"],
            limit=100,
        )
        res_by_id = {r["id"]: r for r in res}

        # Row 0: full arrays
        assert res_by_id[0]["int_arr"] == [10, 20, 30]
        assert res_by_id[0]["str_arr"] == ["hello", "world"]
        assert res_by_id[0]["bool_arr"] == [True, False, True]
        # Float comparison with tolerance
        assert len(res_by_id[0]["flt_arr"]) == 3
        assert abs(res_by_id[0]["flt_arr"][0] - 1.5) < 1e-6

        # Row 1: empty arrays
        assert res_by_id[1]["int_arr"] == []
        assert res_by_id[1]["str_arr"] == []
        assert res_by_id[1]["flt_arr"] == []
        assert res_by_id[1]["bool_arr"] == []

        # Row 2: single-element arrays
        assert res_by_id[2]["int_arr"] == [99]
        assert res_by_id[2]["str_arr"] == ["only"]
        assert res_by_id[2]["bool_arr"] == [False]

        milvus_client.drop_collection(name)


class TestArrayEmptyResults:
    """Verify filter returns empty result set when no rows match.

    Ref: test_search_with_expression_invalid_array_one (search_invalid.py)
    — subscript beyond array length yields empty results.
    """

    def test_array_contains_no_match(self, milvus_client):
        """array_contains with a value that no row has.

        Ref: Derived from TestSearchArrayShared filter validation.
        int_arr values are all in [0..9], so 99 matches nothing.
        """
        name = COLLECTION_PREFIX + "no_match"
        rows = _make_rows(10)
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="array_contains(int_arr, 99)",
            output_fields=["int_arr"], limit=100,
        )
        assert len(res) == 0

        milvus_client.drop_collection(name)

    def test_array_contains_all_no_match(self, milvus_client):
        """array_contains_all with values that no single row has together.

        int_arr has 3 consecutive mod-10 values.  [0, 5] can never
        appear together in a length-3 consecutive window.
        """
        name = COLLECTION_PREFIX + "all_no_match"
        rows = _make_rows(10)
        _setup(milvus_client, name, rows)

        res = milvus_client.query(
            name, filter="array_contains_all(int_arr, [0, 5])",
            output_fields=["int_arr"], limit=100,
        )
        assert len(res) == 0

        milvus_client.drop_collection(name)
