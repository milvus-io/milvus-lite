"""CRUD alignment tests adapted from upstream Milvus test suite.

Each test is a self-contained adaptation of a specific Milvus pymilvus test
method. The original test name and file are cited in the docstring. Data is
reduced to 10-20 rows with dim=4 for speed. All tests use BRUTE_FORCE index
with COSINE metric and the ``milvus_client`` fixture from conftest.py.

Adapted from:
  - milvus/tests/python_client/milvus_client/test_milvus_client_delete.py
  - milvus/tests/python_client/milvus_client/test_milvus_client_query.py
  - milvus/tests/python_client/milvus_client_v2/test_milvus_client_search_v2.py
"""

import math

import numpy as np
import pytest
from pymilvus import DataType, MilvusClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 4
NB = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng():
    return np.random.default_rng(seed=19530)


def _create_and_insert(client, name, nb=NB, *, extra_fields=True):
    """Create a collection with id/vec/float_field/varchar_field, insert nb
    rows, and return the row dicts.

    Does NOT create index or load -- caller decides.
    """
    schema = MilvusClient.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)
    if extra_fields:
        schema.add_field("float_field", DataType.FLOAT)
        schema.add_field("varchar_field", DataType.VARCHAR, max_length=64)
    client.create_collection(name, schema=schema)

    rng = _rng()
    rows = [
        {
            "id": i,
            "vec": list(rng.random(DIM).astype(float)),
            **(
                {
                    "float_field": float(i),
                    "varchar_field": str(i),
                }
                if extra_fields
                else {}
            ),
        }
        for i in range(nb)
    ]
    client.insert(name, rows)
    return rows


def _index_and_load(client, name):
    """Create a BRUTE_FORCE/COSINE index and load the collection."""
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="vec",
        index_type="BRUTE_FORCE",
        metric_type="COSINE",
        params={},
    )
    client.create_index(name, idx)
    client.load_collection(name)


# ===================================================================
# 1. Delete with filters
#    Source: test_milvus_client_delete.py  lines 182-221
#           test_milvus_client_delete_with_filters
# ===================================================================


def test_delete_with_filters(milvus_client):
    """Adapted from test_milvus_client_delete_with_filters
    (test_milvus_client_delete.py:182).

    Validates that delete(filter="id < N") removes the correct rows, and
    that subsequent search and query no longer return deleted entries.
    """
    name = "del_filter"
    rows = _create_and_insert(milvus_client, name)
    _index_and_load(milvus_client, name)

    # Delete rows with id < 3
    delete_num = 3
    milvus_client.delete(name, filter=f"id < {delete_num}")

    # Search -- deleted ids must not appear
    rng = _rng()
    query_vec = [list(rng.random(DIM).astype(float))]
    search_results = milvus_client.search(
        name,
        data=query_vec,
        limit=NB,
        search_params={"metric_type": "COSINE"},
        output_fields=["id"],
    )
    returned_ids = {hit["id"] for hit in search_results[0]}
    for deleted_id in range(delete_num):
        assert deleted_id not in returned_ids, (
            f"deleted id {deleted_id} still appears in search results"
        )
    assert len(search_results[0]) == NB - delete_num

    # Query -- verify remaining rows match
    query_rows = milvus_client.query(
        name,
        filter="id >= 0",
        output_fields=["id", "float_field", "varchar_field"],
        limit=NB,
    )
    query_ids = {r["id"] for r in query_rows}
    assert query_ids == set(range(delete_num, NB))

    milvus_client.drop_collection(name)


# ===================================================================
# 2. Delete with filters on nullable vector field
#    Source: test_milvus_client_delete.py  lines 224-275
#           test_milvus_client_delete_with_filters_nullable_vector_field
# ===================================================================


def test_delete_with_filters_nullable_vector_field(milvus_client):
    """Adapted from test_milvus_client_delete_with_filters_nullable_vector_field
    (test_milvus_client_delete.py:224).

    Validates delete-by-id-filter and delete-by-float-filter on a collection
    where the vector field is nullable (even rows have vectors, odd rows None).
    Verifies surviving row count via count(*) after each round of deletes.

    NOTE: MilvusLite may not yet support nullable vector fields. If the
    collection creation fails with an unsupported-feature error the test is
    skipped rather than failing.
    """
    name = "del_nullable_vec"
    nb = 20

    schema = MilvusClient.create_schema(enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM, nullable=True)
    schema.add_field("float_field", DataType.FLOAT)

    try:
        milvus_client.create_collection(name, schema=schema)
    except Exception as exc:
        pytest.skip(f"nullable vector field not supported: {exc}")

    rng = _rng()
    rows = [
        {
            "id": i,
            "vec": list(rng.random(DIM).astype(float)) if i % 2 == 0 else None,
            "float_field": float(i),
        }
        for i in range(nb)
    ]
    milvus_client.insert(name, rows)

    # Delete a few specific ids
    ids_to_delete = [0, 3, 7, 11, 15]
    milvus_client.delete(
        name, filter=f"id in {ids_to_delete}"
    )

    _index_and_load(milvus_client, name)

    # count(*) should reflect deleted ids
    result = milvus_client.query(name, filter="", output_fields=["count(*)"])
    assert result[0]["count(*)"] == nb - len(ids_to_delete)

    # Second round: delete by float filter (float_field < 10)
    milvus_client.delete(name, filter="float_field < 10")

    # Remaining: ids with float_field >= 10 that were not already deleted
    surviving = {i for i in range(nb) if i not in ids_to_delete and i >= 10}
    result2 = milvus_client.query(name, filter="", output_fields=["count(*)"])
    assert result2[0]["count(*)"] == len(surviving)

    milvus_client.drop_collection(name)


# ===================================================================
# 3. Count with expr
#    Source: test_milvus_client_query.py  lines 4957-4989
#           test_milvus_client_count_with_expr
# ===================================================================


def test_count_with_expr(milvus_client):
    """Adapted from test_milvus_client_count_with_expr
    (test_milvus_client_query.py:4957).

    Validates that query(output_fields=['count(*)']) returns the correct
    total count with a broad filter, and that a narrower 'in' filter returns
    the correct subset count.
    """
    name = "cnt_expr"
    _create_and_insert(milvus_client, name)
    _index_and_load(milvus_client, name)

    # count with id >= 0 (all rows)
    result = milvus_client.query(
        name, filter="id >= 0", output_fields=["count(*)"]
    )
    assert result[0]["count(*)"] == NB

    # count with term expression (should return exactly 2)
    result2 = milvus_client.query(
        name, filter="id in [0, 1]", output_fields=["count(*)"]
    )
    assert result2[0]["count(*)"] == 2

    milvus_client.drop_collection(name)


# ===================================================================
# 4. Get with output_fields
#    Source: test_milvus_client_query.py  lines 4100-4126
#           test_milvus_client_get_output_fields
# ===================================================================


def test_get_output_fields(milvus_client):
    """Adapted from test_milvus_client_get_output_fields
    (test_milvus_client_query.py:4100).

    Validates that get(ids, output_fields=[...]) returns only the
    requested fields (plus the primary key), and that get with a single id
    returns the same result as get with a list containing that id.
    """
    name = "get_of"
    _create_and_insert(milvus_client, name)
    _index_and_load(milvus_client, name)

    output_fields = ["id", "vec", "float_field", "varchar_field"]

    # get by list of ids
    rows_list = milvus_client.get(name, ids=[0], output_fields=output_fields)
    assert len(rows_list) == 1
    assert len(rows_list[0]) == len(output_fields)

    # get by scalar id -- should return identical data
    rows_scalar = milvus_client.get(name, ids=0, output_fields=output_fields)
    assert len(rows_scalar) == 1
    assert dict(rows_list[0]) == dict(rows_scalar[0])

    # verify each requested field is present
    for field in output_fields:
        assert field in rows_list[0]

    milvus_client.drop_collection(name)


# ===================================================================
# 5. Query output_fields=["*"] (all fields)
#    Source: test_milvus_client_query.py  lines 1032-1091
#           test_milvus_client_query_output_fields_all
# ===================================================================


def test_query_output_fields_all(milvus_client):
    """Adapted from test_milvus_client_query_output_fields_all
    (test_milvus_client_query.py:1032).

    Validates that query(output_fields=['*']) returns all schema-defined
    fields (id, vec, float_field, varchar_field) for each result row.
    Tests both query-by-ids and query-by-filter code paths.
    """
    name = "qry_all_fields"
    nb = 10
    rows = _create_and_insert(milvus_client, name, nb=nb)
    _index_and_load(milvus_client, name)

    all_fields = {"id", "vec", "float_field", "varchar_field"}

    # query by ids with wildcard output
    result_by_ids = milvus_client.query(
        name,
        ids=[i for i in range(nb)],
        output_fields=["*"],
    )
    assert len(result_by_ids) == nb
    assert set(result_by_ids[0].keys()) == all_fields

    # query by filter with wildcard output
    result_by_filter = milvus_client.query(
        name,
        filter="id >= 0",
        output_fields=["*"],
        limit=nb,
    )
    assert len(result_by_filter) == nb
    assert set(result_by_filter[0].keys()) == all_fields

    milvus_client.drop_collection(name)


# ===================================================================
# 6. Search round_decimal
#    Source: test_milvus_client_search_v2.py  lines 291-329
#           test_search_round_decimal
# ===================================================================


@pytest.mark.parametrize("round_decimal", [0, 1, 2, 3, 4, 5, 6])
def test_search_round_decimal(milvus_client, round_decimal):
    """Adapted from test_search_round_decimal
    (test_milvus_client_search_v2.py:291).

    Validates that search(round_decimal=N) rounds distance values to N
    decimal places. Compares distances from a rounded search against an
    unrounded search and verifies they match within tolerance.
    """
    name = f"rd_{round_decimal}"
    nb = 10
    _create_and_insert(milvus_client, name, nb=nb)
    _index_and_load(milvus_client, name)

    limit = 5
    rng = _rng()
    query_vec = [list(rng.random(DIM).astype(float))]

    # Unrounded reference search
    res_raw = milvus_client.search(
        name,
        data=query_vec,
        limit=limit,
        search_params={"metric_type": "COSINE"},
        output_fields=["id"],
    )

    # Rounded search
    res_round = milvus_client.search(
        name,
        data=query_vec,
        limit=limit,
        search_params={"metric_type": "COSINE"},
        round_decimal=round_decimal,
        output_fields=["id"],
    )

    # Build id->distance map from raw results
    dist_map = {hit["id"]: hit["distance"] for hit in res_raw[0]}

    abs_tol = pow(10, 1 - round_decimal)
    matched_count = 0
    for hit in res_round[0]:
        pk = hit["id"]
        if pk in dist_map:
            expected = round(dist_map[pk], round_decimal)
            actual = hit["distance"]
            assert math.isclose(actual, expected, rel_tol=0, abs_tol=abs_tol), (
                f"round_decimal={round_decimal}: id={pk} "
                f"expected={expected}, actual={actual}"
            )
            matched_count += 1

    assert matched_count > 0, (
        "no matching PKs found between rounded and unrounded results"
    )

    milvus_client.drop_collection(name)
