"""End-to-end tests for filter expression templates (issue #327).

pymilvus exposes Milvus's ``expr_template_values`` as ``filter_params`` on
search / query / delete.  These tests exercise the full adapter path:

    pymilvus -> SearchRequest.expr_template_values
             -> translator renders literals
             -> engine filter parser/evaluator
"""

import pytest
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker


def _make_schema():
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=False
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=2)
    schema.add_field("city", DataType.VARCHAR, max_length=64)
    schema.add_field("age", DataType.INT64)
    schema.add_field("score", DataType.FLOAT)
    schema.add_field("active", DataType.BOOL)
    schema.add_field("message", DataType.VARCHAR, max_length=128)
    return schema


@pytest.fixture
def template_client(milvus_client):
    """Three rows with distinct scalar values, loaded for search/query."""
    milvus_client.create_collection("tpl", schema=_make_schema())
    milvus_client.insert(
        "tpl",
        [
            {
                "id": 0,
                "vec": [0.0, 0.0],
                "city": "北京",
                "age": 30,
                "score": 0.5,
                "active": True,
                "message": "E1234 ok",
            },
            {
                "id": 1,
                "vec": [1.0, 0.0],
                "city": "上海",
                "age": 20,
                "score": 1.5,
                "active": False,
                "message": "DEBUG",
            },
            {
                "id": 2,
                "vec": [2.0, 0.0],
                "city": "O'Reilly",
                "age": 40,
                "score": 2.5,
                "active": True,
                "message": "E5678",
            },
        ],
    )

    indexes = milvus_client.prepare_index_params()
    indexes.add_index(
        field_name="vec",
        index_type="FLAT",
        metric_type="L2",
        params={},
    )
    milvus_client.create_index("tpl", indexes)
    milvus_client.load_collection("tpl")
    return milvus_client


def _ids(rows):
    return sorted(row["id"] for row in rows)


def test_query_scalar_and_array_templates(template_client):
    rows = template_client.query(
        "tpl",
        filter="age > {age} AND city IN {city}",
        output_fields=["id"],
        filter_params={"age": 25, "city": ["北京", "上海"]},
    )
    # 上海 matches the IN list but its age is 20, not > 25.
    assert _ids(rows) == [0]


def test_search_with_template_filter(template_client):
    results = template_client.search(
        "tpl",
        data=[[0.0, 0.0]],
        limit=3,
        filter="city IN {city}",
        output_fields=["id"],
        filter_params={"city": ["北京", "O'Reilly"]},
    )
    assert sorted(hit["id"] for hit in results[0]) == [0, 2]


def test_float_and_bool_templates(template_client):
    rows = template_client.query(
        "tpl",
        filter="score >= {score} AND active == {active}",
        output_fields=["id"],
        filter_params={"score": 1.5, "active": False},
    )
    assert _ids(rows) == [1]


def test_string_values_are_quoted_not_injected(template_client):
    rows = template_client.query(
        "tpl",
        filter="city == {city}",
        output_fields=["id"],
        filter_params={"city": "O'Reilly"},
    )
    assert _ids(rows) == [2]

    injected = template_client.query(
        "tpl",
        filter="city == {city}",
        output_fields=["id"],
        filter_params={"city": "' OR id >= 0 OR city == '"},
    )
    assert injected == []


def test_delete_with_template(template_client):
    template_client.delete(
        "tpl",
        filter="age < {age}",
        filter_params={"age": 25},
    )
    rows = template_client.query("tpl", filter="id >= 0", output_fields=["id"])
    assert _ids(rows) == [0, 2]


def test_get_by_pk_fast_path_with_array_template(template_client):
    rows = template_client.query(
        "tpl",
        filter="id in {ids}",
        output_fields=["id"],
        filter_params={"ids": [0, 2]},
    )
    assert _ids(rows) == [0, 2]


def test_count_star_with_template(template_client):
    rows = template_client.query(
        "tpl",
        filter="active == {active}",
        output_fields=["count(*)"],
        filter_params={"active": True},
    )
    assert rows[0]["count(*)"] == 2


def test_hybrid_search_per_route_template(template_client):
    request = AnnSearchRequest(
        data=[[0.0, 0.0]],
        anns_field="vec",
        param={},
        limit=3,
        expr="city == {city}",
        expr_params={"city": "北京"},
    )
    results = template_client.hybrid_search(
        "tpl",
        reqs=[request],
        ranker=RRFRanker(),
        limit=3,
        output_fields=["id"],
    )
    assert [hit["id"] for hit in results[0]] == [0]


def test_missing_template_value_is_an_error(template_client):
    with pytest.raises(Exception, match="missing filter template value"):
        template_client.query(
            "tpl",
            filter="age > {age}",
            output_fields=["id"],
            filter_params={},
        )
