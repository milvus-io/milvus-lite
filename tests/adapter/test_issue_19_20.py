"""Tests for issue #19 (dynamic field int/float values) and #20 (jieba analyzer key)."""

import pytest
from pymilvus import DataType, MilvusClient


DIM = 16


class TestDynamicFieldTypes:
    """Issue #19: dynamic field int/float values must be preserved."""

    def test_dynamic_int_value_returned(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_int", schema=schema, index_params=idx)
        milvus_client.insert("dyn_int", [
            {"id": 1, "vec": [0.1] * DIM, "priority": 42},
        ])
        milvus_client.load_collection("dyn_int")
        res = milvus_client.query("dyn_int", filter="id == 1",
                                  output_fields=["priority"])
        assert res[0]["priority"] == 42
        assert isinstance(res[0]["priority"], int)

    def test_dynamic_float_value_returned(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_float", schema=schema, index_params=idx)
        milvus_client.insert("dyn_float", [
            {"id": 1, "vec": [0.1] * DIM, "score": 3.14},
        ])
        milvus_client.load_collection("dyn_float")
        res = milvus_client.query("dyn_float", filter="id == 1",
                                  output_fields=["score"])
        assert abs(res[0]["score"] - 3.14) < 0.01
        assert isinstance(res[0]["score"], float)

    def test_dynamic_bool_value_returned(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_bool", schema=schema, index_params=idx)
        milvus_client.insert("dyn_bool", [
            {"id": 1, "vec": [0.1] * DIM, "active": True},
        ])
        milvus_client.load_collection("dyn_bool")
        res = milvus_client.query("dyn_bool", filter="id == 1",
                                  output_fields=["active"])
        assert res[0]["active"] is True

    def test_dynamic_mixed_types(self, milvus_client):
        """All dynamic field types in one record."""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_mix", schema=schema, index_params=idx)
        milvus_client.insert("dyn_mix", [
            {"id": 1, "vec": [0.1] * DIM,
             "tag": "hello", "count": 99, "ratio": 0.5, "flag": False},
        ])
        milvus_client.load_collection("dyn_mix")
        res = milvus_client.query("dyn_mix", filter="id == 1",
                                  output_fields=["tag", "count", "ratio", "flag"])
        assert res[0]["tag"] == "hello"
        assert res[0]["count"] == 99
        assert abs(res[0]["ratio"] - 0.5) < 0.01
        assert res[0]["flag"] is False

    def test_dynamic_list_value_returned(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_list", schema=schema, index_params=idx)
        milvus_client.insert("dyn_list", [
            {"id": 1, "vec": [0.1] * DIM, "tags": [1, 2, 3]},
        ])
        milvus_client.load_collection("dyn_list")
        res = milvus_client.query("dyn_list", filter="id == 1",
                                  output_fields=["tags"])
        assert res[0]["tags"] == [1, 2, 3]
        assert isinstance(res[0]["tags"], list)

    def test_dynamic_dict_value_returned(self, milvus_client):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIM)

        idx = milvus_client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="FLAT", metric_type="COSINE")
        milvus_client.create_collection("dyn_dict", schema=schema, index_params=idx)
        milvus_client.insert("dyn_dict", [
            {"id": 1, "vec": [0.1] * DIM, "info": {"nested": "val", "n": 42}},
        ])
        milvus_client.load_collection("dyn_dict")
        res = milvus_client.query("dyn_dict", filter="id == 1",
                                  output_fields=["info"])
        assert res[0]["info"] == {"nested": "val", "n": 42}
        assert isinstance(res[0]["info"], dict)


class TestJiebaAnalyzerKey:
    """Issue #20: analyzer_params with 'type' key should work like 'tokenizer'."""

    def test_jieba_via_type_key(self):
        """create_analyzer({"type": "jieba"}) should return JiebaAnalyzer."""
        pytest.importorskip("jieba")
        from milvus_lite.analyzer.factory import create_analyzer
        a = create_analyzer({"type": "jieba"})
        assert type(a).__name__ == "JiebaAnalyzer"

    def test_jieba_via_tokenizer_key_still_works(self):
        """create_analyzer({"tokenizer": "jieba"}) should still work."""
        pytest.importorskip("jieba")
        from milvus_lite.analyzer.factory import create_analyzer
        a = create_analyzer({"tokenizer": "jieba"})
        assert type(a).__name__ == "JiebaAnalyzer"

    def test_standard_via_type_key(self):
        from milvus_lite.analyzer.factory import create_analyzer
        a = create_analyzer({"type": "standard"})
        assert type(a).__name__ == "StandardAnalyzer"

    def test_default_without_key(self):
        from milvus_lite.analyzer.factory import create_analyzer
        a = create_analyzer({})
        assert type(a).__name__ == "StandardAnalyzer"
