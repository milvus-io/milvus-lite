import pytest
import numpy as np
from base.client_base import TestcaseBase
from common.common_type import CaseLabel, CheckTasks
from common import common_func as cf
from common import common_type as ct
from base.high_level_api_wrapper import HighLevelApiWrapper

client_w = HighLevelApiWrapper()
prefix = "milvus_local_api_collection"
default_dim = ct.default_dim
default_primary_key_field_name = "id"
default_float_field_name = ct.default_float_field_name
default_vector_field_name = "vector"
default_limit = ct.default_limit
default_nb = ct.default_nb
default_search_exp = "id >= 0"
exp_res = "exp_res"
default_string_field_name = ct.default_string_field_name


class TestMilvusLocalCloseAPI(TestcaseBase):
    """
    Add milvus local unique case
    """
    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2", "IP"])
    def metric_type(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["TRIE", "STL_SORT", "AUTOINDEX"])
    def scalar_index(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_local_reconnect(self, enable_milvus_local_api):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        result = client_w.has_collection(client, collection_name)[0]
        assert result == True
        client.close()
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        result = client_w.has_collection(client, collection_name)[0]
        assert result == True

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("index, params",
                             zip(ct.all_index_types[:7],
                                 ct.default_index_params[:7]))
    def test_milvus_client_index_with_params(self, enable_milvus_local_api, index, params, metric_type):
        """
        target: reconnect search normal case
        method: create connection, collection, insert, close and search
        expected: search successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        client_w.release_collection(client, collection_name)
        client_w.drop_index(client, collection_name, "vector")
        res = client_w.list_indexes(client, collection_name)[0]
        assert res == []
        # 2. prepare index params
        index_params = client_w.prepare_index_params(client)[0]
        index_params.add_index(field_name="vector", index_type=index, params=params, metric_type=metric_type)
        # 3. create index
        client_w.create_index(client, collection_name, index_params)
        # 4. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 5. load collection
        client_w.load_collection(client, collection_name)
        client_w.release_collection(client, collection_name)
        # 6. close and reconnect
        client.close()
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        # 7. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        client_w.search(client, collection_name, vectors_to_search,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": default_limit})
        client_w.drop_collection(client, collection_name)


class TestMilvusLocalCloseInvalid(TestcaseBase):

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_local_close_multiple(self, enable_milvus_local_api):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        client_w.create_collection(client, collection_name, default_dim)
        client.close()
        client.close()
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        result = client_w.has_collection(client, collection_name)[0]
        assert result == True
        c_name = collection_name + "_2"
        client_w.create_collection(client, c_name, default_dim)
        result = client_w.has_collection(client, c_name)[0]
        assert result == True



