import time

import pytest

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
default_nq = ct.default_nq
default_string_field_name = ct.default_string_field_name


class TestMilvusLocalSparse(TestcaseBase):

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("index, index_params", zip(ct.all_index_types[9:11], ct.default_index_params[9:11]))
    def test_milvus_local_sparse_search(self, index, index_params,  enable_milvus_local_api):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        params = {"index_type": index, "metric_type": "IP", "params": index_params}
        client_w.create_index(ct.default_sparse_vec_field_name, params, index_name=index)
        data = cf.gen_default_list_sparse_data()
        client_w.load()
        client_w.search(data[-1][-1:], ct.default_sparse_vec_field_name,
                            ct.default_sparse_search_params, default_limit,
                            check_task=CheckTasks.check_search_results,
                            check_items={"nq": default_nq,
                                         "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("index, index_params", zip(ct.all_index_types[9:11], ct.default_index_params[9:11]))
    @pytest.mark.parametrize("dim", [ct.min_sparse_vector_dim, 32768, ct.max_sparse_vector_dim])
    def test_milvus_lcoal_sparse_index_dim(self, index, index_params, dim, enable_milvus_local_api):
        """
        target: validating the sparse index in different dimensions
        method: create connection, collection, insert and hybrid search
        expected: search successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_sparse_schema(auto_id=False)
        client_w.create_collection(client, c_name, schema=schema, sparse=True)
        data = cf.gen_default_list_sparse_data(dim=dim)
        client_w.insert(data)
        params = {"index_type": index, "metric_type": "IP", "params": index_params}
        client_w.create_index(ct.default_sparse_vec_field_name, params, index_name=index)

        client_w.load()
        client_w.search(data[-1][-1:], ct.default_sparse_vec_field_name,
                            ct.default_sparse_search_params, default_limit,
                            check_task=CheckTasks.check_search_results,
                            check_items={"nq": default_nq,
                                         "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("index, index_params", zip(ct.all_index_types[9:11], ct.default_index_params[9:11]))
    @pytest.mark.parametrize("dim", [ct.min_sparse_vector_dim - 1, 0.5, ct.max_sparse_vector_dim + 2])
    def test_sparse_index_invalid_dim(self, index, index_params, dim, enable_milvus_local_api):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_sparse_schema(auto_id=False)
        client_w.create_collection(client, c_name, schema=schema, sparse=True)
        data = cf.gen_default_list_sparse_data(dim=dim)
        client_w.insert(data)
        params = {"index_type": index, "metric_type": "IP", "params": index_params}
        client_w.create_index(ct.default_sparse_vec_field_name, params, index_name=index)
        client_w.load()
        client_w.search(data[-1][-1:], ct.default_sparse_vec_field_name,
                            ct.default_sparse_search_params, default_limit,
                            check_task=CheckTasks.check_search_results,
                            check_items={"nq": default_nq,
                                         "limit": default_limit})