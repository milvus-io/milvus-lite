import math
import multiprocessing
import numbers
import random
import numpy
import threading

import numpy as np
import pytest
import pandas as pd
import decimal
from decimal import Decimal, getcontext
from time import sleep
import heapq

from base.client_base import TestcaseBase
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.util_pymilvus import *
from common.constants import *
from pymilvus.orm.types import CONSISTENCY_STRONG, CONSISTENCY_BOUNDED, CONSISTENCY_SESSION, CONSISTENCY_EVENTUALLY
from base.high_level_api_wrapper import HighLevelApiWrapper
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker


prefix = "milvus_client_api_search"
epsilon = ct.epsilon
default_nb = ct.default_nb
default_nb_medium = ct.default_nb_medium
default_nq = ct.default_nq
default_dim = ct.default_dim
default_limit = ct.default_limit
default_search_exp = "id >= 0"
default_expr = "int64 >= 0"
exp_res = "exp_res"
default_search_string_exp = "varchar >= \"0\""
default_search_mix_exp = "int64 >= 0 && varchar >= \"0\""
default_invaild_string_exp = "varchar >= 0"
default_json_search_exp = "json_field[\"number\"] >= 0"
perfix_expr = 'varchar like "0%"'
default_search_field = ct.default_float_vec_field_name
default_search_params = ct.default_search_params
default_primary_key_field_name = "id"
default_vector_field_name = "vector"
default_float_field_name = ct.default_float_field_name
default_bool_field_name = ct.default_bool_field_name
default_string_field_name = ct.default_string_field_name
default_int32_array_field_name = ct.default_int32_array_field_name
default_string_array_field_name = ct.default_string_array_field_name
default_int64_field_name = ct.default_int64_field_name
max_hybrid_search_req_num = ct.max_hybrid_search_req_num
default_json_field_name = ct.default_json_field_name
hybrid_search_epsilon = 0.01
nq = 1
min_dim = ct.min_dim
max_limit = ct.max_limit
client_w = HighLevelApiWrapper()


class TestMilvusClientSearchInvalid(TestcaseBase):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_invalid_primary_field(self, enable_milvus_local_api):
        """
        target: test high level api: client.create_collection
        method: create collection with invalid primary field
        expected: Raise exception
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 1, ct.err_msg: f"Param id_type must be int or string"}
        client_w.create_collection(client, collection_name, default_dim, id_type="invalid",
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_string_auto_id(self, enable_milvus_local_api):
        """
        target: test high level api: client.create_collection
        method: create collection with auto id on string primary key
        expected: Raise exception
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 65535, ct.err_msg: f"type param(max_length) should be specified for varChar "
                                                 f"field of collection {collection_name}"}
        client_w.create_collection(client, collection_name, default_dim, id_type="string", auto_id=True,
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_create_same_collection_different_params(self, enable_milvus_local_api):
        """
        target: test high level api: client.create_collection
        method: create
        expected: 1. Successfully to create collection with same params
                  2. Report errors for creating collection with same name and different params
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. create collection with same params
        client_w.create_collection(client, collection_name, default_dim)
        # 3. create collection with same name and different params
        error = {ct.err_code: 1, ct.err_msg: f"create duplicate collection with different parameters, "
                                             f"collection: {collection_name}"}
        client_w.create_collection(client, collection_name, default_dim + 1,
                                   check_task=CheckTasks.err_res, check_items=error)
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_invalid_metric_type(self, enable_milvus_local_api):
        """
        target: test high level api: client.create_collection
        method: create collection with auto id on string primary key
        expected: Raise exception
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 1100,
                 ct.err_msg: "metric type not found or not supported, supported: [L2 IP COSINE HAMMING JACCARD]"}
        client_w.create_collection(client, collection_name, default_dim, metric_type="invalid",
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip("https://github.com/milvus-io/milvus/issues/29880")
    def test_milvus_client_search_not_consistent_metric_type(self, enable_milvus_local_api, metric_type):
        """
        target: test search with inconsistent metric type (default is IP) with that of index
        method: create connection, collection, insert and search with not consistent metric type
        expected: Raise exception
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. search
        rng = np.random.default_rng(seed=19530)
        vectors_to_search = rng.random((1, 8))
        search_params = {"metric_type": metric_type}
        error = {ct.err_code: 1100,
                 ct.err_msg: f"metric type not match: invalid parameter[expected=IP][actual={metric_type}]"}
        client_w.search(client, collection_name, vectors_to_search, limit=default_limit,
                        search_params=search_params,
                        check_task=CheckTasks.err_res, check_items=error)
        client_w.drop_collection(client, collection_name)


class TestMilvusClientSearchValid(TestcaseBase):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_search_query_default(self, enable_milvus_local_api):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        client_w.using_database(client, "default")
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        collections = client_w.list_collections(client)[0]
        assert collection_name in collections
        client_w.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "consistency_level": 0})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # client_w.flush(client, collection_name)
        # assert client_w.num_entities(client, collection_name)[0] == default_nb
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        client_w.search(client, collection_name, vectors_to_search,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": default_limit})
        # 4. query
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.release_collection(client, collection_name)
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_rename_search_query_default(self, enable_milvus_local_api):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        collections = client_w.list_collections(client)[0]
        assert collection_name in collections
        client_w.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "consistency_level": 0})
        old_name = collection_name
        new_name = collection_name + "new"
        client_w.rename_collection(client, old_name, new_name)
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, new_name, rows)
        # client_w.flush(client, collection_name)
        # assert client_w.num_entities(client, collection_name)[0] == default_nb
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        client_w.search(client, new_name, vectors_to_search,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": default_limit})
        # 4. query
        client_w.query(client, new_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.release_collection(client, new_name)
        client_w.drop_collection(client, new_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_array_insert_search(self, enable_milvus_local_api):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        collections = client_w.list_collections(client)[0]
        assert collection_name in collections
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{
            default_primary_key_field_name: i,
            default_vector_field_name: list(rng.random((1, default_dim))[0]),
            default_float_field_name: i * 1.0,
            default_int32_array_field_name: [i, i + 1, i + 2],
            default_string_array_field_name: [str(i), str(i + 1), str(i + 2)]
        } for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        client_w.search(client, collection_name, vectors_to_search,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_search_query_string(self, enable_milvus_local_api):
        """
        target: test search (high level api) for string primary key
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, id_type="string", max_length=ct.default_length)
        client_w.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "auto_id": auto_id})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [
            {default_primary_key_field_name: str(i), default_vector_field_name: list(rng.random((1, default_dim))[0]),
             default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        client_w.flush(client, collection_name)
        assert client_w.num_entities(client, collection_name)[0] == default_nb
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        client_w.search(client, collection_name, vectors_to_search,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "limit": default_limit})
        # 4. query
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows,
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_search_different_metric_types_not_specifying_in_search_params(self, enable_milvus_local_api,
                                                                                         metric_type, auto_id):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search successfully with limit(topK)
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, metric_type=metric_type, auto_id=auto_id,
                                   consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        if auto_id:
            for row in rows:
                row.pop(default_primary_key_field_name)
        client_w.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        # search_params = {"metric_type": metric_type}
        client_w.search(client, collection_name, vectors_to_search, limit=default_limit,
                        output_fields=[default_primary_key_field_name],
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "limit": default_limit})
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_search_different_metric_types_specifying_in_search_params(self, enable_milvus_local_api,
                                                                                     metric_type, auto_id):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search successfully with limit(topK)
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, metric_type=metric_type, auto_id=auto_id,
                                   consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        if auto_id:
            for row in rows:
                row.pop(default_primary_key_field_name)
        client_w.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        search_params = {"metric_type": metric_type}
        client_w.search(client, collection_name, vectors_to_search, limit=default_limit,
                        search_params=search_params,
                        output_fields=[default_primary_key_field_name],
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "limit": default_limit})
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_delete_with_ids(self, enable_milvus_local_api):
        """
        target: test delete (high level api)
        method: create connection, collection, insert delete, and search
        expected: search/query successfully without deleted data
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        default_nb = 1000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        pks = client_w.insert(client, collection_name, rows)[0]
        # 3. delete
        delete_num = 3
        client_w.delete(client, collection_name, ids=[i for i in range(delete_num)])
        # 4. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        for insert_id in range(delete_num):
            if insert_id in insert_ids:
                insert_ids.remove(insert_id)
        limit = default_nb - delete_num
        client_w.search(client, collection_name, vectors_to_search, limit=default_nb,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": limit})
        # 5. query
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows[delete_num:],
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_delete_with_filters(self, enable_milvus_local_api):
        """
        target: test delete (high level api)
        method: create connection, collection, insert delete, and search
        expected: search/query successfully without deleted data
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        default_nb = 1000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        pks = client_w.insert(client, collection_name, rows)[0]
        # 3. delete
        delete_num = 3
        client_w.delete(client, collection_name, filter=f"id < {delete_num}")
        # 4. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        for insert_id in range(delete_num):
            if insert_id in insert_ids:
                insert_ids.remove(insert_id)
        limit = default_nb - delete_num
        client_w.search(client, collection_name, vectors_to_search, limit=default_nb,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": limit})
        # 5. query
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows[delete_num:],
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.drop_collection(client, collection_name)


class TestCollectionHybridSearchValid(TestcaseBase):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[1, 10])
    def nq(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[default_nb_medium])
    def nb(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[32, 128])
    def dim(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[False, True])
    def _async(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["JACCARD", "HAMMING"])
    def metrics(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[False, True])
    def is_flush(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[True, False])
    def enable_dynamic_field(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["IP", "COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[True, False])
    def random_primary_key(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"])
    def vector_data_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases for hybrid_search
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("nq", [16384])
    def test_hybrid_search_normal_max_nq(self, nq, enable_milvus_local_api):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = self.init_collection_general(prefix, True)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [1]
        vectors = cf.gen_vectors_based_on_vector_type(nq, default_dim, "FLOAT_VECTOR")
        # 4. get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 5. hybrid search
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": nq,
                                                             "ids": insert_ids,
                                                             "limit": default_limit})[0]

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.skip(reason="issue 32288")
    @pytest.mark.parametrize("nq", [0, 16385])
    def test_hybrid_search_normal_over_max_nq(self, nq):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w = self.init_collection_general(prefix, True)[0]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [1]
        vectors = cf.gen_vectors_based_on_vector_type(nq, default_dim, "FLOAT_VECTOR")
        # 4. get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 5. hybrid search
        err_msg = "nq (number of search vector per search request) should be in range [1, 16384]"
        collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                   check_task=CheckTasks.err_res,
                                   check_items={"err_code": 65535,
                                                "err_msg": err_msg})

    @pytest.mark.tags(CaseLabel.L1)
    def test_hybrid_search_no_limit(self):
        """
        target: test hybrid search with no limit
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        multiple_dim_array = [default_dim, default_dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        vectors = cf.gen_vectors_based_on_vector_type(nq, default_dim, "FLOAT_VECTOR")

        # get hybrid search req list
        search_param = {
            "data": vectors,
            "anns_field": vector_name_list[0],
            "param": {"metric_type": "COSINE"},
            "limit": default_limit,
            "expr": "int64 > 0"}
        req = AnnSearchRequest(**search_param)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_WeightedRanker_empty_reqs(self, primary_field):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, primary_field=primary_field,
                                         multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. hybrid search with empty reqs
        collection_w.hybrid_search([], WeightedRanker(), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": 0})

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.skip(reason="issue 29839")
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_as_search(self, nq, primary_field, is_flush):
        """
        target: test hybrid search to search as the original search interface
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK), and the result should be equal to search
        """
        # 1. initialize collection with data
        dim = 3
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_flush=is_flush,
                                         primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]

        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
        for search_field in vector_name_list:
            # 2. prepare search params
            req_list = []
            search_param = {
                "data": vectors,
                "anns_field": search_field,
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            # 3. hybrid search
            hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(1), default_limit,
                                                    check_task=CheckTasks.check_search_results,
                                                    check_items={"nq": nq,
                                                                 "ids": insert_ids,
                                                                 "limit": default_limit})[0]
            search_res = collection_w.search(vectors[:nq], search_field,
                                             default_search_params, default_limit,
                                             default_search_exp,
                                             check_task=CheckTasks.check_search_results,
                                             check_items={"nq": nq,
                                                          "ids": insert_ids,
                                                          "limit": default_limit})[0]
            # 4. the effect of hybrid search to one field should equal to search
            log.info("The distance list is:\n")
            for i in range(nq):
                log.info(hybrid_res[0].distances)
                log.info(search_res[0].distances)
                assert hybrid_res[i].ids == search_res[i].ids

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_different_metric_type(self, nq, primary_field, is_flush, metric_type):
        """
        target: test hybrid search for fields with different metric type
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 128
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_flush=is_flush, is_index=False,
                                         primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for vector_name in vector_name_list:
            search_param = {
                "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
                "anns_field": vector_name,
                "param": {"metric_type": metric_type, "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9, 1), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_different_metric_type_each_field(self, nq, primary_field, is_flush, metric_type):
        """
        target: test hybrid search for fields with different metric type
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 91
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_flush=is_flush, is_index=False,
                                         primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "L2"}
        collection_w.create_index(vector_name_list[0], flat_index)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "IP"}
        collection_w.create_index(vector_name_list[1], flat_index)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "COSINE"}
        collection_w.create_index(vector_name_list[2], flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        search_param = {
            "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
            "anns_field": vector_name_list[0],
            "param": {"metric_type": "L2", "offset": 0},
            "limit": default_limit,
            "expr": "int64 > 0"}
        req = AnnSearchRequest(**search_param)
        req_list.append(req)
        search_param = {
            "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
            "anns_field": vector_name_list[1],
            "param": {"metric_type": "IP", "offset": 0},
            "limit": default_limit,
            "expr": "int64 > 0"}
        req = AnnSearchRequest(**search_param)
        req_list.append(req)
        search_param = {
            "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
            "anns_field": vector_name_list[2],
            "param": {"metric_type": "COSINE", "offset": 0},
            "limit": default_limit,
            "expr": "int64 > 0"}
        req = AnnSearchRequest(**search_param)
        req_list.append(req)
        # 4. hybrid search
        hybrid_search = collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9, 1), default_limit,
                                                   check_task=CheckTasks.check_search_results,
                                                   check_items={"nq": nq,
                                                                "ids": insert_ids,
                                                                "limit": default_limit})[0]

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_different_dim(self, nq, primary_field, metric_type):
        """
        target: test hybrid search for fields with different dim
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        default_limit = 100
        # 1. initialize collection with data
        dim = 121
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(nq)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type, "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        hybrid_search_0 = collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                                     check_task=CheckTasks.check_search_results,
                                                     check_items={"nq": nq,
                                                                  "ids": insert_ids,
                                                                  "limit": default_limit})[0]
        hybrid_search_1 = collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                                     check_task=CheckTasks.check_search_results,
                                                     check_items={"nq": nq,
                                                                  "ids": insert_ids,
                                                                  "limit": default_limit})[0]
        for i in range(nq):
            assert hybrid_search_0[i].ids == hybrid_search_1[i].ids
            assert hybrid_search_0[i].distances == hybrid_search_1[i].distances

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_overall_limit_larger_sum_each_limit(self, nq, primary_field, metric_type):
        """
        target: test hybrid search: overall limit which is larger than sum of each limit
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 200
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        id_list_nq = []
        vectors = []
        default_search_params = {"metric_type": metric_type, "offset": 0}
        for i in range(len(vector_name_list)):
            vectors.append([])
        for i in range(nq):
            id_list_nq.append([])
        for k in range(nq):
            for i in range(len(vector_name_list)):
                vectors_search = [random.random() for _ in range(multiple_dim_array[i])]
                vectors[i].append(vectors_search)
        default_search_exp = "int64 >= 0"
        # 4. search for the comparision for hybrid search
        for i in range(len(vector_name_list)):
            search_res = collection_w.search(vectors[i], vector_name_list[i],
                                             default_search_params, default_limit,
                                             default_search_exp,
                                             check_task=CheckTasks.check_search_results,
                                             check_items={"nq": nq,
                                                          "ids": insert_ids,
                                                          "limit": default_limit})[0]
            for k in range(nq):
                id_list_nq[k].extend(search_res[k].ids)
        # 5. prepare hybrid search params
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors[i],
                "anns_field": vector_name_list[i],
                "param": default_search_params,
                "limit": default_limit,
                "expr": default_search_exp}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 6. hybrid search
        hybrid_search = \
            collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit * len(req_list) + 1)[0]
        assert len(hybrid_search) == nq
        for i in range(nq):
            assert len(hybrid_search[i].ids) == len(list(set(id_list_nq[i])))
            assert set(hybrid_search[i].ids) == set(id_list_nq[i])

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_overall_different_limit(self, primary_field, metric_type):
        """
        target: test hybrid search with different limit params
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 100
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(nq)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type, "offset": 0},
                "limit": default_limit - i,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_min_limit(self, primary_field, metric_type):
        """
        target: test hybrid search with minimum limit params
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 99
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        id_list = []
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(1)]
            search_params = {"metric_type": metric_type, "offset": 0}
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": search_params,
                "limit": min_dim,
                "expr": default_expr}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            search_res = collection_w.search(vectors[:1], vector_name_list[i],
                                             search_params, min_dim,
                                             default_expr,
                                             check_task=CheckTasks.check_search_results,
                                             check_items={"nq": 1,
                                                          "ids": insert_ids,
                                                          "limit": min_dim})[0]
            id_list.extend(search_res[0].ids)
        # 4. hybrid search
        hybrid_search = collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit)[0]
        assert len(hybrid_search) == 1
        assert len(hybrid_search[0].ids) == len(list(set(id_list)))

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_max_limit(self, primary_field, metric_type):
        """
        target: test hybrid search with maximum limit params
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 66
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(nq)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type},
                "limit": 16384,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_max_min_limit(self, primary_field, metric_type):
        """
        target: test hybrid search with maximum and minimum limit params
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 66
        multiple_dim_array = [dim + dim, dim - 10]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            limit = max_limit
            if i == 1:
                limit = 1
            search_param = {
                "data": [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(nq)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type},
                "limit": limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_same_anns_field(self, primary_field, metric_type):
        """
        target: test hybrid search: multiple search on same anns field
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 55
        multiple_dim_array = [dim, dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(multiple_dim_array[i])] for _ in range(nq)],
                "anns_field": vector_name_list[0],
                "param": {"metric_type": metric_type, "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_different_offset_single_field(self, primary_field, is_flush, metric_type):
        """
        target: test hybrid search for fields with different offset
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 100
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, auto_id=False, dim=dim, is_flush=is_flush, is_index=False,
                                         primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(dim)] for _ in range(nq)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type, "offset": i},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.1, 0.9, 1), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": nq,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_max_reqs_num(self, primary_field):
        """
        target: test hybrid search with maximum reqs number
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 128
        multiple_dim_array = [dim, dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=dim, is_index=False, primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "COSINE"}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        reqs_max_num = max_hybrid_search_req_num
        # 3. prepare search params
        req_list = []
        for i in range(reqs_max_num):
            search_param = {
                "data": [[random.random() for _ in range(dim)] for _ in range(1)],
                "anns_field": default_search_field,
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        weights = [random.random() for _ in range(len(req_list))]
        log.info(weights)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": 1,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_WeightedRanker_different_parameters(self, primary_field, is_flush, metric_type):
        """
        target: test hybrid search for fields with different offset
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        dim = 63
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, auto_id=True, dim=dim, is_flush=is_flush, is_index=False,
                                         primary_field=primary_field,
                                         enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": metric_type}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.load()
        # 3. prepare search params
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": [[random.random() for _ in range(dim)] for _ in range(1)],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": metric_type, "offset": i},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(0.2, 0.03, 0.9), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": 1,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L2)
    def test_hybrid_search_invalid_WeightedRanker_params(self):
        """
        target: test hybrid search with invalid params type to WeightedRanker
        method: create connection, collection, insert and search
        expected: raise exception
        """
        # 1. initialize collection with data
        multiple_dim_array = [default_dim, default_dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=default_dim, is_index=False,
                                         multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "COSINE"}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        reqs_num = 2
        # 3. prepare search params
        req_list = []
        for i in range(reqs_num):
            search_param = {
                "data": [[random.random() for _ in range(default_dim)] for _ in range(1)],
                "anns_field": default_search_field,
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search with list in WeightedRanker
        error = {"err_code": 1100, "err_msg": "The type of rank param weight should be float: invalid parameter"}
        collection_w.hybrid_search(req_list, WeightedRanker([0.9, 0.1]), default_limit,
                                   check_task=CheckTasks.err_res, check_items=error)

    # @pytest.mark.tags(CaseLabel.L2)
    # def test_hybrid_search_over_maximum_reqs_num(self):
    #     """
    #     target: test hybrid search over maximum reqs number
    #     method: create connection, collection, insert and search
    #     expected: raise exception
    #     """
    #     # 1. initialize collection with data
    #     multiple_dim_array = [default_dim, default_dim]
    #     collection_w, _, _, insert_ids, time_stamp = \
    #         self.init_collection_general(prefix, True, dim=default_dim, is_index=False,
    #                                      multiple_dim_array=multiple_dim_array)[0:5]
    #     # 2. extract vector field name
    #     vector_name_list = cf.extract_vector_field_name_list(collection_w)
    #     flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "COSINE"}
    #     for vector_name in vector_name_list:
    #         collection_w.create_index(vector_name, flat_index)
    #     collection_w.create_index(ct.default_float_vec_field_name, flat_index)
    #     collection_w.load()
    #     reqs_max_num = max_hybrid_search_req_num + 1
    #     # 3. prepare search params
    #     req_list = []
    #     for i in range(reqs_max_num):
    #         search_param = {
    #             "data": [[random.random() for _ in range(default_dim)] for _ in range(1)],
    #             "anns_field": default_search_field,
    #             "param": {"metric_type": "COSINE"},
    #             "limit": default_limit,
    #             "expr": "int64 > 0"}
    #         req = AnnSearchRequest(**search_param)
    #         req_list.append(req)
    #     weights = [random.random() for _ in range(len(req_list))]
    #     log.info(weights)
    #     # 4. hybrid search
    #     collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,)
    #                                # check_task=CheckTasks.err_res,
    #                                # check_items={"err_code": 65535,
    #                                #              "err_msg": 'maximum of ann search requests is 1024'})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_with_range_search(self, primary_field):
        """
        target: test hybrid search with range search
        method: create connection, collection, insert and search
        expected: raise exception (not support yet)
        """
        # 1. initialize collection with data
        multiple_dim_array = [default_dim, default_dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=default_dim, is_index=False,
                                         primary_field=primary_field,
                                         multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        flat_index = {"index_type": "FLAT", "params": {}, "metric_type": "COSINE"}
        for vector_name in vector_name_list:
            collection_w.create_index(vector_name, flat_index)
        collection_w.create_index(ct.default_float_vec_field_name, flat_index)
        collection_w.load()
        reqs_max_num = 2
        # 3. prepare search params
        req_list = []
        for i in range(reqs_max_num):
            search_param = {
                "data": [[random.random() for _ in range(default_dim)] for _ in range(1)],
                "anns_field": default_search_field,
                "param": {"metric_type": "COSINE", "params": {"radius": 0, "range_filter": 1000}},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        weights = [random.random() for _ in range(len(req_list))]
        log.info(weights)
        # 4. hybrid search
        collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": 1,
                                                "ids": insert_ids,
                                                "limit": default_limit})

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_RRFRanker_default_parameter(self, primary_field):
        """
        target: test hybrid search with default value to RRFRanker
        method: create connection, collection, insert and search.
                Note: here the result check is through comparing the score, the ids could not be compared
                because the high probability of the same score, then the id is not fixed in the range of
                the same score
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=default_dim, primary_field=primary_field,
                                         multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params for each vector field
        req_list = []
        search_res_dict_array = []
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            search_res_dict = {}
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            # search for get the base line of hybrid_search
            search_res = collection_w.search(vectors[:1], vector_name_list[i],
                                             default_search_params, default_limit,
                                             default_expr,
                                             check_task=CheckTasks.check_search_results,
                                             check_items={"nq": 1,
                                                          "ids": insert_ids,
                                                          "limit": default_limit})[0]
            ids = search_res[0].ids
            for j in range(len(ids)):
                search_res_dict[ids[j]] = 1 / (j + 60 + 1)
            search_res_dict_array.append(search_res_dict)
        # 4. calculate hybrid search base line for RRFRanker
        ids_answer, score_answer = cf.get_hybrid_search_base_results_rrf(search_res_dict_array)
        # 5. hybrid search
        hybrid_search_0 = collection_w.hybrid_search(req_list, RRFRanker(), default_limit,
                                                     check_task=CheckTasks.check_search_results,
                                                     check_items={"nq": 1,
                                                                  "ids": insert_ids,
                                                                  "limit": default_limit})[0]
        # 6. compare results through the re-calculated distances
        for i in range(len(score_answer[:default_limit])):
            assert score_answer[i] - hybrid_search_0[0].distances[i] < hybrid_search_epsilon
        # 7. run hybrid search with the same parameters twice, and compare the results
        hybrid_search_1 = collection_w.hybrid_search(req_list, RRFRanker(), default_limit,
                                                     check_task=CheckTasks.check_search_results,
                                                     check_items={"nq": 1,
                                                                  "ids": insert_ids,
                                                                  "limit": default_limit})[0]

        assert hybrid_search_0[0].ids == hybrid_search_1[0].ids
        assert hybrid_search_0[0].distances == hybrid_search_1[0].distances

    # @pytest.mark.tags(CaseLabel.L2)
    # @pytest.mark.parametrize("k", [1, 60, 1000, 16383])
    # @pytest.mark.parametrize("offset", [0, 1, 5])
    # @pytest.mark.skip("https://github.com/milvus-io/milvus/issues/32650")
    # def test_hybrid_search_RRFRanker_different_k(self, is_flush, k, offset):
    #     """
    #     target: test hybrid search normal case
    #     method: create connection, collection, insert and search.
    #             Note: here the result check is through comparing the score, the ids could not be compared
    #             because the high probability of the same score, then the id is not fixed in the range of
    #             the same score
    #     expected: hybrid search successfully with limit(topK)
    #     """
    #     # 1. initialize collection with data
    #     dim = 200
    #     collection_w, _, _, insert_ids, time_stamp = \
    #         self.init_collection_general(prefix, True, auto_id=False, dim=dim, is_flush=is_flush,
    #                                      enable_dynamic_field=False, multiple_dim_array=[dim, dim])[0:5]
    #     # 2. extract vector field name
    #     vector_name_list = cf.extract_vector_field_name_list(collection_w)
    #     vector_name_list.append(ct.default_float_vec_field_name)
    #     # 3. prepare search params for each vector field
    #     req_list = []
    #     search_res_dict_array = []
    #     for i in range(len(vector_name_list)):
    #         vectors = [[random.random() for _ in range(dim)] for _ in range(1)]
    #         search_res_dict = {}
    #         search_param = {
    #             "data": vectors,
    #             "anns_field": vector_name_list[i],
    #             "param": {"metric_type": "COSINE"},
    #             "limit": default_limit,
    #             "expr": "int64 > 0"}
    #         req = AnnSearchRequest(**search_param)
    #         req_list.append(req)
    #         # search for get the baseline of hybrid_search
    #         search_res = collection_w.search(vectors[:1], vector_name_list[i],
    #                                          default_search_params, default_limit,
    #                                          default_search_exp, offset=0,
    #                                          check_task=CheckTasks.check_search_results,
    #                                          check_items={"nq": 1,
    #                                                       "ids": insert_ids,
    #                                                       "limit": default_limit})[0]
    #         ids = search_res[0].ids
    #         for j in range(len(ids)):
    #             search_res_dict[ids[j]] = 1 / (j + k + 1)
    #         search_res_dict_array.append(search_res_dict)
    #     # 4. calculate hybrid search baseline for RRFRanker
    #     ids_answer, score_answer = cf.get_hybrid_search_base_results_rrf(search_res_dict_array)
    #     # 5. hybrid search
    #     hybrid_res = collection_w.hybrid_search(req_list, RRFRanker(k), default_limit,
    #                                             offset=offset,
    #                                             check_task=CheckTasks.check_search_results,
    #                                             check_items={"nq": 1,
    #                                                          "ids": insert_ids,
    #                                                          "limit": default_limit})[0]
    #     # 6. compare results through the re-calculated distances
    #     for i in range(len(score_answer[:default_limit])):
    #         assert score_answer[i] - hybrid_res[0].distances[i] < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("offset", [0, 1, 5])
    @pytest.mark.parametrize("rerank", [RRFRanker(), WeightedRanker(0.1, 0.9, 1)])
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_offset_inside_outside_params(self, primary_field, offset, rerank):
        """
        target: test hybrid search with offset inside and outside params
        method: create connection, collection, insert and search.
                Note: here the result check is through comparing the score, the ids could not be compared
                because the high probability of the same score, then the id is not fixed in the range of
                the same score
        expected: hybrid search successfully with limit(topK), and the result should be the same
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, primary_field=primary_field,
                                         multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        req_list = []
        vectors_list = []
        # 3. generate vectors
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            vectors_list.append(vectors)
        # 4. prepare search params for each vector field
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors_list[i],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": offset},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search with offset inside the params
        hybrid_res_inside = collection_w.hybrid_search(req_list, rerank, default_limit,
                                                       check_task=CheckTasks.check_search_results,
                                                       check_items={"nq": 1,
                                                                    "ids": insert_ids,
                                                                    "limit": default_limit})[0]
        # 5. hybrid search with offset parameter
        req_list = []
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors_list[i],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        hybrid_res = collection_w.hybrid_search(req_list, rerank, default_limit - offset,
                                                offset=offset,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": 1,
                                                             "ids": insert_ids,
                                                             "limit": default_limit - offset})[0]

        assert hybrid_res_inside[0].distances[offset:] == hybrid_res[0].distances

    @pytest.mark.tags(CaseLabel.L2)
    def test_hybrid_search_RRFRanker_empty_reqs(self):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. hybrid search with empty reqs
        collection_w.hybrid_search([], RRFRanker(), default_limit,
                                   check_task=CheckTasks.check_search_results,
                                   check_items={"nq": 0})

    # @pytest.mark.tags(CaseLabel.L1)
    # @pytest.mark.parametrize("k", [0, 16385])
    # @pytest.mark.skip(reason="issue #29867")
    # def test_hybrid_search_RRFRanker_k_out_of_range(self, k):
    #     """
    #     target: test hybrid search with default value to RRFRanker
    #     method: create connection, collection, insert and search.
    #             Note: here the result check is through comparing the score, the ids could not be compared
    #             because the high probability of the same score, then the id is not fixed in the range of
    #             the same score
    #     expected: hybrid search successfully with limit(topK)
    #     """
    #     # 1. initialize collection with data
    #     collection_w, _, _, insert_ids, time_stamp = \
    #         self.init_collection_general(prefix, True, dim=default_dim,
    #                                      multiple_dim_array=[default_dim, default_dim])[0:5]
    #     # 2. extract vector field name
    #     vector_name_list = cf.extract_vector_field_name_list(collection_w)
    #     vector_name_list.append(ct.default_float_vec_field_name)
    #     # 3. prepare search params for each vector field
    #     req_list = []
    #     search_res_dict_array = []
    #     for i in range(len(vector_name_list)):
    #         vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
    #         search_res_dict = {}
    #         search_param = {
    #             "data": vectors,
    #             "anns_field": vector_name_list[i],
    #             "param": {"metric_type": "COSINE", "offset": 0},
    #             "limit": default_limit,
    #             "expr": "int64 > 0"}
    #         req = AnnSearchRequest(**search_param)
    #         req_list.append(req)
    #         # search for get the base line of hybrid_search
    #         search_res = collection_w.search(vectors[:1], vector_name_list[i],
    #                                          default_search_params, default_limit,
    #                                          default_search_exp,
    #                                          check_task=CheckTasks.check_search_results,
    #                                          check_items={"nq": 1,
    #                                                       "ids": insert_ids,
    #                                                       "limit": default_limit})[0]
    #         ids = search_res[0].ids
    #         for j in range(len(ids)):
    #             search_res_dict[ids[j]] = 1 / (j + k + 1)
    #         search_res_dict_array.append(search_res_dict)
    #     # 4. calculate hybrid search base line for RRFRanker
    #     ids_answer, score_answer = cf.get_hybrid_search_base_results_rrf(search_res_dict_array)
    #     # 5. hybrid search
    #     hybrid_res = collection_w.hybrid_search(req_list, RRFRanker(k), default_limit,
    #                                             check_task=CheckTasks.check_search_results,
    #                                             check_items={"nq": 1,
    #                                                          "ids": insert_ids,
    #                                                          "limit": default_limit})[0]
    #     # 6. compare results through the re-calculated distances
    #     for i in range(len(score_answer[:default_limit])):
    #         delta = math.fabs(score_answer[i] - hybrid_res[0].distances[i])
    #         assert delta < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("limit", [1, 100, 16384])
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_different_limit_round_decimal(self, primary_field, limit):
        """
        target: test hybrid search with different valid limit and round decimal
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, primary_field=primary_field,
                                         multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        search_res_dict_array = []
        if limit > default_nb:
            limit = default_limit
        metrics = []
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            search_res_dict = {}
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            metrics.append("COSINE")
            # search to get the base line of hybrid_search
            search_res = collection_w.search(vectors[:1], vector_name_list[i],
                                             default_search_params, limit,
                                             default_expr,
                                             check_task=CheckTasks.check_search_results,
                                             check_items={"nq": 1,
                                                          "ids": insert_ids,
                                                          "limit": limit})[0]
            ids = search_res[0].ids
            distance_array = search_res[0].distances
            for j in range(len(ids)):
                search_res_dict[ids[j]] = distance_array[j]
            search_res_dict_array.append(search_res_dict)
        # 4. calculate hybrid search base line
        ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array, weights, metrics)
        # 5. hybrid search
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), limit,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": 1,
                                                             "ids": insert_ids,
                                                             "limit": limit})[0]
        # 6. compare results through the re-calculated distances
        for i in range(len(score_answer[:limit])):
            delta = math.fabs(score_answer[i] - hybrid_res[0].distances[i])
            assert delta < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L1)
    def test_hybrid_search_limit_out_of_range_max(self):
        """
        target: test hybrid search with over maximum limit
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search with over maximum limit
        limit = 16385
        error = {ct.err_code: 1100, ct.err_msg: "invalid max query result window, (offset+limit) "
                                                 "should be in range [1, 16384], but got %d" % limit}
        collection_w.hybrid_search(req_list, WeightedRanker(*weights), limit,
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_hybrid_search_limit_out_of_range_min(self):
        """
        target: test hybrid search with over minimum limit
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search with over maximum limit
        limit = 0
        error = {ct.err_code: 1, ct.err_msg: "`limit` value 0 is illegal"}
        collection_w.hybrid_search(req_list, WeightedRanker(*weights), limit,
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_with_output_fields(self, nq, dim, auto_id, is_flush, enable_dynamic_field,
                                              primary_field, vector_data_type):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        nq = 10
        multiple_dim_array = [dim, dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, auto_id=auto_id, dim=dim, is_flush=is_flush,
                                         primary_field=primary_field,
                                         enable_dynamic_field=enable_dynamic_field,
                                         multiple_dim_array=multiple_dim_array,
                                         vector_data_type=vector_data_type)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        metrics = []
        search_res_dict_array = []
        search_res_dict_array_nq = []
        vectors = cf.gen_vectors_based_on_vector_type(nq, dim, vector_data_type)

        # get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            metrics.append("COSINE")

        # get the result of search with the same params of the following hybrid search
        single_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        for k in range(nq):
            for i in range(len(vector_name_list)):
                search_res_dict = {}
                search_res_dict_array = []
                vectors_search = vectors[k]
                # 5. search to get the base line of hybrid_search
                search_res = collection_w.search([vectors_search], vector_name_list[i],
                                                 single_search_param, default_limit,
                                                 default_expr,
                                                 check_task=CheckTasks.check_search_results,
                                                 check_items={"nq": 1,
                                                              "ids": insert_ids,
                                                              "limit": default_limit})[0]
                ids = search_res[0].ids
                distance_array = search_res[0].distances
                for j in range(len(ids)):
                    search_res_dict[ids[j]] = distance_array[j]
                search_res_dict_array.append(search_res_dict)
            search_res_dict_array_nq.append(search_res_dict_array)

        # 6. calculate hybrid search base line
        score_answer_nq = []
        for k in range(nq):
            ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array_nq[k], weights, metrics)
            score_answer_nq.append(score_answer)
        # 7. hybrid search
        output_fields = [ct.default_int64_field_name]
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                                output_fields=output_fields,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": nq,
                                                             "ids": insert_ids,
                                                             "limit": default_limit})[0]
        # 8. compare results through the re-calculated distances
        for k in range(len(score_answer_nq)):
            for i in range(len(score_answer_nq[k][:default_limit])):
                assert score_answer_nq[k][i] - hybrid_res[k].distances[i] < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_with_output_fields_all_fields(self, nq, dim, auto_id, is_flush, enable_dynamic_field,
                                                         primary_field, vector_data_type):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        nq = 10
        multiple_dim_array = [dim, dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, auto_id=auto_id, dim=dim, is_flush=is_flush,
                                         primary_field=primary_field,
                                         enable_dynamic_field=enable_dynamic_field,
                                         multiple_dim_array=multiple_dim_array,
                                         vector_data_type=vector_data_type)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        metrics = []
        search_res_dict_array = []
        search_res_dict_array_nq = []
        vectors = cf.gen_vectors_based_on_vector_type(nq, dim, vector_data_type)

        # get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            metrics.append("COSINE")

        # get the result of search with the same params of the following hybrid search
        single_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        for k in range(nq):
            for i in range(len(vector_name_list)):
                search_res_dict = {}
                search_res_dict_array = []
                vectors_search = vectors[k]
                # 5. search to get the base line of hybrid_search
                search_res = collection_w.search([vectors_search], vector_name_list[i],
                                                 single_search_param, default_limit,
                                                 default_search_exp,
                                                 check_task=CheckTasks.check_search_results,
                                                 check_items={"nq": 1,
                                                              "ids": insert_ids,
                                                              "limit": default_limit})[0]
                ids = search_res[0].ids
                distance_array = search_res[0].distances
                for j in range(len(ids)):
                    search_res_dict[ids[j]] = distance_array[j]
                search_res_dict_array.append(search_res_dict)
            search_res_dict_array_nq.append(search_res_dict_array)

        # 6. calculate hybrid search base line
        score_answer_nq = []
        for k in range(nq):
            ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array_nq[k], weights, metrics)
            score_answer_nq.append(score_answer)
        # 7. hybrid search
        output_fields = [default_int64_field_name, default_float_field_name, default_string_field_name,
                         default_json_field_name]
        output_fields = output_fields + vector_name_list
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                                output_fields=output_fields,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": nq,
                                                             "ids": insert_ids,
                                                             "limit": default_limit})[0]
        # 8. compare results through the re-calculated distances
        for k in range(len(score_answer_nq)):
            for i in range(len(score_answer_nq[k][:default_limit])):
                assert score_answer_nq[k][i] - hybrid_res[k].distances[i] < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_with_output_fields_all_fields(self, nq, dim, auto_id, is_flush, enable_dynamic_field,
                                                         primary_field, vector_data_type):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        nq = 10
        multiple_dim_array = [dim, dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, auto_id=auto_id, dim=dim, is_flush=is_flush,
                                         primary_field=primary_field,
                                         enable_dynamic_field=enable_dynamic_field,
                                         multiple_dim_array=multiple_dim_array,
                                         vector_data_type=vector_data_type)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        metrics = []
        search_res_dict_array = []
        search_res_dict_array_nq = []
        vectors = cf.gen_vectors_based_on_vector_type(nq, dim, vector_data_type)

        # get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            metrics.append("COSINE")

        # get the result of search with the same params of the following hybrid search
        single_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        for k in range(nq):
            for i in range(len(vector_name_list)):
                search_res_dict = {}
                search_res_dict_array = []
                vectors_search = vectors[k]
                # 5. search to get the base line of hybrid_search
                search_res = collection_w.search([vectors_search], vector_name_list[i],
                                                 single_search_param, default_limit,
                                                 default_search_exp,
                                                 check_task=CheckTasks.check_search_results,
                                                 check_items={"nq": 1,
                                                              "ids": insert_ids,
                                                              "limit": default_limit})[0]
                ids = search_res[0].ids
                distance_array = search_res[0].distances
                for j in range(len(ids)):
                    search_res_dict[ids[j]] = distance_array[j]
                search_res_dict_array.append(search_res_dict)
            search_res_dict_array_nq.append(search_res_dict_array)

        # 6. calculate hybrid search base line
        score_answer_nq = []
        for k in range(nq):
            ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array_nq[k], weights, metrics)
            score_answer_nq.append(score_answer)
        # 7. hybrid search
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                                output_fields=["*"],
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": nq,
                                                             "ids": insert_ids,
                                                             "limit": default_limit})[0]
        # 8. compare results through the re-calculated distances
        for k in range(len(score_answer_nq)):
            for i in range(len(score_answer_nq[k][:default_limit])):
                assert score_answer_nq[k][i] - hybrid_res[k].distances[i] < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("output_fields",
                             [[default_search_field], [default_search_field, default_int64_field_name]])
    @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    def test_hybrid_search_with_output_fields_sync_async(self, nq, primary_field, output_fields, _async):
        """
        target: test hybrid search normal case
        method: create connection, collection, insert and search
        expected: hybrid search successfully with limit(topK)
        """
        # 1. initialize collection with data
        multiple_dim_array = [default_dim, default_dim]
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, dim=default_dim,
                                         primary_field=primary_field,
                                         multiple_dim_array=multiple_dim_array)[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        metrics = []
        search_res_dict_array = []
        search_res_dict_array_nq = []
        vectors = cf.gen_vectors_based_on_vector_type(nq, default_dim, "FLOAT_VECTOR")

        # get hybrid search req list
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE"},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            metrics.append("COSINE")

        # get the result of search with the same params of the following hybrid search
        single_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        for k in range(nq):
            for i in range(len(vector_name_list)):
                search_res_dict = {}
                search_res_dict_array = []
                vectors_search = vectors[k]
                # 5. search to get the base line of hybrid_search
                search_res = collection_w.search([vectors_search], vector_name_list[i],
                                                 single_search_param, default_limit,
                                                 default_expr, _async=_async,
                                                 check_task=CheckTasks.check_search_results,
                                                 check_items={"nq": 1,
                                                              "ids": insert_ids,
                                                              "limit": default_limit,
                                                              "_async": _async})[0]
                if _async:
                    search_res.done()
                    search_res = search_res.result()
                ids = search_res[0].ids
                distance_array = search_res[0].distances
                for j in range(len(ids)):
                    search_res_dict[ids[j]] = distance_array[j]
                search_res_dict_array.append(search_res_dict)
            search_res_dict_array_nq.append(search_res_dict_array)

        # 6. calculate hybrid search base line
        score_answer_nq = []
        for k in range(nq):
            ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array_nq[k], weights, metrics)
            score_answer_nq.append(score_answer)
        # 7. hybrid search
        hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
                                                output_fields=output_fields, _async=_async,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": nq,
                                                             "ids": insert_ids,
                                                             "limit": default_limit,
                                                             "_async": _async})[0]
        if _async:
            hybrid_res.done()
            hybrid_res = hybrid_res.result()
        # 8. compare results through the re-calculated distances
        for k in range(len(score_answer_nq)):
            for i in range(len(score_answer_nq[k][:default_limit])):
                assert score_answer_nq[k][i] - hybrid_res[k].distances[i] < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("rerank", [RRFRanker(), WeightedRanker(0.1, 0.9, 1)])
    def test_hybrid_search_offset_both_inside_outside_params(self, rerank):
        """
        target: test hybrid search with offset inside and outside params
        method: create connection, collection, insert and search.
                Note: here the result check is through comparing the score, the ids could not be compared
                because the high probability of the same score, then the id is not fixed in the range of
                the same score
        expected: Raise exception
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        req_list = []
        vectors_list = []
        # 3. generate vectors
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(1)]
            vectors_list.append(vectors)
        # 4. prepare search params for each vector field
        for i in range(len(vector_name_list)):
            search_param = {
                "data": vectors_list[i],
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search with offset inside the params
        error = {ct.err_code: 1, ct.err_msg: "Provide offset both in kwargs and param, expect just one"}
        collection_w.hybrid_search(req_list, rerank, default_limit, offset=2,
                                   check_task=CheckTasks.err_res, check_items=error)

    # @pytest.mark.tags(CaseLabel.L2)
    # @pytest.mark.parametrize("limit", [1, 100, 16384])
    # @pytest.mark.parametrize("primary_field", [ct.default_int64_field_name, ct.default_string_field_name])
    # def test_hybrid_search_is_partition_key(self, nq, primary_field, limit, vector_data_type):
    #     """
    #     target: test hybrid search with different valid limit and round decimal
    #     method: create connection, collection, insert and search
    #     expected: hybrid search successfully with limit(topK)
    #     """
    #     # 1. initialize collection with data
    #     collection_w, _, _, insert_ids, time_stamp = \
    #         self.init_collection_general(prefix, True, primary_field=primary_field,
    #                                      multiple_dim_array=[default_dim, default_dim],
    #                                      vector_data_type=vector_data_type,
    #                                      is_partition_key=ct.default_float_field_name)[0:5]
    #     # 2. extract vector field name
    #     vector_name_list = cf.extract_vector_field_name_list(collection_w)
    #     vector_name_list.append(ct.default_float_vec_field_name)
    #     # 3. prepare search params
    #     req_list = []
    #     weights = [0.2, 0.3, 0.5]
    #     metrics = []
    #     search_res_dict_array = []
    #     search_res_dict_array_nq = []
    #     vectors = cf.gen_vectors_based_on_vector_type(nq, default_dim, vector_data_type)
    #
    #     # get hybrid search req list
    #     for i in range(len(vector_name_list)):
    #         search_param = {
    #             "data": vectors,
    #             "anns_field": vector_name_list[i],
    #             "param": {"metric_type": "COSINE"},
    #             "limit": default_limit,
    #             "expr": "int64 > 0"}
    #         req = AnnSearchRequest(**search_param)
    #         req_list.append(req)
    #         metrics.append("COSINE")
    #
    #     # get the result of search with the same params of the following hybrid search
    #     single_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    #     for k in range(nq):
    #         for i in range(len(vector_name_list)):
    #             search_res_dict = {}
    #             search_res_dict_array = []
    #             vectors_search = vectors[k]
    #             # 5. search to get the base line of hybrid_search
    #             search_res = collection_w.search([vectors_search], vector_name_list[i],
    #                                              single_search_param, default_limit,
    #                                              default_search_exp,
    #                                              check_task=CheckTasks.check_search_results,
    #                                              check_items={"nq": 1,
    #                                                           "ids": insert_ids,
    #                                                           "limit": default_limit})[0]
    #             ids = search_res[0].ids
    #             distance_array = search_res[0].distances
    #             for j in range(len(ids)):
    #                 search_res_dict[ids[j]] = distance_array[j]
    #             search_res_dict_array.append(search_res_dict)
    #         search_res_dict_array_nq.append(search_res_dict_array)
    #
    #     # 6. calculate hybrid search base line
    #     score_answer_nq = []
    #     for k in range(nq):
    #         ids_answer, score_answer = cf.get_hybrid_search_base_results(search_res_dict_array_nq[k], weights, metrics)
    #         score_answer_nq.append(score_answer)
    #     # 7. hybrid search
    #     hybrid_res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), default_limit,
    #                                             check_task=CheckTasks.check_search_results,
    #                                             check_items={"nq": nq,
    #                                                          "ids": insert_ids,
    #                                                          "limit": default_limit})[0]
    #     # 8. compare results through the re-calculated distances
    #     for k in range(len(score_answer_nq)):
    #         for i in range(len(score_answer_nq[k][:default_limit])):
    #             assert score_answer_nq[k][i] - hybrid_res[k].distances[i] < hybrid_search_epsilon


    @pytest.mark.tags(CaseLabel.L1)
    def test_hybrid_search_result_order(self, nq):
        """
        target: test hybrid search result having correct order for cosine distance
        method: create connection, collection, insert and search
        expected: hybrid search successfully and result order is correct
        """
        # 1. initialize collection with data
        collection_w, _, _, insert_ids, time_stamp = \
            self.init_collection_general(prefix, True, multiple_dim_array=[default_dim, default_dim])[0:5]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        vector_name_list.append(ct.default_float_vec_field_name)
        # 3. prepare search params
        req_list = []
        weights = [0.2, 0.3, 0.5]
        for i in range(len(vector_name_list)):
            vectors = [[random.random() for _ in range(default_dim)] for _ in range(nq)]
            search_param = {
                "data": vectors,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "COSINE", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
        # 4. hybrid search
        res = collection_w.hybrid_search(req_list, WeightedRanker(*weights), 10)[0]
        is_sorted_descend = lambda lst: all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))
        for i in range(nq):
            assert is_sorted_descend(res[i].distances)

    @pytest.mark.tags(CaseLabel.L2)
    def test_hybrid_search_sparse_normal(self):
        """
        target: test hybrid search after loading sparse vectors
        method: Test hybrid search after loading sparse vectors
        expected: hybrid search successfully with limit(topK)
        """
        nb, auto_id, dim, enable_dynamic_field = 20000, False, 768, False
        # 1. init collection
        collection_w, insert_vectors, _, insert_ids = self.init_collection_general(prefix, True, nb=nb,
                                                                                   multiple_dim_array=[dim, dim * 2],
                                                                                   with_json=False,
                                                                                   vector_data_type="SPARSE_FLOAT_VECTOR")[
                                                      0:4]
        # 2. extract vector field name
        vector_name_list = cf.extract_vector_field_name_list(collection_w)
        # 3. prepare search params
        req_list = []
        search_res_dict_array = []
        k = 60
        search_params = {'params': {'nprobe': 32}}
        for i in range(len(vector_name_list)):
            # vector = cf.gen_sparse_vectors(1, dim)
            vector = insert_vectors[0][i + 3][-1:]
            search_res_dict = {}
            search_param = {
                "data": vector,
                "anns_field": vector_name_list[i],
                "param": {"metric_type": "IP", "offset": 0},
                "limit": default_limit,
                "expr": "int64 > 0"}
            req = AnnSearchRequest(**search_param)
            req_list.append(req)
            # search for get the base line of hybrid_search
            search_res = collection_w.search(vector, vector_name_list[i],
                                             search_params, default_limit,
                                             default_expr,
                                             )[0]
            ids = search_res[0].ids
            for j in range(len(ids)):
                search_res_dict[ids[j]] = 1 / (j + k + 1)
            search_res_dict_array.append(search_res_dict)
        # 4. calculate hybrid search base line for RRFRanker
        ids_answer, score_answer = cf.get_hybrid_search_base_results_rrf(search_res_dict_array)
        # 5. hybrid search
        hybrid_res = collection_w.hybrid_search(req_list, RRFRanker(k), default_limit,
                                                check_task=CheckTasks.check_search_results,
                                                check_items={"nq": 1,
                                                             "ids": insert_ids,
                                                             "limit": default_limit})[0]
        # 6. compare results through the re-calculated distances
        for i in range(len(score_answer[:default_limit])):
            delta = math.fabs(score_answer[i] - hybrid_res[0].distances[i])
            assert delta < hybrid_search_epsilon

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_lite_delete_search(self, enable_milvus_local_api):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. insert
        default_nb = 1000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i,
                 default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0,
                 default_string_field_name: str(i)}
                for i in range(default_nb)]
        pks = client_w.insert(client, collection_name, rows)[0]
        # 3. delete
        delete_num = 3
        client_w.delete(client, collection_name, filter=f"id < {delete_num}")
        # 4. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        for insert_id in range(delete_num):
            if insert_id in insert_ids:
                insert_ids.remove(insert_id)
        limit = default_nb - delete_num
        client_w.search(client, collection_name, vectors_to_search, limit=default_nb,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": len(vectors_to_search),
                                     "ids": insert_ids,
                                     "limit": limit})
        # 5. query
        client_w.query(client, collection_name, filter=default_search_exp,
                       check_task=CheckTasks.check_query_results,
                       check_items={exp_res: rows[delete_num:],
                                    "with_vec": True,
                                    "primary_field": default_primary_key_field_name})
        client_w.drop_collection(client, collection_name)