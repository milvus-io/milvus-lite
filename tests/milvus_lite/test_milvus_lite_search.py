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
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker, connections, CollectionSchema, Collection, \
    FieldSchema, DataType

from base.client_base import TestcaseBase
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.util_pymilvus import *
from common.constants import *
from pymilvus.orm.types import CONSISTENCY_STRONG, CONSISTENCY_BOUNDED, CONSISTENCY_SESSION, CONSISTENCY_EVENTUALLY
from base.high_level_api_wrapper import HighLevelApiWrapper
from base.collection_wrapper import ApiCollectionWrapper

client_w = HighLevelApiWrapper()

prefix = "milvus_client_api_search"
epsilon = ct.epsilon
default_nb = ct.default_nb
default_nb_medium = ct.default_nb_medium
default_nq = ct.default_nq
default_dim = ct.default_dim
default_limit = ct.default_limit
default_search_exp = "id >= 0"
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
hybrid_search_epsilon = 0.01
max_hybrid_search_req_num = ct.max_hybrid_search_req_num
default_json_field_name = ct.default_json_field_name
default_int64_field_name = ct.default_int64_field_name
max_limit = ct.max_limit
min_dim = ct.min_dim


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

    @pytest.fixture(scope="function", params=[default_nb, default_nb_medium])
    def nb(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[2, 500])
    def nq(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=[32, 128])
    def dim(self, request):
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

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_range_search_cosine(self, enable_milvus_local_api):
        """
        target: test delete (high level api)
        method: create connection, collection, insert delete, and search
        expected: search/query successfully without deleted data
        """
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. insert
        default_nb = 2000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)
        range_filter = random.uniform(0.5, 1)
        radius = random.uniform(0, range_filter)
        # 2. range search
        range_search_params = {"metric_type": "COSINE",
                               "params": {"radius": radius, "range_filter": range_filter}}
        search_res = client_w.search(client, collection_name, rng.random((1, default_dim)), limit=3,
                                         search_params=range_search_params)[0]
        # 3. check search results
        for hits in search_res[0]:
            distance = hits["distance"]
            assert range_filter >= distance > radius

    @pytest.mark.tags(CaseLabel.L2)
    def test_range_search_load_release_load(self, enable_milvus_local_api):
        """
        target: test range search when load before flush
        method: 1. insert data and load
                2. flush, and load
                3. search the collection
        expected: search success with limit(topK)
        """
        dim = 128
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=enable_milvus_local_api)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. insert data
        default_nb = 2000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        client_w.insert(client, collection_name, rows)

        index_params = client_w.prepare_index_params(client)[0]
        index_params.add_index(field_name="vector", index_type="HNSW", metric_type="IP")
        client_w.create_index(client, collection_name, index_params)
        # 3. load
        client_w.load_collection(client, collection_name)
        # 4. reload
        client_w.release_collection(client, collection_name)
        client_w.load_collection(client, collection_name)
        # 5. search
        vectors = [[random.random() for _ in range(dim)]
                   for _ in range(default_nq)]
        range_filter, radius = 1000, 0.5
        range_search_params = {"metric_type": "COSINE", "params": {"ef": 32, "radius": 0.5,
                                                                   "range_filter": 1000}}
        search_res = client_w.search(client, collection_name, vectors[:default_nq], search_params=range_search_params,
                                                output_fields=[default_primary_key_field_name])
        for hits in search_res[0][0]:
            distance = hits["distance"]
            assert range_filter >= distance > radius

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("index", ["BIN_FLAT"])
    def test_range_search_binary_jaccard(self, index):
        """
        target: range search on binary_array
        method: range search on binary_array
        expected: successful search
        """
        connections.connect(uri='./test.db')
        collection_name = cf.gen_unique_str(prefix)
        int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
        dim = 128
        nb = 3000
        vector_field_name = "binary_vector"
        binary_vector = FieldSchema(name=vector_field_name, dtype=DataType.BINARY_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[int64_field, binary_vector], enable_dynamic_field=True)
        collection_w = Collection(collection_name, schema)
        _ , vectors = cf.gen_binary_vectors(nb, dim)
        rows = []
        for vec in vectors:
            _data = {vector_field_name: vec}
            rows.append(_data)
        collection_w.insert(rows)
        collection_w.flush()
        default_index = {"index_type": index, "params": {
            "nlist": 128}, "metric_type": "JACCARD"}
        collection_w.create_index(vector_field_name, index_params=default_index)
        search_params = {"metric_type": "JACCARD",
                         "params": {"radius": 0.8, "range_filter": 0.2}}
        collection_w.load()
        collection_w.search(vectors[:default_nq], "binary_vector",
                            search_params, default_limit, default_search_exp)
        # 5. range search
        radius, range_filter = 0.7, 0.1
        search_params = {"metric_type": "JACCARD", "params": {"nprobe": 10, "radius": radius,
                                                              "range_filter": range_filter}}
        search_res = collection_w.search(vectors[:default_nq], "binary_vector",
                            search_params)
        for hits in search_res[0]:
            assert radius > hits.distance > range_filter

    @pytest.mark.tags(CaseLabel.L2)
    def test_range_search_sparse(self):
        """
        target: test sparse index normal range search
        method: create connection, collection, insert and range search
        expected: range search successfully
        """
        # 1. initialize with data
        collection_w = self.init_collection_general(prefix, True, nb=5000,
                                                    with_json=True,
                                                    vector_data_type=ct.sparse_vector)[0]
        range_filter = random.uniform(0.5, 1)
        radius = random.uniform(0, 0.5)

        # 2. range search
        range_search_params = {"metric_type": "IP",
                               "params": {"radius": radius, "range_filter": range_filter}}
        d = cf.gen_default_list_sparse_data(nb=1)
        search_res = collection_w.search(d[-1][-1:], ct.default_sparse_vec_field_name,
                                         range_search_params, default_limit,
                                         default_search_exp)[0]

        # 3. check search results
        for hits in search_res:
            for distance in hits.distances:
                assert range_filter >= distance > radius

