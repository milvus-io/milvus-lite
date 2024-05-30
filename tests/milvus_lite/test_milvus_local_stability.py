import logging
import random
import string
import time
import numpy as np
import pytest

from base.client_base import TestcaseBase
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.thread_util import ConcurrencyObj
from common.constants import *
from base.high_level_api_wrapper import HighLevelApiWrapper
client_w = HighLevelApiWrapper()
prefix = "milvus_local_stability"
default_dim = ct.default_dim
default_primary_key_field_name = "id"
default_nb = ct.default_nb
default_search_exp = "id in ['1','9','90', '900', '9000']"
# default_search_exp = "varchar < 500.0"
default_float_field_name = ct.default_float_field_name
default_vector_field_name = "vector"
default_string_field_name = ct.default_string_field_name
default_limit = ct.default_limit
exp_res = "exp_res"


class TestMilvusLocalStability(TestcaseBase):

    @pytest.mark.tags(CaseLabel.L3)
    def test_milvus_local_stability(self):
        timeout = 3600
        s_name = ''.join(random.sample(string.ascii_letters + string.digits, 4))
        file_name = "./milvus_lite_stab" + s_name + ".db"
        file = cf.gen_unique_str(file_name)
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=file)
        c_name = cf.gen_unique_str(prefix)
        nb, dim = 500000, 768
        client_w.create_collection(client, c_name, dim)
        collections = client_w.list_collections(client)[0]
        assert c_name in collections
        client_w.describe_collection(client, c_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": c_name,
                                                  "dim": dim})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows, row = [], []
        count, num_c = 0, 0
        for i in range(nb):
            r = {default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, dim))[0]),
                        default_float_field_name: i * 1.0,
                        default_string_field_name: str(i)}
            row = [r]
            count += 1
            if count == 1000:
                num_c += 1
                client_w.insert(client, c_name, row)
                count = 0
                row.clear()
            rows.append(r)
        sta = time.time()
        client_w.load_collection(client, c_name)
        end = time.time()

        def scene_search_query():
            vectors_to_search = rng.random((1, dim))
            client_w.search(client, c_name, vectors_to_search,
                            check_task=CheckTasks.check_search_results,
                            check_items={"enable_milvus_client_api": True,
                                         "nq": len(vectors_to_search),
                                         "limit": default_limit})
            #
            # client_w.query(client, c_name, filter=default_search_exp,
            #                check_task=CheckTasks.check_query_results,
            #                check_items={exp_res: rows,
            #                             "with_vec": True,
            #                             "primary_field": default_primary_key_field_name})
            logging.info(f"scene_search_query for collection={c_name}")

        current = ConcurrencyObj(func_obj=scene_search_query, interval=1)
        current.multi_func(num=20)
        time.sleep(timeout)
        current.set_stop_flag()

    @pytest.mark.tags(CaseLabel.L3)
    def test_milvus_local_hnsw_serial_performance(self):
        s_name = ''.join(random.sample(string.ascii_letters + string.digits, 4))
        file_name = "./milvus_lite_stab" + s_name + ".db"
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=file_name)
        c_name = cf.gen_unique_str(prefix)
        nb, dim = 500000, 768
        client_w.create_collection(client, c_name, dim)
        collections = client_w.list_collections(client)[0]
        assert c_name in collections
        client_w.describe_collection(client, c_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": c_name,
                                                  "dim": dim})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows, row = [], []
        count, num_c = 0, 0
        insert_start = time.time()
        for i in range(nb):
            r = {default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, dim))[0]),
                        default_float_field_name: i * 1.0,
                        default_string_field_name: str(i)}
            row = [r]
            count += 1
            if count == 10000:
                num_c += 1
                client_w.insert(client, c_name, row)
                count = 0
                row.clear()
            rows.append(r)
        insert_time = round(time.time() - insert_start, 2)
        logging.info(f"insert done. insert time: {insert_time}")
        client_w.release_collection(client, c_name)
        client_w.load_collection(client, c_name)
        response_time = []
        for i in range(500):
            vectors_to_search = rng.random((1, dim))
            # insert_ids = [i for i in range(nb)]
            start = time.time()
            client_w.search(client, c_name, vectors_to_search)
            rt = time.time() - start
            response_time.append(rt * 1000)
            logging.info(f"search response time : {rt} s")
        logging.info("------- end ------ ")
        logging.info(f"avg response time : {sum(response_time)/len(response_time)}")
