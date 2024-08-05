import json
import random
import time
from multiprocessing import Process
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from pymilvus.client.types import LoadState
import pytest
import os
from base.client_base import TestcaseBase
from common.common_type import CaseLabel, CheckTasks
from base.high_level_api_wrapper import HighLevelApiWrapper
from common import common_type as ct


client_w = HighLevelApiWrapper()
prefix = "milvus_lite_migrate"
default_dim = ct.default_dim
default_primary_key_field_name = "id"
default_vector_field_name = "vector"
default_float_field_name = ct.default_float_field_name
default_bool_field_name = ct.default_bool_field_name
# default_nb = ct.default_nb
default_limit = ct.default_limit
default_nq = ct.default_nq
default_string_field_name = ct.default_string_field_name
default_int32_array_field_name = ct.default_int32_array_field_name
c_name = "coll_migrate_1"
d_file = f"local_migrate_case_test.db"


class TestMilvusLiteMigrate(TestcaseBase):
    """
    1. create collection for data preparation after normal search
    2. dump to json file
    3. verify that the key in the generated json file matches the schema
    """
    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_lite_migrate_json_file(self):
        num_entities, dim = 3000, 8
        connections.connect("default", uri=d_file)
        fields = [
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "collection is the simplest demo to introduce the APIs")
        hello_milvus = Collection(c_name, schema)
        rng = np.random.default_rng(seed=19530)
        entities = [
            # provide the pk field because `auto_id` is set to False
            rng.random(num_entities).tolist(),  # field random, only supports list
            [str(i) for i in range(num_entities)],
            rng.random((num_entities, dim)),  # field embeddings, supports numpy.ndarray and list
        ]
        insert_result = hello_milvus.insert(entities)
        index = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }

        hello_milvus.create_index("embeddings", index)
        hello_milvus.load()
        assert utility.load_state(c_name) == LoadState.Loaded
        vectors_to_search = entities[-1][-2:]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])

        for hits in result:
            for hit in hits:
                print(f"hit: {hit}, random field: {hit.entity.get('random')}")

        result = hello_milvus.query(expr="random > 0.5", output_fields=["random", "embeddings"])
        r1 = hello_milvus.query(expr="random > 0.5", limit=4, output_fields=["random"])
        r2 = hello_milvus.query(expr="random > 0.5", offset=1, limit=3, output_fields=["random"])
        result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5",
                                     output_fields=["random"])

        ids = insert_result.primary_keys
        expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
        result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])

    @pytest.mark.tags(CaseLabel.L2)
    def test_check_json_file_key(self):
        """
        pytest test_milvus_lite_migrate.py::TestMilvusLiteMigrate::test_check_json_file_key -s
        To check whether the data export is working properly using the above command, start a new command line
        """
        command = f"milvus-lite dump -d ./{d_file} -c {c_name} -p ./data_json"
        os.system(command)
        time.sleep(3)
        dir = os.popen("ls data_json/").read()
        dir = dir.replace("\n", "")
        with open(f'data_json/{dir}/1.json') as user_file:
            file_contents = user_file.read()
        parsed_json = json.loads(file_contents)
        os.popen("rm -rf data_json")
        keys = parsed_json["rows"][0].keys()
        assert list(keys) == ['random', 'pk', 'embeddings']