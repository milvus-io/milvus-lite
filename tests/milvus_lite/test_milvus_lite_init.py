import pytest
import numpy as np
from base.client_base import TestcaseBase
from common.common_type import CaseLabel, CheckTasks
from common import common_func as cf
from common import common_type as ct
from base.high_level_api_wrapper import HighLevelApiWrapper

client_w = HighLevelApiWrapper()
default_primary_key_field_name = "id"
default_vector_field_name = "vector"
default_dim = ct.default_dim
prefix = "milvus_lite_api_collection"
default_float_field_name = ct.default_float_field_name
default_string_field_name = ct.default_string_field_name
default_limit = ct.default_limit
default_nb = ct.default_nb


class TestMilvusLocalFilePath(TestcaseBase):

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("file_path", ["./local1.db", "file", "/root/file1.db", "../../file.docx", "/file.db"])
    def test_milvus_local_file_path(self, file_path):
        client = self._connect(enable_milvus_client_api=True, enable_milvus_local_api=file_path)
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        client_w.create_collection(client, collection_name, default_dim)
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        results = client_w.insert(client, collection_name, rows)[0]
        assert results['insert_count'] == default_nb
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
    @pytest.mark.xfail(reason="illegal init")
    @pytest.mark.parametrize("file_path", ["/", "/cc/cc/cc/cc.db", "    ", "/dir ", 0])
    def test_milvus_local_invalid_file_path(self, file_path):
        self._connect(enable_milvus_client_api=True, enable_milvus_local_api=file_path)