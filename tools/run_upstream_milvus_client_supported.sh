#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MILVUS_CLIENT_TEST_DIR="${MILVUS_CLIENT_TEST_DIR:-$HOME/Workspace/dev/milvus/tests/python_client/milvus_client}"
DB_URI="${DB_URI:-/tmp/milvus_lite_supported_client_suite.db}"
CACHE_DIR="${CACHE_DIR:-/tmp/milvus_lite_pytest_cache}"

export PYTHONPATH="$ROOT_DIR/tools/pytest_compat${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"$ROOT_DIR/.venv/bin/python" -m pytest \
  -p no:rerunfailures \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasValid::test_milvus_client_alter_alias_default" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasOperation::test_milvus_client_rename_back_old_alias" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasOperation::test_milvus_client_rename_back_old_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasOperationInvalid::test_milvus_client_drop_not_exist_alias" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasOperationInvalid::test_milvus_client_drop_same_alias_twice" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_alias.py::TestMilvusClientAliasOperationInvalid::test_milvus_client_reuse_alias_name" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientHasCollectionValid::test_milvus_client_has_collection_multithread" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientHasCollectionInvalid::test_milvus_client_has_collection_not_existed" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientHasCollectionInvalid::test_milvus_client_has_collection_deleted_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientListCollection::test_milvus_client_list_collections_multi_collections" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientReleaseCollectionValid::test_milvus_client_release_unloaded_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientLoadCollectionValid::test_milvus_client_load_loaded_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientLoadCollectionValid::test_milvus_client_load_collection_repeatedly" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionValid::test_milvus_client_collection_multi_create_drop" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionValid::test_milvus_client_collection_after_drop" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionValid::test_milvus_client_collection_valid_naming_rules" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientRenameCollectionValid::test_milvus_client_rename_collection_multiple_times" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionCountIP::test_milvus_client_collection_count_after_index_created[1]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionCountIP::test_milvus_client_collection_count_after_index_created[1000]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionCountIP::test_milvus_client_collection_count_after_index_created[2001]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionMultiCollections::test_milvus_client_collection_count_multi_collections_l2[1]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionMultiCollections::test_milvus_client_collection_count_multi_collections_l2[1000]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_collection.py::TestMilvusClientCollectionMultiCollections::test_milvus_client_collection_count_multi_collections_l2[2001]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_insert.py::TestMilvusClientInsertValid::test_milvus_client_insert_empty_data" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_insert.py::TestMilvusClientInsertValid::test_insert_row_data[1]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_insert.py::TestMilvusClientInsertValid::test_insert_row_data[3000]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_insert.py::TestInsertOperation::test_insert_partition_with_ids" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_insert.py::TestInsertOperation::test_insert_drop_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestPartitionParams::test_partition_default_name" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestPartitionParams::test_partition_naming_rules[_Partiti0n]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestPartitionParams::test_partition_naming_rules[pArt1_ti0n]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestPartitionParams::test_partition_drop" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestShowBase::test_list_partitions" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_partition.py::TestHasBase::test_has_partition" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_field_none_or_empty[True]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_field_none_or_empty[False]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_one_field[True]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_one_field[False]" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_empty_collection" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_empty_term_array" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_with_dup_primary_key" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_empty" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_empty_with_pagination" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_by_int64" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_not_in_empty_and_all" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_random_values" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_not_in_random" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_json_contains" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_expr_json_contains_combined_with_normal" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_primary_field" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_output_multi_float_vec_field" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_multi_logical_exprs" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryString::test_milvus_client_query_compare_two_fields" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryString::test_milvus_client_query_string_with_mix_expr" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_pagination" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_pagination_with_insert_data" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_pagination_without_limit" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientQueryValid::test_milvus_client_query_pagination_with_offset_over_num_entities" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientGetValid::test_milvus_client_get_normal" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestMilvusClientGetValid::test_milvus_client_get_output_fields" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_count_with_expr" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_count_consistency_level" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_count_delete_insert_duplicate_ids" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_query_count_expr_json" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_count_bool_expressions" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_query.py::TestQueryCount::test_milvus_client_count_expression_comparative" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_search.py::TestMilvusClientSearchValid::test_milvus_client_search_different_metric_types_not_specifying_in_search_params" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_search.py::TestMilvusClientSearchValid::test_milvus_client_array_insert_search" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_upsert.py::TestMilvusClientUpsertValid::test_milvus_client_upsert_empty_data" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_upsert.py::TestMilvusClientUpsertValid::test_milvus_client_upsert_data_pk_not_exist" \
  "$MILVUS_CLIENT_TEST_DIR/test_milvus_client_upsert.py::TestMilvusClientUpsertValid::test_milvus_client_upsert_same_with_inserted_data" \
  --uri "$DB_URI" \
  -o "cache_dir=$CACHE_DIR" \
  --tb=short \
  -q \
  --show-capture=no
