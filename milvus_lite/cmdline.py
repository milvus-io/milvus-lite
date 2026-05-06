"""Command line tools for Milvus Lite.

Provides the unified ``milvus-lite`` entry point.  Heavy optional dump
dependencies are imported only when their subcommand is used.
"""

from __future__ import annotations

import argparse
import pathlib
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np

from milvus_lite.adapter.grpc.server import (
    DEFAULT_HOST,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PORT,
    run_server,
)


DEFAULT_CHUNK_SIZE = 512 * 1024 * 1024
DEFAULT_BATCH_SIZE = 1000
DUMP_ALIAS = "milvus_lite_dump"


def bfloat16_to_float32(byte_data: bytes) -> list[float]:
    """Convert a bfloat16 byte payload returned by pymilvus to floats."""
    bfloat16_array = np.frombuffer(byte_data, dtype=np.uint16)
    float32_as_uint32 = bfloat16_array.astype(np.uint32) << 16
    return float32_as_uint32.view(np.float32).tolist()


def binary_to_int_list(packed_bytes: bytes) -> list[int]:
    """Convert packed binary vector bytes to a list of 0/1 integers."""
    byte_array = np.frombuffer(packed_bytes, dtype=np.uint8)
    return np.unpackbits(byte_array).astype(int).tolist()


def dump_collection(
    db_file: Optional[str],
    collection_name: str,
    path: str,
    *,
    uri: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Dump one collection to local Milvus BulkWriter JSON files.

    Args:
        db_file: local Milvus Lite ``.db`` path.  The path is used as the
            pymilvus URI.  It may be a file for old milvus-lite or a directory
            for the pure-Python engine.
        collection_name: collection to export.
        path: local output directory for BulkWriter files.
        uri: explicit Milvus URI.  When provided, ``db_file`` is ignored.
        batch_size: query iterator batch size.
        chunk_size: BulkWriter chunk size in bytes.

    Returns:
        A small summary dict with collection, path, and row count.
    """
    if not collection_name:
        raise RuntimeError("collection name is required")
    if not path:
        raise RuntimeError("dump path is required")
    if uri is None:
        if not db_file:
            raise RuntimeError("db_file or uri is required")
        db_path = pathlib.Path(db_file)
        if not db_path.exists():
            raise RuntimeError(f"db_file: {db_file} not exists")
        uri = db_file

    out_path = pathlib.Path(path)
    parent = out_path.parent
    if not parent.is_dir():
        raise RuntimeError(
            f"dump path({path})'s parent dir not exists: {parent} not exists"
        )
    if out_path.exists() and not out_path.is_dir():
        raise RuntimeError(f"dump path({path}) exists and is not a directory")

    pymilvus_api = _load_pymilvus_api()
    pymilvus_api.connections.connect(DUMP_ALIAS, uri=uri)
    try:
        if not pymilvus_api.utility.has_collection(
            collection_name, using=DUMP_ALIAS
        ):
            raise RuntimeError(f"Collection: {collection_name} not exists")

        collection = pymilvus_api.Collection(collection_name, using=DUMP_ALIAS)
        # Current Milvus Lite requires loaded collections for query paths.
        collection.load()

        total_rows = _collection_row_count(collection)
        auto_pk_name = (
            collection.primary_field.name
            if getattr(collection.primary_field, "auto_id", False)
            else None
        )
        bfloat16_fields, binary_fields = _special_vector_fields(
            collection.schema.fields,
            pymilvus_api.DataType,
        )

        LocalBulkWriter, BulkFileType = _load_bulk_writer()
        writer = LocalBulkWriter(
            schema=collection.schema,
            local_path=path,
            chunk_size=chunk_size,
            file_type=BulkFileType.JSON,
        )

        dumped = 0
        iterator = collection.query_iterator(
            batch_size=batch_size,
            output_fields=["*"],
        )
        progress = _make_progress(
            total_rows,
            f"Dump collection {collection_name}'s data",
        )
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                for row in rows:
                    _normalize_row(
                        row,
                        auto_pk_name=auto_pk_name,
                        bfloat16_fields=bfloat16_fields,
                        binary_fields=binary_fields,
                    )
                    writer.append_row(_to_plain(row))
                dumped += len(rows)
                progress.update(len(rows))
        finally:
            iterator.close()
            progress.close()

        writer.commit()
        return {
            "collection": collection_name,
            "path": path,
            "row_count": dumped,
        }
    finally:
        try:
            pymilvus_api.connections.disconnect(DUMP_ALIAS)
        except Exception:
            pass


def dump_func(args: argparse.Namespace) -> dict[str, Any]:
    return dump_collection(
        args.db_file,
        args.collection,
        args.path,
        uri=args.uri,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


def server_func(args: argparse.Namespace) -> None:
    run_server(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(prog="milvus-lite")
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    result = args.func(args)
    if isinstance(result, dict) and args.command == "dump":
        print(
            f"Dump collection {result['collection']} success: "
            f"{result['row_count']} rows -> {result['path']}"
        )
    return 0


def build_parser(prog: str = "milvus-lite") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(
        dest="command",
        description="milvus-lite command line tool.",
    )

    add_server_parser(subparsers)
    add_dump_parser(subparsers)
    return parser


def add_server_parser(subparsers) -> argparse.ArgumentParser:
    server_cmd = subparsers.add_parser(
        "server",
        help="start the Milvus Lite gRPC server",
    )
    server_cmd.add_argument(
        "--data-dir",
        required=True,
        help="Milvus Lite data directory (created if it doesn't exist).",
    )
    server_cmd.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"bind host (default: {DEFAULT_HOST})",
    )
    server_cmd.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"bind port (default: {DEFAULT_PORT})",
    )
    server_cmd.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"thread pool size (default: {DEFAULT_MAX_WORKERS})",
    )
    server_cmd.set_defaults(func=server_func)
    return server_cmd


def add_dump_parser(subparsers) -> argparse.ArgumentParser:
    dump_cmd = subparsers.add_parser(
        "dump",
        help="dump a Milvus Lite collection to Milvus bulk JSON files",
    )
    source = dump_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-d",
        "--db-file",
        type=str,
        help="local Milvus Lite .db path",
    )
    source.add_argument(
        "--uri",
        type=str,
        help="Milvus server URI, for example http://127.0.0.1:19530",
    )
    dump_cmd.add_argument(
        "-c",
        "--collection",
        type=str,
        required=True,
        help="collection to dump",
    )
    dump_cmd.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="dump file storage path",
    )
    dump_cmd.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"query iterator batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    dump_cmd.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"BulkWriter chunk size in bytes (default: {DEFAULT_CHUNK_SIZE})",
    )
    dump_cmd.set_defaults(func=dump_func)
    return dump_cmd


def _load_pymilvus_api():
    try:
        from pymilvus import Collection, DataType, connections, utility
    except ImportError as e:
        raise RuntimeError(
            "milvus-lite dump requires pymilvus to already be installed."
        ) from e
    return SimpleNamespace(
        Collection=Collection,
        DataType=DataType,
        connections=connections,
        utility=utility,
    )


def _load_bulk_writer():
    try:
        from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "milvus-lite dump requires pymilvus bulk writer dependencies "
            "to already be installed."
        ) from e
    return LocalBulkWriter, BulkFileType


def _collection_row_count(collection) -> int:
    rows = collection.query("", output_fields=["count(*)"])
    if not rows:
        return 0
    return int(rows[0].get("count(*)", 0))


def _special_vector_fields(
    fields: Iterable[Any],
    data_type,
) -> tuple[list[str], list[str]]:
    bfloat16_type = getattr(data_type, "BFLOAT16_VECTOR", object())
    binary_type = getattr(data_type, "BINARY_VECTOR", object())
    bfloat16_fields = [
        field.name for field in fields
        if getattr(field, "dtype", None) == bfloat16_type
    ]
    binary_fields = [
        field.name for field in fields
        if getattr(field, "dtype", None) == binary_type
    ]
    return bfloat16_fields, binary_fields


def _normalize_row(
    row: dict[str, Any],
    *,
    auto_pk_name: Optional[str],
    bfloat16_fields: Iterable[str],
    binary_fields: Iterable[str],
) -> None:
    if auto_pk_name:
        row.pop(auto_pk_name, None)

    for name in bfloat16_fields:
        if name in row and row[name] is not None:
            row[name] = bfloat16_to_float32(_first_bytes(row[name]))

    for name in binary_fields:
        if name in row and row[name] is not None:
            row[name] = binary_to_int_list(_first_bytes(row[name]))


def _first_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, (list, tuple)) and value:
        return _first_bytes(value[0])
    if isinstance(value, np.ndarray):
        return value.tobytes()
    raise TypeError(f"expected bytes-like vector payload, got {type(value).__name__}")


def _to_plain(value: Any) -> Any:
    """Recursively convert numpy values into JSON/BulkWriter-friendly values."""
    if isinstance(value, dict):
        return {_to_plain(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_progress(total: int, desc: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoopProgress()
    return tqdm(total=total, desc=desc)


class _NoopProgress:
    def update(self, _n: int) -> None:
        return None

    def close(self) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
