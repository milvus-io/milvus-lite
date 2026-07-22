"""Manifest-based collection snapshots.

Snapshots are lightweight metadata records that pin immutable data, delta,
and index files referenced by a collection manifest at creation time.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from milvus_lite.index.files import parse_index_sidecar_name
from milvus_lite.storage.paths import normalize_rel_path, persisted_rel_path


SNAPSHOTS_DIRNAME = "snapshots"
SNAPSHOT_METADATA_DIRNAME = "metadata"
SNAPSHOT_MANIFESTS_DIRNAME = "manifests"
SNAPSHOT_FORMAT_VERSION = 1

_SNAPSHOT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_snapshot_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"snapshot name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("snapshot name must not be empty")
    if name in {".", ".."} or not _SNAPSHOT_NAME_RE.match(name):
        raise ValueError(
            "snapshot name may only contain letters, digits, dot, underscore, and dash"
        )


def snapshot_root(collection_dir: str) -> str:
    return os.path.join(collection_dir, SNAPSHOTS_DIRNAME)


def metadata_dir(collection_dir: str) -> str:
    return os.path.join(snapshot_root(collection_dir), SNAPSHOT_METADATA_DIRNAME)


def manifests_dir(collection_dir: str) -> str:
    return os.path.join(snapshot_root(collection_dir), SNAPSHOT_MANIFESTS_DIRNAME)


def metadata_path(collection_dir: str, snapshot_name: str) -> str:
    validate_snapshot_name(snapshot_name)
    return os.path.join(metadata_dir(collection_dir), f"{snapshot_name}.json")


def manifest_dir(collection_dir: str, snapshot_name: str) -> str:
    validate_snapshot_name(snapshot_name)
    return os.path.join(manifests_dir(collection_dir), snapshot_name)


def create_snapshot(
    collection_dir: str,
    collection_name: str,
    snapshot_name: str,
    description: str,
    schema_filename: str,
    manifest_filename: str,
    data_files: Dict[str, List[str]],
    delta_files: Dict[str, List[str]],
    index_files: Dict[str, List[str]],
    current_seq: int,
    manifest_version: int,
) -> Dict[str, Any]:
    validate_snapshot_name(snapshot_name)
    meta_path = metadata_path(collection_dir, snapshot_name)
    if os.path.exists(meta_path):
        raise FileExistsError(f"snapshot {snapshot_name!r} already exists")

    snap_manifest_dir = manifest_dir(collection_dir, snapshot_name)
    tmp_path = meta_path + ".tmp"
    try:
        os.makedirs(metadata_dir(collection_dir), exist_ok=True)
        if os.path.isdir(snap_manifest_dir):
            shutil.rmtree(snap_manifest_dir, ignore_errors=False)
        os.makedirs(snap_manifest_dir, exist_ok=False)

        schema_src = os.path.join(collection_dir, schema_filename)
        manifest_src = os.path.join(collection_dir, manifest_filename)
        schema_dst = os.path.join(snap_manifest_dir, schema_filename)
        manifest_dst = os.path.join(snap_manifest_dir, manifest_filename)
        shutil.copy2(schema_src, schema_dst)
        shutil.copy2(manifest_src, manifest_dst)
        _fsync_file(schema_dst)
        _fsync_file(manifest_dst)
        _fsync_dir(snap_manifest_dir)

        created_at = datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "id": uuid.uuid4().hex,
            "name": snapshot_name,
            "description": description,
            "collection_name": collection_name,
            "created_at": created_at,
            "created_seq": int(current_seq),
            "manifest_version": int(manifest_version),
            "schema_file": persisted_rel_path(
                SNAPSHOTS_DIRNAME, SNAPSHOT_MANIFESTS_DIRNAME, snapshot_name, schema_filename
            ),
            "manifest_file": persisted_rel_path(
                SNAPSHOTS_DIRNAME, SNAPSHOT_MANIFESTS_DIRNAME, snapshot_name, manifest_filename
            ),
            "data_files": _copy_file_map(data_files),
            "delta_files": _copy_file_map(delta_files),
            "index_files": _copy_file_map(index_files),
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
        _fsync_dir(os.path.dirname(meta_path))
        return payload
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if os.path.isdir(snap_manifest_dir):
            shutil.rmtree(snap_manifest_dir, ignore_errors=True)
        raise


def load_snapshot(collection_dir: str, snapshot_name: str) -> Dict[str, Any]:
    path = metadata_path(collection_dir, snapshot_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"snapshot metadata {path!r} must be an object")
    _normalize_snapshot_paths(data)
    validate_snapshot_metadata(data)
    return data


def validate_snapshot_metadata(data: Dict[str, Any]) -> None:
    schema_file = data.get("schema_file")
    manifest_file = data.get("manifest_file")
    if not isinstance(schema_file, str) or not isinstance(manifest_file, str):
        raise ValueError("snapshot metadata must contain schema_file and manifest_file")
    _validate_rel_path(schema_file, allowed_prefixes=(SNAPSHOTS_DIRNAME,))
    _validate_rel_path(manifest_file, allowed_prefixes=(SNAPSHOTS_DIRNAME,))

    for key, allowed_prefix in (
        ("data_files", "data"),
        ("delta_files", "delta"),
        ("index_files", "indexes"),
    ):
        file_map = data.get(key, {})
        if not isinstance(file_map, dict):
            raise ValueError(f"snapshot metadata {key!r} must be an object")
        for partition, rels in file_map.items():
            if not isinstance(partition, str) or not partition:
                raise ValueError(f"snapshot metadata {key!r} has invalid partition")
            _validate_path_component(partition, "partition")
            if not isinstance(rels, list):
                raise ValueError(
                    f"snapshot metadata {key!r} for partition {partition!r} must be a list"
                )
            for rel in rels:
                if not isinstance(rel, str):
                    raise ValueError(
                        f"snapshot metadata {key!r} for partition {partition!r} has non-string path"
                    )
                _validate_rel_path(rel, allowed_prefixes=(allowed_prefix,))


def list_snapshots(collection_dir: str) -> List[Dict[str, Any]]:
    root = metadata_dir(collection_dir)
    if not os.path.isdir(root):
        return []
    out: List[Dict[str, Any]] = []
    for entry in os.listdir(root):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(root, entry)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _normalize_snapshot_paths(data)
                out.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda d: (str(d.get("created_at", "")), str(d.get("name", ""))))
    return out


def drop_snapshot(collection_dir: str, snapshot_name: str) -> None:
    meta_path = metadata_path(collection_dir, snapshot_name)
    snap_manifest_dir = manifest_dir(collection_dir, snapshot_name)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    if os.path.isdir(snap_manifest_dir):
        shutil.rmtree(snap_manifest_dir, ignore_errors=False)


def snapshot_references(
    collection_dir: str,
) -> Tuple[Dict[str, set[str]], Dict[str, set[str]], Dict[str, set[str]]]:
    data_refs: Dict[str, set[str]] = {}
    delta_refs: Dict[str, set[str]] = {}
    index_refs: Dict[str, set[str]] = {}
    for snap in list_snapshots(collection_dir):
        _merge_refs(data_refs, snap.get("data_files"))
        _merge_refs(delta_refs, snap.get("delta_files"))
        _merge_refs(index_refs, snap.get("index_files"))
    return data_refs, delta_refs, index_refs


def collect_index_files(
    collection_dir: str,
    data_files: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for partition, files in data_files.items():
        stems = {os.path.splitext(os.path.basename(rel))[0] for rel in files}
        index_dir = os.path.join(collection_dir, "partitions", partition, "indexes")
        if not os.path.isdir(index_dir):
            continue
        rels: List[str] = []
        for entry in os.listdir(index_dir):
            sidecar = parse_index_sidecar_name(entry)
            if sidecar is None:
                continue
            if sidecar.source_stem in stems:
                rels.append(persisted_rel_path("indexes", entry))
        if rels:
            out[partition] = sorted(rels)
    return out


def _validate_rel_path(path: str, allowed_prefixes: Tuple[str, ...]) -> None:
    if not path or path.startswith("/"):
        raise ValueError(f"snapshot path must be relative: {path!r}")
    normalized = posixpath.normpath(path)
    if normalized != path or normalized in {".", ".."}:
        raise ValueError(f"snapshot path is not normalized: {path!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"snapshot path contains invalid component: {path!r}")
    if parts[0] not in allowed_prefixes:
        raise ValueError(
            f"snapshot path {path!r} must start with one of {allowed_prefixes!r}"
        )


def _validate_path_component(value: str, label: str) -> None:
    if value in {".", ".."} or os.path.isabs(value):
        raise ValueError(f"snapshot {label} is invalid: {value!r}")
    if os.path.normpath(value) != value or "/" in value or "\\" in value:
        raise ValueError(f"snapshot {label} is invalid: {value!r}")


def _fsync_file(path: str) -> None:
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    try:
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _copy_file_map(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        str(k): [normalize_rel_path(rel) for rel in v]
        for k, v in values.items()
        if v
    }


def _normalize_snapshot_paths(data: Dict[str, Any]) -> None:
    for key in ("schema_file", "manifest_file"):
        value = data.get(key)
        if isinstance(value, str):
            data[key] = normalize_rel_path(value)

    for key in ("data_files", "delta_files", "index_files"):
        values = data.get(key)
        if isinstance(values, dict):
            data[key] = _copy_file_map(values)


def _merge_refs(target: Dict[str, set[str]], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for partition, rels in values.items():
        if not isinstance(partition, str) or not isinstance(rels, list):
            continue
        bucket = target.setdefault(partition, set())
        for rel in rels:
            if isinstance(rel, str):
                bucket.add(normalize_rel_path(rel))
