"""Helpers for index sidecar filenames."""

from __future__ import annotations

import os
from typing import NamedTuple, Optional


VECTOR_INDEX_FILE_EXT = ".idx"
SCALAR_INDEX_FILE_EXT = ".sidx"
INDEX_FILE_EXTENSIONS = (VECTOR_INDEX_FILE_EXT, SCALAR_INDEX_FILE_EXT)


class IndexSidecarName(NamedTuple):
    source_stem: str
    field_name: str
    index_type: str
    extension: str


def index_sidecar_suffix(field_name: str, index_type: str, *, scalar: bool) -> str:
    ext = SCALAR_INDEX_FILE_EXT if scalar else VECTOR_INDEX_FILE_EXT
    return f".{field_name}.{index_type.lower()}{ext}"


def index_sidecar_path(
    index_dir: str,
    source_file_path: str,
    field_name: str,
    index_type: str,
    *,
    scalar: bool,
) -> str:
    source_stem = os.path.splitext(os.path.basename(source_file_path))[0]
    return os.path.join(
        index_dir,
        f"{source_stem}{index_sidecar_suffix(field_name, index_type, scalar=scalar)}",
    )


def parse_index_sidecar_name(filename: str) -> Optional[IndexSidecarName]:
    ext = next((suffix for suffix in INDEX_FILE_EXTENSIONS if filename.endswith(suffix)), None)
    if ext is None:
        return None
    base = filename[: -len(ext)]
    stem_field, sep, index_type = base.rpartition(".")
    if not sep or not index_type:
        return None
    source_stem, sep, field_name = stem_field.rpartition(".")
    if not sep or not source_stem or not field_name:
        return None
    return IndexSidecarName(source_stem, field_name, index_type, ext)
