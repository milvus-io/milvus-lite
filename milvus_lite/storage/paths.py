"""Helpers for relative paths persisted in manifests and snapshots."""

from __future__ import annotations

import posixpath


def normalize_rel_path(path: str) -> str:
    """Return a persisted relative path using forward slashes."""
    return path.replace("\\", "/")


def persisted_rel_path(*parts: str) -> str:
    """Join components using the platform-independent persisted format."""
    return posixpath.join(*(normalize_rel_path(part) for part in parts))
