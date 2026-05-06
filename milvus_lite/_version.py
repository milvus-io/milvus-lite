"""Version helpers.

``pyproject.toml`` is the single source of truth. In a source checkout we
read it directly so version changes take effect without reinstalling the
editable package; in an installed wheel we fall back to package metadata.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_DISTRIBUTION_NAME = "milvus-lite"


def _version_from_pyproject() -> str | None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return None
    with pyproject.open("rb") as fp:
        data = tomllib.load(fp)
    version = data.get("project", {}).get("version")
    return str(version) if version else None


def get_version() -> str:
    version = _version_from_pyproject()
    if version is not None:
        return version
    return metadata.version(_DISTRIBUTION_NAME)
