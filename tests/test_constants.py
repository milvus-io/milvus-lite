"""Flush budget defaults and environment overrides without module cache changes."""

import runpy

import pytest

from milvus_lite import constants


MEMORY_ENV = "MILVUS_LITE_MEMTABLE_MEMORY_LIMIT_BYTES"
WAL_ENV = "MILVUS_LITE_WAL_SIZE_LIMIT_BYTES"


@pytest.fixture(autouse=True)
def clear_budget_env(monkeypatch):
    monkeypatch.delenv(MEMORY_ENV, raising=False)
    monkeypatch.delenv(WAL_ENV, raising=False)


def test_default_flush_budgets():
    values = runpy.run_path(constants.__file__)
    assert values["MEMTABLE_MEMORY_LIMIT"] == 128 * 1024 * 1024
    assert values["WAL_SIZE_LIMIT"] == 256 * 1024 * 1024


@pytest.mark.parametrize("env_name,constant", [
    (MEMORY_ENV, "MEMTABLE_MEMORY_LIMIT"),
    (WAL_ENV, "WAL_SIZE_LIMIT"),
])
def test_flush_budget_environment_override(monkeypatch, env_name, constant):
    monkeypatch.setenv(env_name, "1048576")
    values = runpy.run_path(constants.__file__)
    assert values[constant] == 1048576
    other = "WAL_SIZE_LIMIT" if env_name == MEMORY_ENV else "MEMTABLE_MEMORY_LIMIT"
    assert values[other] == (256 if env_name == MEMORY_ENV else 128) * 1024 * 1024


@pytest.mark.parametrize("env_name", [MEMORY_ENV, WAL_ENV])
@pytest.mark.parametrize("value", ["0", "-1", "", "1.5", "128MiB"])
def test_invalid_flush_budget_fails_with_setting_name(monkeypatch, env_name, value):
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=f"{env_name} must be a positive integer in bytes"):
        runpy.run_path(constants.__file__)
