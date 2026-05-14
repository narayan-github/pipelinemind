"""Integration tests for discovery tools — require seeded DuckDB."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _check_db():
    from pm_config import settings
    if not settings.duckdb_path.exists():
        pytest.skip("DuckDB not seeded")


def test_list_catalogue_tables_returns_results():
    from agent.tools.discovery_tools import list_catalogue_tables
    result = list_catalogue_tables()
    assert "tables" in result
    assert result["total_count"] > 0
    names = [t["table_name"] for t in result["tables"]]
    assert "orders_fact" in names


def test_list_catalogue_tables_domain_filter():
    from agent.tools.discovery_tools import list_catalogue_tables
    result = list_catalogue_tables(domain_filter="finance")
    for t in result["tables"]:
        assert t["domain"] == "finance"


def test_list_pipeline_ids_returns_valid_ids():
    from agent.tools.discovery_tools import list_pipeline_ids
    result = list_pipeline_ids()
    assert "valid_ids" in result
    assert len(result["valid_ids"]) > 0
    expected = {"orders", "users", "inventory", "sessions", "metrics"}
    actual   = set(result["valid_ids"])
    assert expected.issubset(actual), f"Missing pipeline IDs: {expected - actual}"


def test_list_pipeline_ids_no_inventory_snapshot_pipeline():
    """The class name InventorySnapshotPipeline must NOT appear as a pipeline ID."""
    from agent.tools.discovery_tools import list_pipeline_ids
    result = list_pipeline_ids()
    class_names = [i for i in result["valid_ids"] if "Pipeline" in i or i[0].isupper()]
    assert not class_names, f"Class names leaked into pipeline IDs: {class_names}"
