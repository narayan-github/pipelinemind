"""Verify intent allowlist includes discovery tools."""
from __future__ import annotations

from agent.agent_loop import INTENT_TOOL_ALLOWLIST, INTENT_MAX_ITERATIONS


def test_catalogue_includes_list_tables():
    assert "list_catalogue_tables" in INTENT_TOOL_ALLOWLIST["CATALOGUE"]


def test_health_includes_list_pipelines():
    assert "list_pipeline_ids" in INTENT_TOOL_ALLOWLIST["HEALTH"]


def test_catalogue_max_iters_is_two():
    assert INTENT_MAX_ITERATIONS["CATALOGUE"] == 2


def test_health_max_iters_is_two():
    assert INTENT_MAX_ITERATIONS["HEALTH"] == 2


def test_action_has_all_tools_including_discovery():
    allowed = INTENT_TOOL_ALLOWLIST["ACTION"]
    assert "list_catalogue_tables" in allowed
    assert "list_pipeline_ids" in allowed
