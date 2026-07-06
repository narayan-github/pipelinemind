"""Tests for tool argument type coercion."""
from __future__ import annotations

import pytest
from agent.agent_loop import _coerce_args


def test_depth_string_to_int():
    result = _coerce_args("get_lineage_graph", {"table_name": "orders", "depth": "1"})
    assert result["depth"] == 1
    assert isinstance(result["depth"], int)


def test_depth_already_int_unchanged():
    result = _coerce_args("get_lineage_graph", {"table_name": "orders", "depth": 2})
    assert result["depth"] == 2


def test_lookback_hours_string_to_int():
    result = _coerce_args("get_pipeline_status", {"pipeline_id": "orders", "lookback_hours": "48"})
    assert result["lookback_hours"] == 48
    assert isinstance(result["lookback_hours"], int)


def test_window_days_string_to_int():
    result = _coerce_args("get_slo_report", {"pipeline_id": "orders", "window_days": "7"})
    assert result["window_days"] == 7


def test_tool_with_no_coercion_unchanged():
    args = {"table_name": "dim_users", "dropped_columns": ["user_id"]}
    result = _coerce_args("analyze_lineage_impact", args)
    assert result == args


def test_unknown_tool_passthrough():
    args = {"foo": "bar"}
    result = _coerce_args("nonexistent_tool", args)
    assert result == args
