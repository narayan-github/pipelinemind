"""
Discovery tools: list available tables and pipeline IDs.
These tools prevent the agent from hallucinating table names or pipeline IDs
by giving it the real names from DuckDB before calling other tools.
"""
from __future__ import annotations

import logging
from typing import Any

import duckdb

from pm_config import settings

logger = logging.getLogger(__name__)


def list_catalogue_tables(domain_filter: str | None = None) -> dict[str, Any]:
    """
    List all table names in the data catalogue.
    Call this FIRST when the user mentions a table by a general description
    (e.g., 'fact table', 'users table') to resolve the exact table name
    before calling get_lineage_graph or analyze_lineage_impact.
    """
    logger.info("list_catalogue_tables | domain_filter=%s", domain_filter)
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        if domain_filter:
            rows = con.execute(
                "SELECT table_name, domain, description, pii_flag, row_count "
                "FROM catalogue_tables WHERE domain = ? ORDER BY table_name",
                [domain_filter],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT table_name, domain, description, pii_flag, row_count "
                "FROM catalogue_tables ORDER BY table_name"
            ).fetchall()
        tables = [
            {
                "table_name":  r[0],
                "domain":      r[1],
                "description": r[2],
                "pii_flag":    r[3],
                "row_count":   r[4],
            }
            for r in rows
        ]
        return {
            "tables":       tables,
            "total_count":  len(tables),
            "domain_filter": domain_filter,
        }
    finally:
        con.close()


def list_pipeline_ids() -> dict[str, Any]:
    """
    List all valid pipeline IDs from the pipeline_runs table.
    Call this FIRST when the user asks about pipeline health without
    specifying a pipeline name, to avoid guessing pipeline IDs.
    """
    logger.info("list_pipeline_ids")
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                pipeline_id,
                COUNT(*) AS total_runs,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                MAX(start_time) AS last_run,
                LAST_VALUE(status) OVER (
                    PARTITION BY pipeline_id
                    ORDER BY start_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS last_status
            FROM pipeline_runs
            GROUP BY pipeline_id
            ORDER BY pipeline_id
            """
        ).fetchall()
        pipelines = []
        for r in rows:
            total = r[1] or 1
            pipelines.append({
                "pipeline_id":   r[0],
                "total_runs":    r[1],
                "success_rate":  round((r[2] or 0) / total * 100, 1),
                "last_run":      str(r[3]) if r[3] else None,
                "last_status":   r[4],
            })
        return {
            "pipelines":    pipelines,
            "valid_ids":    [p["pipeline_id"] for p in pipelines],
            "total_count":  len(pipelines),
        }
    except Exception as exc:
        # DuckDB LAST_VALUE window may not work on all versions — fallback
        try:
            rows2 = con.execute(
                "SELECT DISTINCT pipeline_id FROM pipeline_runs ORDER BY pipeline_id"
            ).fetchall()
            ids = [r[0] for r in rows2]
            return {"pipelines": [{"pipeline_id": i} for i in ids], "valid_ids": ids}
        except Exception:
            return {"error": str(exc), "valid_ids": []}
    finally:
        con.close()
