"""
Synthetic RAG evaluation dataset for PipelineMind.

Each entry has:
  query:           the user question
  relevant_chunks: list of source_file + function_name that must appear in
                   the top-K results for the answer to be correct
  intent:          expected intent classification
  expected_tables: tables that must appear in a lineage/catalogue answer

Ground truth is derived from the known synthetic data fixtures so it can
be computed deterministically without human annotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalQuery:
    query_id:        str
    query:           str
    intent:          str
    relevant_files:  list[str]          # source_file patterns that must be in top-K
    relevant_terms:  list[str]          # terms that must appear in retrieved docs
    expected_tables: list[str] = field(default_factory=list)
    description:     str = ""


EVAL_DATASET: list[EvalQuery] = [
    # ── CODE_QA queries ───────────────────────────────────────────────────────
    EvalQuery(
        query_id="cq01",
        query="Why does the orders pipeline use a MERGE strategy instead of INSERT OVERWRITE?",
        intent="CODE_QA",
        relevant_files=["orders_pipeline.py"],
        relevant_terms=["MERGE", "upsert", "ON CONFLICT", "watermark"],
        description="Core architecture question about the orders ETL strategy",
    ),
    EvalQuery(
        query_id="cq02",
        query="How does the SCD2 delta detection work in the users pipeline?",
        intent="CODE_QA",
        relevant_files=["users_pipeline.py"],
        relevant_terms=["row_hash", "SCD", "compute_deltas", "is_current", "valid_from"],
        description="SCD2 implementation understanding",
    ),
    EvalQuery(
        query_id="cq03",
        query="What is the session inactivity timeout used in the sessions pipeline?",
        intent="CODE_QA",
        relevant_files=["sessions_pipeline.py"],
        relevant_terms=["SESSION_TIMEOUT_MINUTES", "30", "inactivity", "gap"],
        description="Configuration value embedded in code",
    ),
    EvalQuery(
        query_id="cq04",
        query="How does the inventory pipeline detect low stock SKUs?",
        intent="CODE_QA",
        relevant_files=["inventory_pipeline.py"],
        relevant_terms=["LOW_STOCK_THRESHOLD", "stock_status", "alert", "quantity_on_hand"],
        description="Business logic embedded in pipeline code",
    ),
    EvalQuery(
        query_id="cq05",
        query="What KPI definitions does the metrics pipeline track?",
        intent="CODE_QA",
        relevant_files=["metrics_pipeline.py"],
        relevant_terms=["gmv", "daily_active_users", "conversion_rate", "KPI_DEFINITIONS"],
        description="Configuration constants query",
    ),

    # ── CATALOGUE queries ─────────────────────────────────────────────────────
    EvalQuery(
        query_id="cat01",
        query="What tables does the vw_revenue_by_tier view depend on?",
        intent="CATALOGUE",
        relevant_files=["manifest.json"],
        relevant_terms=["orders_fact", "dim_users", "vw_revenue_by_tier"],
        expected_tables=["orders_fact", "dim_users"],
        description="Lineage DAG query — the original failing query",
    ),
    EvalQuery(
        query_id="cat02",
        query="What PII columns exist in the dim_users table?",
        intent="CATALOGUE",
        relevant_files=["manifest.json", "users_schema.sql"],
        relevant_terms=["email", "phone_number", "date_of_birth", "PII_HIGH"],
        expected_tables=["dim_users"],
        description="PII discovery query",
    ),
    EvalQuery(
        query_id="cat03",
        query="Which tables are downstream of sessions_agg?",
        intent="CATALOGUE",
        relevant_files=["manifest.json"],
        relevant_terms=["kpi_daily_metrics", "vw_daily_funnel", "ml_feature_store"],
        expected_tables=["sessions_agg"],
        description="Downstream lineage traversal",
    ),
    EvalQuery(
        query_id="cat04",
        query="What is the schema of the orders_fact table?",
        intent="CATALOGUE",
        relevant_files=["orders_schema.sql", "manifest.json"],
        relevant_terms=["order_id", "customer_id", "total_amount", "order_date"],
        expected_tables=["orders_fact"],
        description="Schema discovery query",
    ),

    # ── YAML/CONFIG queries ────────────────────────────────────────────────────
    EvalQuery(
        query_id="yq01",
        query="What is the schedule interval for the orders ETL DAG?",
        intent="CODE_QA",
        relevant_files=["orders_dag.yml"],
        relevant_terms=["0 * * * *", "hourly", "schedule_interval"],
        description="DAG configuration value",
    ),
    EvalQuery(
        query_id="yq02",
        query="What SLO target does the users dimension pipeline have?",
        intent="CODE_QA",
        relevant_files=["users_dag.yml"],
        relevant_terms=["99.0", "success_rate_target_pct", "slo"],
        description="SLO configuration lookup",
    ),

    # ── HEALTH queries (agent tools) ──────────────────────────────────────────
    EvalQuery(
        query_id="hq01",
        query="Show me the SLO adherence for the orders pipeline over the last 7 days",
        intent="HEALTH",
        relevant_files=[],
        relevant_terms=["orders", "success_rate", "slo_target", "compliant"],
        description="SLO report tool invocation",
    ),
    EvalQuery(
        query_id="hq02",
        query="Which pipelines had failures in the last 24 hours?",
        intent="HEALTH",
        relevant_files=[],
        relevant_terms=["failed", "error_message", "pipeline_id", "status"],
        description="Pipeline health status query",
    ),
]
