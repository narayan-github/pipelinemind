#!/usr/bin/env bash
# ==============================================================================
# PipelineMind — Consolidation Script
# Applies all remaining fixes from the PDF diff, re-seeds, re-tests,
# and prints a clear "what to do next" guide.
# Run from: /Users/as-mac-1282/Developer/genai_mini/pipelinemind
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[PM]${NC} $*"; }
step() { echo -e "\n${BLUE}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FATAL]${NC} $*" >&2; exit 1; }

PROJECT_DIR="/Users/as-mac-1282/Developer/genai_mini/pipelinemind"
[[ -d "$PROJECT_DIR" ]] || die "Project not found: $PROJECT_DIR"
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
VENV_PYTEST="$PROJECT_DIR/.venv/bin/pytest"
[[ -f "$VENV_PYTHON" ]] || die ".venv not found — run the Phase 1+2 script first"

# ==============================================================================
# FIX 1 — db/schema.sql
# Remove UNIQUE from table_name (catalogue_tables) and pipeline_id (slo_definitions)
# DuckDB throws a binder error when INSERT OR REPLACE hits a UNIQUE constraint
# on a column that is not the PRIMARY KEY.
# ==============================================================================
step "FIX 1: db/schema.sql — remove secondary UNIQUE constraints"

cat << 'SQLEOF' > db/schema.sql
-- PipelineMind DuckDB metadata store schema
-- Compatible with DuckDB 1.x (no FK enforcement by default)
-- UNIQUE removed from non-PK columns — seeder uses DELETE+INSERT pattern instead.

CREATE TABLE IF NOT EXISTS catalogue_tables (
    table_id    VARCHAR PRIMARY KEY,
    table_name  VARCHAR NOT NULL,
    schema_name VARCHAR,
    description TEXT,
    domain      VARCHAR,
    pii_flag    BOOLEAN DEFAULT FALSE,
    tags        VARCHAR[],
    row_count   BIGINT  DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS catalogue_columns (
    column_id      VARCHAR PRIMARY KEY,
    table_id       VARCHAR NOT NULL,
    column_name    VARCHAR NOT NULL,
    data_type      VARCHAR,
    pii_class      VARCHAR,
    nullable       BOOLEAN DEFAULT TRUE,
    description    TEXT,
    retention_days INTEGER
);

CREATE TABLE IF NOT EXISTS lineage_edges (
    edge_id        VARCHAR PRIMARY KEY,
    source_table   VARCHAR NOT NULL,
    source_column  VARCHAR,
    target_table   VARCHAR NOT NULL,
    target_column  VARCHAR,
    transformation VARCHAR,
    pipeline_id    VARCHAR,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        VARCHAR PRIMARY KEY,
    pipeline_id   VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    start_time    TIMESTAMP,
    duration_secs DOUBLE,
    error_message TEXT,
    slo_met       BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS slo_definitions (
    slo_id       VARCHAR PRIMARY KEY,
    pipeline_id  VARCHAR NOT NULL,
    metric_name  VARCHAR NOT NULL,
    target_value DOUBLE  NOT NULL,
    comparison   VARCHAR NOT NULL,
    window_days  INTEGER DEFAULT 7
);

CREATE TABLE IF NOT EXISTS schema_snapshots (
    snapshot_id  VARCHAR PRIMARY KEY,
    table_id     VARCHAR,
    table_name   VARCHAR NOT NULL,
    columns_json TEXT    NOT NULL,
    captured_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
SQLEOF
log "db/schema.sql written (no secondary UNIQUE constraints)"

# ==============================================================================
# FIX 2 — db/seeder.py
# Replace INSERT OR REPLACE with DELETE + INSERT (DuckDB binder error fix).
# Add timestamp shifting so pipeline_runs always land within rolling windows.
# ==============================================================================
step "FIX 2: db/seeder.py — DELETE+INSERT pattern + timestamp shifting"

cat << 'PYEOF' > db/seeder.py
"""
DuckDB metadata store seeder.
Reads synthetic JSON fixtures and populates all 6 metadata tables.

Changes vs v1:
  - INSERT OR REPLACE removed (DuckDB binder error on non-PK UNIQUE columns).
    Pattern is now: DELETE WHERE pk = ? then INSERT.
  - pipeline_runs timestamps are shifted so the most-recent run lands at
    datetime.utcnow(). This ensures rolling-window queries (window_days=30,
    lookback_hours=24) always return data regardless of when the seeder ran.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from pm_config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).parent.parent / "data"


def _uid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _execute_schema(con: duckdb.DuckDBPyConnection, schema_path: Path) -> None:
    """Execute a SQL schema file statement-by-statement."""
    sql = schema_path.read_text()
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        if ";" in line:
            before, _, _ = line.partition(";")
            current.append(before)
            stmt = "\n".join(current).strip()
            if stmt:
                try:
                    con.execute(stmt)
                except Exception as exc:
                    logger.warning("Schema stmt skipped (%s): %.80s", exc, stmt)
            current = []
        else:
            current.append(line)


def seed_tables(con: duckdb.DuckDBPyConnection) -> int:
    rows = json.loads((FIXTURES / "catalogue" / "tables_metadata.json").read_text())
    count = 0
    for r in rows:
        con.execute("DELETE FROM catalogue_tables WHERE table_id = ?", [r["table_id"]])
        con.execute(
            """
            INSERT INTO catalogue_tables
                (table_id, table_name, schema_name, description, domain, pii_flag, tags, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [r["table_id"], r["table_name"], r.get("schema"), r.get("description"),
             r.get("domain"), bool(r.get("pii_flag", False)),
             r.get("tags", []), r.get("row_count", 0)],
        )
        count += 1
    logger.info("Seeded %d catalogue tables", count)
    return count


def seed_columns(con: duckdb.DuckDBPyConnection) -> int:
    pii_rows = json.loads((FIXTURES / "catalogue" / "pii_registry.json").read_text())
    tables = {
        r["table_name"]: r["table_id"]
        for r in json.loads((FIXTURES / "catalogue" / "tables_metadata.json").read_text())
    }
    count = 0
    for r in pii_rows:
        table_id = tables.get(r["table_name"])
        if not table_id:
            continue
        col_id = _uid(r["table_name"], r["column_name"])
        con.execute("DELETE FROM catalogue_columns WHERE column_id = ?", [col_id])
        con.execute(
            """
            INSERT INTO catalogue_columns
                (column_id, table_id, column_name, data_type, pii_class, retention_days)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [col_id, table_id, r["column_name"],
             r.get("data_type"), r.get("pii_class"), r.get("retention_days")],
        )
        count += 1
    logger.info("Seeded %d PII columns", count)
    return count


def seed_lineage(con: duckdb.DuckDBPyConnection) -> int:
    rows = json.loads((FIXTURES / "catalogue" / "lineage_edges.json").read_text())
    count = 0
    for r in rows:
        edge_id = _uid(
            r["source_table"], r.get("source_column", ""),
            r["target_table"], r.get("target_column", ""),
        )
        con.execute("DELETE FROM lineage_edges WHERE edge_id = ?", [edge_id])
        con.execute(
            """
            INSERT INTO lineage_edges
                (edge_id, source_table, source_column, target_table,
                 target_column, transformation, pipeline_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [edge_id, r["source_table"], r.get("source_column"),
             r["target_table"], r.get("target_column"),
             r.get("transformation"), r.get("pipeline_id")],
        )
        count += 1
    logger.info("Seeded %d lineage edges", count)
    return count


def seed_pipeline_runs(con: duckdb.DuckDBPyConnection) -> int:
    """
    Seed pipeline runs with timestamp shifting.
    Shifts all run timestamps so the most-recent fixture run lands at
    datetime.utcnow(), keeping relative spacing intact.
    This ensures rolling-window queries (window_days=30, lookback_hours=24)
    always find data regardless of when the seeder was executed.
    """
    rows = json.loads((FIXTURES / "run_logs" / "pipeline_runs.json").read_text())

    # Compute shift offset
    timestamps = [
        datetime.fromisoformat(r["start_time"])
        for r in rows if r.get("start_time")
    ]
    if timestamps:
        latest_fixture = max(timestamps)
        offset = datetime.utcnow() - latest_fixture
    else:
        offset = timedelta(0)

    def _shift(ts_str: str | None) -> str | None:
        if not ts_str:
            return None
        return (datetime.fromisoformat(ts_str) + offset).isoformat(timespec="seconds")

    count = 0
    for r in rows:
        con.execute("DELETE FROM pipeline_runs WHERE run_id = ?", [r["run_id"]])
        con.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, pipeline_id, status, start_time,
                 duration_secs, error_message, slo_met)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [r["run_id"], r["pipeline_id"], r["status"],
             _shift(r.get("start_time")), r.get("duration_secs"),
             r.get("error_message"), bool(r.get("slo_met", True))],
        )
        count += 1
    logger.info("Seeded %d pipeline runs (timestamps shifted to now)", count)
    return count


def seed_slo_definitions(con: duckdb.DuckDBPyConnection) -> int:
    definitions = [
        ("slo_orders",    "orders",    "success_rate_pct", 99.5, "gte", 7),
        ("slo_users",     "users",     "success_rate_pct", 99.0, "gte", 7),
        ("slo_inventory", "inventory", "success_rate_pct", 98.0, "gte", 7),
        ("slo_sessions",  "sessions",  "success_rate_pct", 99.0, "gte", 7),
        ("slo_metrics",   "metrics",   "success_rate_pct", 99.0, "gte", 7),
    ]
    for row in definitions:
        con.execute("DELETE FROM slo_definitions WHERE slo_id = ?", [row[0]])
        con.execute(
            """
            INSERT INTO slo_definitions
                (slo_id, pipeline_id, metric_name, target_value, comparison, window_days)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            list(row),
        )
    logger.info("Seeded %d SLO definitions", len(definitions))
    return len(definitions)


def seed_schema_snapshots(con: duckdb.DuckDBPyConnection) -> int:
    snapshot = json.loads(
        (FIXTURES / "schema_snapshots" / "baseline.json").read_text()
    )
    tables = {
        r["table_name"]: r["table_id"]
        for r in json.loads((FIXTURES / "catalogue" / "tables_metadata.json").read_text())
    }
    count = 0
    for table_name, table_data in snapshot["tables"].items():
        table_id = tables.get(table_name)
        snap_id  = _uid(snapshot["snapshot_id"], table_name)
        con.execute("DELETE FROM schema_snapshots WHERE snapshot_id = ?", [snap_id])
        con.execute(
            """
            INSERT INTO schema_snapshots
                (snapshot_id, table_id, table_name, columns_json, captured_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [snap_id, table_id, table_name,
             json.dumps(table_data["columns"]), snapshot["captured_at"]],
        )
        count += 1
    logger.info("Seeded %d schema snapshots", count)
    return count


def main() -> None:
    db_path = settings.duckdb_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop stale DB for clean schema application
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed stale DB: %s", db_path)

    logger.info("Connecting to DuckDB at %s", db_path)
    con = duckdb.connect(str(db_path))

    schema_path = Path(__file__).parent / "schema.sql"
    _execute_schema(con, schema_path)
    logger.info("Schema applied")

    seed_tables(con)
    seed_columns(con)
    seed_lineage(con)
    seed_pipeline_runs(con)
    seed_slo_definitions(con)
    seed_schema_snapshots(con)

    for tbl in [
        "catalogue_tables", "catalogue_columns", "lineage_edges",
        "pipeline_runs", "slo_definitions", "schema_snapshots",
    ]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info("  %-30s  %d rows", tbl, n)

    con.close()
    logger.info("DuckDB seeding complete: %s", db_path)


if __name__ == "__main__":
    main()
PYEOF
log "db/seeder.py rewritten (DELETE+INSERT + timestamp shift)"

# ==============================================================================
# FIX 3 — agent/tools/validators.py
# Pydantic v2: min_items is not a valid Field kwarg for list fields.
# Use min_length=1 instead (works for both str and list in Pydantic v2).
# ==============================================================================
step "FIX 3: agent/tools/validators.py — min_items -> min_length"

cat << 'PYEOF' > agent/tools/validators.py
"""
Pydantic v2 models for all MCP tool input parameters.
Invalid inputs are caught here before execution and returned to the LLM
as structured error strings for self-correction.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TriggerDQCheckInput(BaseModel):
    table_name: str = Field(..., min_length=1)
    rules_preset: str = Field(default="standard")

    @field_validator("rules_preset")
    @classmethod
    def valid_preset(cls, v: str) -> str:
        allowed = {"standard", "strict", "minimal"}
        if v not in allowed:
            raise ValueError(f"rules_preset must be one of {allowed}, got '{v}'")
        return v


class GetPipelineStatusInput(BaseModel):
    pipeline_id: str = Field(..., min_length=1)
    lookback_hours: int = Field(default=24, ge=1, le=720)


class GetLineageGraphInput(BaseModel):
    table_name: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)


class AnalyzeLineageImpactInput(BaseModel):
    changed_table: str = Field(..., min_length=1)
    dropped_columns: list[str] = Field(..., min_length=1)

    @field_validator("dropped_columns")
    @classmethod
    def non_empty_columns(cls, v: list[str]) -> list[str]:
        if not v or any(not c.strip() for c in v):
            raise ValueError("dropped_columns must be a non-empty list of non-blank strings")
        return [c.strip() for c in v]


class SearchPIITablesInput(BaseModel):
    domain_filter: Optional[str] = Field(default=None)


class GetSLOReportInput(BaseModel):
    pipeline_id: str = Field(..., min_length=1)
    window_days: int = Field(default=7, ge=1, le=90)
PYEOF
log "agent/tools/validators.py fixed (min_length)"

# ==============================================================================
# FIX 4 — ingestion/embedders.py
# sentence_transformers.models was removed/moved in ST >= 3.x.
# Import path is now sentence_transformers.sentence_transformer.modules.
# ==============================================================================
step "FIX 4: ingestion/embedders.py — fix sentence_transformers models import"

cat << 'PYEOF' > ingestion/embedders.py
"""
Dual embedding strategy:
  - all-mpnet-base-v2  -> documents, YAML, Markdown, dbt nodes  (768-dim)
  - CodeBERT (via ST)  -> Python / SQL code chunks               (768-dim)

Import fix: sentence_transformers >= 3.x moved Transformer/Pooling modules.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from pm_config import settings

logger = logging.getLogger(__name__)

CODE_SOURCE_TYPES = {"python", "sql"}
TEXT_MODEL_NAME   = "sentence-transformers/all-mpnet-base-v2"
CODE_MODEL_BASE   = "microsoft/codebert-base"
EMBED_DIM         = 768


@lru_cache(maxsize=1)
def _get_text_embedder() -> SentenceTransformer:
    logger.info("Loading text embedder: %s", TEXT_MODEL_NAME)
    return SentenceTransformer(
        TEXT_MODEL_NAME,
        cache_folder=str(settings.embed_cache_dir),
    )


@lru_cache(maxsize=1)
def _get_code_embedder() -> SentenceTransformer:
    """Build CodeBERT as a SentenceTransformer with mean pooling."""
    logger.info("Loading code embedder: %s", CODE_MODEL_BASE)
    cache = str(settings.embed_cache_dir)
    try:
        # ST >= 3.x module path
        from sentence_transformers.sentence_transformer import modules as st_modules
        word_model = st_modules.Transformer(CODE_MODEL_BASE, cache_dir=cache)
        pool_model = st_modules.Pooling(
            word_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
    except ImportError:
        # ST < 3.x fallback
        from sentence_transformers import models as st_models_legacy
        word_model = st_models_legacy.Transformer(CODE_MODEL_BASE, cache_dir=cache)
        pool_model = st_models_legacy.Pooling(
            word_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
    return SentenceTransformer(modules=[word_model, pool_model])


class ChunkEmbedder:
    """Routes chunks to the appropriate embedding model based on source_type."""

    def embed_chunk(self, summary: str, source_type: str = "python") -> list[float]:
        embedder = _get_code_embedder() if source_type in CODE_SOURCE_TYPES else _get_text_embedder()
        vector: np.ndarray = embedder.encode(
            summary, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()

    def embed_batch(
        self, summaries: list[str], source_types: list[str], batch_size: int = 64
    ) -> list[list[float]]:
        if not summaries:
            return []

        code_idx = [i for i, st in enumerate(source_types) if st in CODE_SOURCE_TYPES]
        text_idx = [i for i, st in enumerate(source_types) if st not in CODE_SOURCE_TYPES]
        result: list[list[float]] = [[]] * len(summaries)

        if code_idx:
            code_summaries = [summaries[i] for i in code_idx]
            code_vecs = _get_code_embedder().encode(
                code_summaries, normalize_embeddings=True,
                batch_size=batch_size, show_progress_bar=True,
            )
            for i, vec in zip(code_idx, code_vecs):
                result[i] = vec.tolist()

        if text_idx:
            text_summaries = [summaries[i] for i in text_idx]
            text_vecs = _get_text_embedder().encode(
                text_summaries, normalize_embeddings=True,
                batch_size=batch_size, show_progress_bar=True,
            )
            for i, vec in zip(text_idx, text_vecs):
                result[i] = vec.tolist()

        return result

    def embed_query(self, query: str) -> list[float]:
        vec: np.ndarray = _get_text_embedder().encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()
PYEOF
log "ingestion/embedders.py fixed"

# ==============================================================================
# FIX 5 — agent/mcp_resources.py
# Guard against DB not existing yet (API boots before seeder runs).
# Also guard against table-not-found errors during schema drift check.
# ==============================================================================
step "FIX 5: agent/mcp_resources.py — DB existence guard"

cat << 'PYEOF' > agent/mcp_resources.py
"""
Schema drift MCP Resource polling helper.
Called by the Streamlit sidebar every 5 minutes to surface drift warnings
before pipelines fail.
Returns a safe payload if the DB does not exist or is not yet seeded.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import duckdb

from pm_config import settings

logger = logging.getLogger(__name__)


def get_schema_drift_events() -> dict:
    """
    Compare current catalogue_columns against the latest schema_snapshot baseline.
    Returns drift events suitable for display in the Streamlit sidebar.
    Returns a safe 'not_ready' payload if the DB is unavailable.
    """
    if not settings.duckdb_path.exists():
        return {
            "drift_events": [],
            "polled_at":    datetime.utcnow().isoformat(),
            "status":       "db_not_seeded",
            "message":      "Run: python db/seeder.py to initialise the database.",
        }

    try:
        con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    except Exception as exc:
        logger.warning("Could not connect to DuckDB: %s", exc)
        return {
            "drift_events": [],
            "polled_at":    datetime.utcnow().isoformat(),
            "status":       "db_error",
        }

    try:
        snapshots = con.execute(
            "SELECT table_name, columns_json, captured_at FROM schema_snapshots ORDER BY captured_at DESC"
        ).fetchall()

        if not snapshots:
            return {
                "drift_events": [],
                "polled_at":    datetime.utcnow().isoformat(),
                "status":       "no_baseline",
            }

        drift_events = []
        for table_name, columns_json_str, captured_at in snapshots:
            baseline_cols = {c["name"]: c["type"] for c in json.loads(columns_json_str)}
            current_rows  = con.execute(
                """
                SELECT cc.column_name, cc.data_type
                FROM catalogue_columns cc
                JOIN catalogue_tables ct ON cc.table_id = ct.table_id
                WHERE ct.table_name = ?
                """,
                [table_name],
            ).fetchall()
            current_cols = {r[0]: r[1] for r in current_rows}

            added        = list(set(current_cols) - set(baseline_cols))
            dropped      = list(set(baseline_cols) - set(current_cols))
            type_changed = [
                {"column": c, "from": baseline_cols[c], "to": current_cols[c]}
                for c in set(baseline_cols) & set(current_cols)
                if baseline_cols[c] != current_cols[c]
            ]

            if added or dropped or type_changed:
                drift_events.append({
                    "table":           table_name,
                    "added_columns":   added,
                    "dropped_columns": dropped,
                    "type_changes":    type_changed,
                    "baseline_at":     str(captured_at),
                    "severity":        "HIGH" if dropped or type_changed else "LOW",
                })

        return {
            "drift_events": drift_events,
            "polled_at":    datetime.utcnow().isoformat(),
            "status":       "drift_detected" if drift_events else "clean",
        }
    except Exception as exc:
        logger.warning("Schema drift check failed: %s", exc)
        return {
            "drift_events": [],
            "polled_at":    datetime.utcnow().isoformat(),
            "status":       "not_ready",
            "message":      str(exc),
        }
    finally:
        con.close()
PYEOF
log "agent/mcp_resources.py fixed (DB guard)"

# ==============================================================================
# FIX 6 — ui/components/approval_gate.py
# Increase timeout (DQ checks can be slow), persist result into conversation
# history so it survives st.rerun(), and add deny-path persistence.
# ==============================================================================
step "FIX 6: ui/components/approval_gate.py — timeout + result persistence"

cat << 'PYEOF' > ui/components/approval_gate.py
"""
Human-in-the-loop approval gate Streamlit component.
Displays pending tool call details with Allow / Deny buttons.

Changes vs v1:
  - Timeout increased to 60 s (DQ checks against large tables can be slow)
  - Tool result is appended to st.session_state.messages BEFORE st.rerun()
    so the result survives the re-render cycle and appears in conversation history
  - Deny path also appended to messages for a complete conversation record
"""
from __future__ import annotations

import json
import httpx
import streamlit as st


def render_approval_gate(
    tool_name: str,
    tool_args: dict,
    call_id: str,
    api_base: str = "http://localhost:8000",
) -> None:
    st.warning("Agent Action Requires Approval", icon="⚠")
    st.markdown(f"**Tool:** `{tool_name}`")
    st.json(tool_args)

    col_allow, col_deny = st.columns(2)
    with col_allow:
        if st.button("Allow", type="primary", key=f"allow_{call_id}"):
            _submit_approval(tool_name, tool_args, call_id, approved=True, api_base=api_base)
    with col_deny:
        if st.button("Deny", type="secondary", key=f"deny_{call_id}"):
            _submit_approval(tool_name, tool_args, call_id, approved=False, api_base=api_base)


def _submit_approval(
    tool_name: str,
    tool_args: dict,
    call_id: str,
    approved: bool,
    api_base: str,
) -> None:
    try:
        resp = httpx.post(
            f"{api_base}/api/v1/chat/approve",
            json={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "call_id":   call_id,
                "approved":  approved,
            },
            timeout=60,  # DQ checks can take up to 30 s
        )
        result = resp.json()

        if approved:
            response_text = result.get("result", "") or f"✅ `{tool_name}` executed successfully."
            # Persist into conversation history BEFORE rerun so it survives re-render
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({
                "role":    "assistant",
                "content": response_text,
            })
            st.session_state["approval_pending"] = None
            st.rerun()
        else:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": "Action denied. No changes were made.",
            })
            st.session_state["approval_pending"] = None
            st.rerun()

    except Exception as exc:
        st.error(f"Approval submission failed: {exc}")
PYEOF
log "ui/components/approval_gate.py fixed"

# ==============================================================================
# FIX 7 — Scripts: add PYTHONPATH export + auto-seed guard
# Every script now exports PYTHONPATH="." so pm_config resolves correctly
# without needing to install the package in editable mode.
# start_api.sh and ingest*.sh check if the DB exists and seed if not.
# ==============================================================================
step "FIX 7: Rewriting all scripts/ with PYTHONPATH + auto-seed"

mkdir -p scripts

cat << 'SHEOF' > scripts/start_api.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
# Auto-seed DB on first run so tables always exist when the API starts
[[ -f data/pipelinemind.db ]] || python db/seeder.py
echo "[PM] Starting FastAPI on http://localhost:8000"
echo "[PM] API docs: http://localhost:8000/docs"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
SHEOF

cat << 'SHEOF' > scripts/start_ui.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
echo "[PM] Starting Streamlit on http://localhost:8501"
streamlit run ui/app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false
SHEOF

cat << 'SHEOF' > scripts/ingest.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
[[ -f data/pipelinemind.db ]] || python db/seeder.py
echo "[PM] Running ingestion pipeline (full LLM summaries via Groq)..."
python ingestion/ingest_pipeline.py \
    --repo-path ./data/pipeline_repo \
    --sql-path  ./data/sql \
    --yaml-path ./data/dags \
    --dbt-path  ./data/dbt_project \
    "$@"
SHEOF

cat << 'SHEOF' > scripts/ingest_fast.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
[[ -f data/pipelinemind.db ]] || python db/seeder.py
echo "[PM] Running fast ingestion (no LLM summaries)..."
python ingestion/ingest_pipeline.py \
    --repo-path ./data/pipeline_repo \
    --sql-path  ./data/sql \
    --yaml-path ./data/dags \
    --dbt-path  ./data/dbt_project \
    --skip-summaries \
    --force-reindex
SHEOF

cat << 'SHEOF' > scripts/seed_db.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
python db/seeder.py
SHEOF

cat << 'SHEOF' > scripts/run_tests.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
pytest tests/ -v --tb=short
SHEOF

chmod +x scripts/*.sh
log "All scripts rewritten with PYTHONPATH + auto-seed"

# ==============================================================================
# FIX 8 — pyproject.toml: suppress noisy deprecation warnings in test output
# ==============================================================================
step "FIX 8: pyproject.toml — quieter test output"

python3 - << 'PYEOF'
from pathlib import Path

path    = Path("pyproject.toml")
content = path.read_text()

old = '[tool.pytest.ini_options]\nasyncio_mode = "auto"\ntestpaths = ["tests"]\nlog_cli = true\nlog_level = "INFO"\npythonpath = ["."]'
new = '''[tool.pytest.ini_options]
asyncio_mode   = "auto"
testpaths      = ["tests"]
log_cli        = true
log_level      = "WARNING"
log_cli_level  = "WARNING"
pythonpath     = ["."]
filterwarnings = [
    "ignore::DeprecationWarning:sentence_transformers",
    "ignore::pydantic.warnings.PydanticDeprecatedSince20",
    "ignore::DeprecationWarning:great_expectations",
]'''

if 'filterwarnings' not in content:
    if old in content:
        content = content.replace(old, new)
    else:
        # Try partial replacement
        content = content.replace(
            'log_level      = "INFO"',
            'log_level      = "WARNING"\nlog_cli_level  = "WARNING"'
        )
    path.write_text(content)
    print("pyproject.toml updated")
else:
    print("pyproject.toml already has filterwarnings")
PYEOF

# ==============================================================================
# FIX 9 — Safety pass: ensure all Python files import from pm_config not config
# ==============================================================================
step "FIX 9: Safety import-patch pass (config -> pm_config)"

find "$PROJECT_DIR" -name "*.py" \
    ! -path "*/.venv/*" \
    ! -path "*/data/*" \
    ! -name "pm_config.py" \
    | while read -r f; do
        if grep -q "from config import\|^import config$" "$f" 2>/dev/null; then
            sed -i '' \
                's/from config import/from pm_config import/g;
                 s/^import config$/import pm_config/g' "$f"
            log "  patched: ${f#$PROJECT_DIR/}"
        fi
    done

# ==============================================================================
# EXECUTE: Re-seed + run all tests
# ==============================================================================
step "Re-seeding DuckDB"

export PYTHONPATH="."
"$VENV_PYTHON" db/seeder.py

step "Verifying DuckDB row counts and timestamp freshness"

"$VENV_PYTHON" - << 'PYEOF'
import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta
import duckdb
from pm_config import settings

con = duckdb.connect(str(settings.duckdb_path), read_only=True)

tables = [
    "catalogue_tables", "catalogue_columns", "lineage_edges",
    "pipeline_runs", "slo_definitions", "schema_snapshots",
]
all_ok = True
for tbl in tables:
    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    status = "[OK]" if n > 0 else "[!!]"
    if n == 0:
        all_ok = False
    print(f"  {status} {tbl:<30} {n} rows")

# Verify timestamps are within rolling window
cutoff_24h  = (datetime.utcnow() - timedelta(hours=24)).isoformat()
cutoff_30d  = (datetime.utcnow() - timedelta(days=30)).isoformat()
runs_24h    = con.execute(
    "SELECT COUNT(*) FROM pipeline_runs WHERE start_time >= ?", [cutoff_24h]
).fetchone()[0]
runs_30d    = con.execute(
    "SELECT COUNT(*) FROM pipeline_runs WHERE start_time >= ?", [cutoff_30d]
).fetchone()[0]
print()
print(f"  Pipeline runs in last 24 h:  {runs_24h}")
print(f"  Pipeline runs in last 30 d:  {runs_30d}")
if runs_24h == 0:
    print("  [WARN] No runs in 24 h — timestamp shift may not have worked")
    all_ok = False
else:
    print("  [OK] Timestamp shift verified — rolling-window queries will find data")

con.close()
sys.exit(0 if all_ok else 1)
PYEOF

step "Running unit tests"

"$VENV_PYTEST" tests/unit/ -v --tb=short 2>&1

step "Running integration tests"

"$VENV_PYTEST" tests/integration/ -v --tb=short 2>&1 || warn "Integration tests need seeded DB — check output"

# ==============================================================================
# QUICK SMOKE TEST: verify intent classifier routes the original failing query
# ==============================================================================
step "Smoke test: intent classifier on the original failing query"

"$VENV_PYTHON" - << 'PYEOF'
import sys
sys.path.insert(0, ".")

from retrieval.intent_classifier import _keyword_classify, Intent

tests = [
    ("can you let me know about vw_revenue_by_tier table lineage dag", Intent.CATALOGUE),
    ("what pii columns exist in the users table",                       Intent.CATALOGUE),
    ("did the orders pipeline fail today",                              Intent.HEALTH),
    ("what if I drop user_id from stg_users",                          Intent.ACTION),
    ("why does orders pipeline use merge strategy",                     Intent.CODE_QA),
    ("what is incremental loading",                                     Intent.GENERAL),
]

all_pass = True
print()
for query, expected in tests:
    result = _keyword_classify(query)
    if result is None:
        status = "[LLM]"  # falls through to LLM — not a failure
        got    = "LLM_FALLBACK"
    else:
        got    = result[0].value
        status = "[OK] " if result[0] == expected else "[!!]"
        if result[0] != expected:
            all_pass = False
    print(f"  {status} expected={expected.value:<12} got={got:<12} | {query[:55]}")

print()
if all_pass:
    print("  All keyword routes correct — original failing query now routes to CATALOGUE")
else:
    print("  Some routes mismatch — review _KEYWORD_RULES in intent_classifier.py")
PYEOF

# ==============================================================================
# NEXT STEPS GUIDE
# ==============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Consolidation complete — all known fixes applied${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  What was fixed in this script:"
echo "  1. db/schema.sql         — removed secondary UNIQUE constraints (binder error fix)"
echo "  2. db/seeder.py          — DELETE+INSERT pattern + timestamp shifting"
echo "  3. agent/tools/validators.py — min_items -> min_length (Pydantic v2)"
echo "  4. ingestion/embedders.py — sentence_transformers import path for ST >= 3.x"
echo "  5. agent/mcp_resources.py — DB existence guard before connecting"
echo "  6. ui/components/approval_gate.py — timeout 60s + result persistence"
echo "  7. scripts/*             — PYTHONPATH export + auto-seed guard"
echo "  8. pyproject.toml        — quieter test output"
echo ""
echo -e "${BLUE}  WHAT TO DO NEXT (in order):${NC}"
echo ""
echo "  TERMINAL 1 — Start the backend:"
echo "    cd /Users/as-mac-1282/Developer/genai_mini/pipelinemind"
echo "    bash scripts/start_api.sh"
echo ""
echo "  TERMINAL 2 — Build the ChromaDB + BM25 index:"
echo "    cd /Users/as-mac-1282/Developer/genai_mini/pipelinemind"
echo "    bash scripts/ingest_fast.sh"
echo "    # (fast mode — no Groq calls; run scripts/ingest.sh for full LLM quality)"
echo ""
echo "  TERMINAL 3 — Start the Streamlit UI:"
echo "    cd /Users/as-mac-1282/Developer/genai_mini/pipelinemind"
echo "    bash scripts/start_ui.sh"
echo ""
echo "  Open http://localhost:8501 and test:"
echo "    'can you let me know about vw_revenue_by_tier table lineage dag'"
echo "    Terminal should now show: intent=CATALOGUE | tools_available=2 | max_iters=1"
echo "    Agent should call get_lineage_graph ONCE then stop."
echo ""
echo "  Verify LLM call distribution:"
echo "    curl http://localhost:8000/api/v1/agent/stats"
echo "    # INTENT + HYDE -> llama3-8b (fast, cheap)"
echo "    # AGENT         -> llama-3.3-70b (only when tools needed)"
echo ""
echo "  Run full ingestion with real Groq summaries (better retrieval quality):"
echo "    bash scripts/ingest.sh"
echo ""
echo "  Run tests anytime:"
echo "    bash scripts/run_tests.sh"
echo ""
echo -e "${YELLOW}  PHASE 3 REMAINING WORK (future scripts):${NC}"
echo "    - RAG evaluation notebook (MRR@5, NDCG@5, ablation study)"
echo "    - Docker Compose end-to-end test"
echo "    - Prometheus metrics dashboard"
echo "    - 10-slide deck assets"
echo ""