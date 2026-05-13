"""
PipelineMind FastAPI application entry point.
Port 8000 — all routes prefixed /api/v1/
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.metrics import (
    CHAT_REQUESTS_TOTAL, CHAT_LATENCY_SECONDS,
    RETRIEVAL_LATENCY_SECONDS, RETRIEVAL_CONFIDENCE,
    LOW_CONFIDENCE_TOTAL, CHROMA_COLLECTION_SIZE,
    PIPELINE_SLO_PCT, SCHEMA_DRIFT_EVENTS,
)
from api.middleware.logging  import StructuredLoggingMiddleware
from api.middleware.pii_guard import PIIGuardMiddleware
from api.routers import chat, pipelines, catalogue, dq, impact
from pm_config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="PipelineMind API",
    version="0.1.0",
    description="RAG-Powered Data Engineering Assistant",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(PIIGuardMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"
app.include_router(chat.router,       prefix=PREFIX, tags=["chat"])
app.include_router(pipelines.router,  prefix=PREFIX, tags=["pipelines"])
app.include_router(catalogue.router,  prefix=PREFIX, tags=["catalogue"])
app.include_router(dq.router,         prefix=PREFIX, tags=["data-quality"])
app.include_router(impact.router,     prefix=PREFIX, tags=["impact"])

# ── Health + Metrics ──────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["observability"])
async def health():
    """Health check with live ChromaDB + DuckDB status."""
    chroma_count = 0
    try:
        import chromadb
        client       = chromadb.PersistentClient(path=str(settings.chroma_path))
        coll         = client.get_or_create_collection(
            "pipelinemind", metadata={"hnsw:space": "cosine"}
        )
        chroma_count = coll.count()
        CHROMA_COLLECTION_SIZE.set(chroma_count)
    except Exception:
        pass

    db_ok = settings.duckdb_path.exists()
    return {
        "status":        "ok",
        "environment":   settings.environment,
        "chroma_docs":   chroma_count,
        "duckdb_seeded": db_ok,
    }


@app.get("/metrics", tags=["observability"])
async def metrics():
    """Prometheus metrics endpoint."""
    # Update live gauges on every scrape
    try:
        from agent.mcp_resources import get_schema_drift_events
        drift = get_schema_drift_events()
        SCHEMA_DRIFT_EVENTS.set(len(drift.get("drift_events", [])))
    except Exception:
        pass

    try:
        import duckdb
        con = duckdb.connect(str(settings.duckdb_path), read_only=True)
        rows = con.execute(
            """
            WITH latest AS (
                SELECT pipeline_id,
                       status,
                       ROW_NUMBER() OVER (PARTITION BY pipeline_id ORDER BY start_time DESC) AS rn
                FROM pipeline_runs
            ),
            window_stats AS (
                SELECT pipeline_id,
                       COUNT(*) AS total,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS ok
                FROM pipeline_runs
                WHERE start_time >= NOW() - INTERVAL '7 days'
                GROUP BY pipeline_id
            )
            SELECT pipeline_id, ok * 100.0 / total AS slo_pct
            FROM window_stats WHERE total > 0
            """
        ).fetchall()
        con.close()
        for pipeline_id, slo_pct in rows:
            PIPELINE_SLO_PCT.labels(pipeline_id=pipeline_id).set(slo_pct)
    except Exception:
        pass

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/schema-drift", tags=["observability"])
async def schema_drift():
    from agent.mcp_resources import get_schema_drift_events
    return get_schema_drift_events()


@app.get("/api/v1/agent/stats", tags=["observability"])
async def agent_stats():
    """LLM router call statistics — shows model usage distribution."""
    from agent.llm_router import router as llm_router
    return llm_router.stats()
