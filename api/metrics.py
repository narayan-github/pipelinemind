"""
Prometheus metrics registry for PipelineMind.
All counters and histograms are defined here and imported where needed.
This avoids duplicate-registration errors on FastAPI reload.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# ── Chat endpoint metrics ─────────────────────────────────────────────────────
CHAT_REQUESTS_TOTAL = Counter(
    "pipelinemind_chat_requests_total",
    "Total chat requests received",
    ["intent", "has_pii"],
)

CHAT_LATENCY_SECONDS = Histogram(
    "pipelinemind_chat_latency_seconds",
    "End-to-end chat request latency (streaming complete)",
    ["intent"],
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0],
)

TOOL_CALLS_TOTAL = Counter(
    "pipelinemind_tool_calls_total",
    "MCP tool calls executed",
    ["tool_name", "approved"],
)

APPROVAL_REQUESTS_TOTAL = Counter(
    "pipelinemind_approval_requests_total",
    "Human-in-the-loop approval requests",
    ["tool_name", "decision"],
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "pipelinemind_retrieval_latency_seconds",
    "Hybrid retrieval pipeline latency",
    ["mode"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0],
)

RETRIEVAL_CONFIDENCE = Histogram(
    "pipelinemind_retrieval_confidence_score",
    "Retrieval confidence scores distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LOW_CONFIDENCE_TOTAL = Counter(
    "pipelinemind_low_confidence_responses_total",
    "Responses where confidence < threshold",
)

# ── LLM call metrics ──────────────────────────────────────────────────────────
LLM_CALLS_TOTAL = Counter(
    "pipelinemind_llm_calls_total",
    "Groq API calls made",
    ["call_type", "model"],
)

LLM_LATENCY_SECONDS = Histogram(
    "pipelinemind_llm_latency_seconds",
    "Groq API call latency",
    ["call_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

RATE_LIMIT_HITS_TOTAL = Counter(
    "pipelinemind_groq_rate_limit_hits_total",
    "Groq 429 rate limit hits",
    ["call_type"],
)

# ── Ingestion metrics ─────────────────────────────────────────────────────────
INGESTION_CHUNKS_TOTAL = Counter(
    "pipelinemind_ingestion_chunks_total",
    "Total chunks indexed into ChromaDB",
    ["source_type"],
)

CHROMA_COLLECTION_SIZE = Gauge(
    "pipelinemind_chroma_collection_size",
    "Current number of documents in ChromaDB collection",
)

# ── Pipeline health metrics ───────────────────────────────────────────────────
PIPELINE_SLO_PCT = Gauge(
    "pipelinemind_pipeline_slo_pct",
    "Current SLO adherence percentage per pipeline",
    ["pipeline_id"],
)

SCHEMA_DRIFT_EVENTS = Gauge(
    "pipelinemind_schema_drift_events_active",
    "Number of active schema drift events detected",
)
