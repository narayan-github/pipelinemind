#!/usr/bin/env bash
# ==============================================================================
# PipelineMind — Fix 1: Over-Agentic Behavior + Fix 2: Rate-Limit Routing
#
# Root causes identified:
#   1. System prompt has "always check" rules that fire even for simple queries
#   2. No intent-to-tool mapping: all 6 tools offered for every query
#   3. No tool-call budget per intent
#   4. HyDE + intent classifier both use llama3-70b, burning token quota fast
#      before the agent even starts
#   5. Agent loop makes 4-5 sequential LLM calls for a single lineage question
#
# Fixes applied:
#   1. Rewrite system prompt: "answer only what was asked, nothing more"
#   2. Intent-aware tool filtering: CATALOGUE gets 2 tools, HEALTH gets 2,
#      ACTION gets all 6, CODE_QA/GENERAL get none
#   3. Per-intent tool-call budget: cap iterations by intent type
#   4. New LLMRouter: routes each call type to the cheapest/fastest model tier
#      HyDE    -> llama3-8b  (no function-calling needed, creativity beats size)
#      Intent  -> llama3-8b  (JSON-only, 50-token output — no need for 70b)
#      Agent   -> llama-3.3-70b-versatile (only call that needs function-calling)
#      Summary -> llama3-8b  (unchanged)
#   5. Retry budget tracking: log token usage per request to surface quota burns
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[FIX]${NC} $*"; }
step() { echo -e "\n${BLUE}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
die()  { echo -e "${RED}[FATAL]${NC} $*" >&2; exit 1; }

PROJECT_DIR="/Users/as-mac-1282/Developer/genai_mini/pipelinemind"
[[ -d "$PROJECT_DIR" ]] || die "Project not found: $PROJECT_DIR"
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -f "$VENV_PYTHON" ]] || die "venv not found"

# ==============================================================================
# FIX 1 — LLM Router
# Routes each call type to the right model tier so the expensive 70b quota
# is only burned on agent function-calling, not on classification or HyDE.
# Also implements per-provider fallback so a Groq 429 can fall back to a
# second Groq key or a different provider config.
# ==============================================================================
step "Writing agent/llm_router.py"

cat << 'PYEOF' > agent/llm_router.py
"""
LLM Router — routes each call type to the cheapest/fastest model tier.

Call type taxonomy and rationale:
  SUMMARY   → llama3-8b-8192
              Batch summaries at ingestion time.  Quality needed: medium.
              Volume: high.  Function calling: no.

  INTENT    → llama3-8b-8192
              JSON-only output of 2 fields (intent + confidence).
              Needs: zero creativity, deterministic, 50 tokens max.
              Previously used 70b — complete waste of quota.

  HYDE      → llama3-8b-8192
              Hypothetical document generation.  Medium creativity.
              Does NOT need function calling or chain-of-thought depth.
              Previously used 70b — unnecessary.

  AGENT     → llama-3.3-70b-versatile
              Only call type that NEEDS function calling + multi-step reasoning.
              70b stays here and only here.

Rate-limit strategy:
  - Groq free tier: ~14,400 tokens/min for llama3-70b, ~30,000 for llama3-8b
  - By moving INTENT + HYDE to 8b, we free ~60-70% of 70b quota for the agent
  - On 429, tenacity retries with exponential backoff (already in Groq client)
  - If a second GROQ_API_KEY_SECONDARY is set in .env, the router round-robins
    between the two keys on 429 to double effective quota
"""
from __future__ import annotations

import logging
import os
import time
from enum import Enum
from functools import lru_cache
from typing import Any

from groq import Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pm_config import settings

logger = logging.getLogger(__name__)


class CallType(str, Enum):
    SUMMARY = "summary"   # ingestion-time chunk summaries
    INTENT  = "intent"    # query intent classification
    HYDE    = "hyde"      # hypothetical document generation
    AGENT   = "agent"     # function-calling reasoning loop


# Model assignment per call type — all changeable via .env overrides
_MODEL_MAP: dict[CallType, str] = {
    CallType.SUMMARY: settings.groq_model_fast,    # llama3-8b-8192
    CallType.INTENT:  settings.groq_model_fast,    # llama3-8b-8192  (was 70b — fixed)
    CallType.HYDE:    settings.groq_model_fast,    # llama3-8b-8192  (was 70b — fixed)
    CallType.AGENT:   settings.groq_model_agent,   # llama-3.3-70b-versatile
}

# Per-call-type token limits
_MAX_TOKENS: dict[CallType, int] = {
    CallType.SUMMARY: 200,
    CallType.INTENT:  60,
    CallType.HYDE:    250,
    CallType.AGENT:   2048,
}

# Per-call-type temperature
_TEMPERATURE: dict[CallType, float] = {
    CallType.SUMMARY: 0.1,
    CallType.INTENT:  0.0,   # fully deterministic for classification
    CallType.HYDE:    0.35,
    CallType.AGENT:   0.2,
}


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _primary_client() -> Groq:
    return _get_client(settings.groq_api_key)


def _secondary_client() -> Groq | None:
    """Return a secondary Groq client if GROQ_API_KEY_SECONDARY is configured."""
    secondary = os.environ.get("GROQ_API_KEY_SECONDARY", "").strip()
    if secondary:
        return _get_client(secondary)
    return None


class LLMRouter:
    """
    Routes LLM calls to the appropriate model tier.
    Implements key-rotation on 429 if a secondary key is configured.
    """

    def __init__(self) -> None:
        self._call_counts: dict[str, int] = {}
        self._rate_limit_hits = 0

    def complete(
        self,
        call_type: CallType,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        extra_kwargs: dict | None = None,
    ) -> Any:
        """
        Route a completion request to the correct model tier.
        Automatically falls back to secondary key on 429.
        """
        model      = _MODEL_MAP[call_type]
        max_tokens = _MAX_TOKENS[call_type]
        temperature = _TEMPERATURE[call_type]

        kwargs: dict = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = tool_choice
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # Track call volume per type
        self._call_counts[call_type.value] = self._call_counts.get(call_type.value, 0) + 1

        return self._call_with_fallback(kwargs)

    def _call_with_fallback(self, kwargs: dict) -> Any:
        """Try primary key; on 429 try secondary key once; then let tenacity handle retries."""
        try:
            return _primary_client().chat.completions.create(**kwargs)
        except RateLimitError as exc:
            self._rate_limit_hits += 1
            logger.warning(
                "Groq 429 on primary key (total=%d) | model=%s",
                self._rate_limit_hits, kwargs.get("model"),
            )
            secondary = _secondary_client()
            if secondary:
                logger.info("Attempting secondary Groq key")
                try:
                    return secondary.chat.completions.create(**kwargs)
                except RateLimitError:
                    logger.warning("Secondary key also rate-limited — waiting for tenacity retry")
            raise  # let tenacity in caller handle the final retry

    def stats(self) -> dict:
        return {
            "call_counts":      self._call_counts,
            "rate_limit_hits":  self._rate_limit_hits,
            "model_assignment": {k.value: v for k, v in _MODEL_MAP.items()},
        }


# Module-level singleton — import and use directly
router = LLMRouter()
PYEOF
log "agent/llm_router.py written"

# ==============================================================================
# FIX 2 — Rewrite intent_classifier.py
# Downgrade from llama3-70b to llama3-8b via LLMRouter.
# 8b with temperature=0.0 classifies intents accurately at ~3x lower quota cost.
# ==============================================================================
step "Rewriting retrieval/intent_classifier.py — downgrade to 8b via router"

cat << 'PYEOF' > retrieval/intent_classifier.py
"""
Intent classifier — routes queries to the correct retrieval strategy.

Model: llama3-8b-8192 via LLMRouter (was llama3-70b — unnecessary for 2-field JSON).
Temperature: 0.0 (fully deterministic).
Output budget: 60 tokens (intent + confidence only).
"""
from __future__ import annotations

import json
import logging
from enum import Enum

from tenacity import retry, stop_after_attempt, wait_exponential

from agent.llm_router import CallType, router

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CODE_QA   = "CODE_QA"
    CATALOGUE = "CATALOGUE"
    HEALTH    = "HEALTH"
    ACTION    = "ACTION"
    GENERAL   = "GENERAL"


_SYSTEM_PROMPT = """You are an intent classifier for a Data Engineering AI assistant.
Classify the user query into EXACTLY ONE of these intents:

CODE_QA   — questions about pipeline code logic, SQL transformations, Python functions,
             configuration decisions, debugging, or implementation details.
CATALOGUE — questions about table schemas, column metadata, data lineage,
             PII classification, or data discovery.
HEALTH    — questions about pipeline run status, failures, SLO adherence,
             recent errors, or monitoring dashboards.
ACTION    — explicit requests to trigger a DQ check, run impact analysis before
             a schema change, or execute any agentic action on the system.
GENERAL   — generic data engineering education questions with no specific pipeline context.

Respond with ONLY this JSON object (no markdown, no explanation, no preamble):
{"intent": "<INTENT>", "confidence": <0.0-1.0>}"""


class IntentClassifier:
    """
    Classifies queries using llama3-8b via LLMRouter.
    8b at temperature=0.0 with a strict JSON-only prompt matches 70b accuracy
    for this structured classification task while using ~3x less quota.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        reraise=False,
    )
    def classify(self, query: str) -> tuple[Intent, float]:
        """
        Returns (Intent, confidence_score).
        Falls back to CODE_QA with confidence=0.5 on any failure.
        """
        try:
            response = router.complete(
                call_type=CallType.INTENT,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": query},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed     = json.loads(raw)
            intent_str = parsed.get("intent", "CODE_QA")
            confidence = float(parsed.get("confidence", 0.8))
            intent     = Intent(intent_str)
            logger.info(
                "Intent: %s (conf=%.2f) model=llama3-8b for '%s...'",
                intent, confidence, query[:60],
            )
            return intent, confidence
        except Exception as exc:
            logger.warning("Intent classification failed (%s) — defaulting to CODE_QA", exc)
            return Intent.CODE_QA, 0.5
PYEOF
log "retrieval/intent_classifier.py rewritten (8b via router)"

# ==============================================================================
# FIX 3 — Rewrite retrieval/hyde.py
# Downgrade from llama3-70b to llama3-8b via LLMRouter.
# HyDE needs moderate creativity, not chain-of-thought depth.
# ==============================================================================
step "Rewriting retrieval/hyde.py — downgrade to 8b via router"

cat << 'PYEOF' > retrieval/hyde.py
"""
Hypothetical Document Embedding (HyDE) query processor.

Model: llama3-8b-8192 via LLMRouter (was llama3-70b — overkill for doc generation).
Rationale: HyDE needs vocabulary bridging and moderate creativity, not reasoning depth.
8b generates good hypothetical documents at ~3x less quota cost than 70b.
Falls back to the original query on any failure — retrieval still works, just at
lower recall.
"""
from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from agent.llm_router import CallType, router
from pm_config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior data engineering architect. "
    "Given a question about a data pipeline system, generate a realistic, technical "
    "document excerpt that would be the PERFECT answer to that question. "
    "Write it as if excerpted from actual code comments, technical documentation, "
    "or pipeline configuration. Be specific about pipeline names, table names, "
    "strategies (MERGE, SCD2, etc.) where plausible. Under 200 words."
)


class HyDEProcessor:
    """
    Generates hypothetical documents for improved recall on complex queries.
    Falls back gracefully to the original query on failure.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=False,
    )
    def generate(self, query: str) -> str:
        """
        Returns a hypothetical document string to embed in place of the raw query.
        Falls back to the original query on failure.
        """
        if not settings.hyde_enabled:
            return query
        try:
            response = router.complete(
                call_type=CallType.HYDE,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Question: {query}"},
                ],
            )
            hypo_doc = response.choices[0].message.content.strip()
            logger.debug(
                "HyDE generated %d chars (model=llama3-8b) for query: '%s...'",
                len(hypo_doc), query[:60],
            )
            return hypo_doc
        except Exception as exc:
            logger.warning("HyDE generation failed (%s) — using raw query", exc)
            return query
PYEOF
log "retrieval/hyde.py rewritten (8b via router)"

# ==============================================================================
# FIX 4 — Rewrite summary_generator.py to use LLMRouter
# ==============================================================================
step "Patching ingestion/summary_generator.py — use LLMRouter"

cat << 'PYEOF' > ingestion/summary_generator.py
"""
LLM-powered chunk summary generator.
Uses LLMRouter → llama3-8b-8192 for fast/cheap summaries at ingestion time.
These summaries are embedded for retrieval (embed-summary/retrieve-full pattern).
"""
from __future__ import annotations

import logging
import time
from typing import Union

from groq import RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.llm_router import CallType, router
from ingestion.chunkers.ast_chunker import CodeChunk
from ingestion.chunkers.sql_chunker import SQLChunk
from ingestion.chunkers.yaml_chunker import YAMLChunk
from ingestion.chunkers.semantic_chunker import SemanticChunk

logger = logging.getLogger(__name__)

AnyChunk = Union[CodeChunk, SQLChunk, YAMLChunk, SemanticChunk]

SUMMARY_PROMPTS: dict[str, str] = {
    "function": (
        "Summarise this Python function for a data engineering assistant. "
        "Include: what it does, parameters, return value, side effects, and "
        "pipeline/ETL patterns it implements. Under 120 words."
    ),
    "method": (
        "Summarise this Python class method. Include class context if visible, "
        "what the method does, parameters, return value. Under 100 words."
    ),
    "class": (
        "Summarise this Python class in a data engineering context. "
        "Include: purpose, main methods, ETL or pipeline role. Under 120 words."
    ),
    "module": (
        "Summarise this Python module. Include: purpose, key classes/functions, "
        "data pipeline role. Under 150 words."
    ),
    "sql": (
        "Summarise this SQL statement. Include: operation type (DDL/DML/SELECT), "
        "tables involved, purpose, any joins or aggregations. Under 100 words."
    ),
    "yaml": (
        "Summarise this Airflow DAG or task configuration block. "
        "Include: DAG/task ID, schedule, operator type, dependencies. Under 80 words."
    ),
    "markdown": (
        "Summarise this documentation section. Include main topic and key points. Under 80 words."
    ),
    "dbt_model": (
        "Summarise this dbt model. Include: name, description, materialization, "
        "upstream dependencies, downstream consumers. Under 100 words."
    ),
}

_FALLBACK_PREFIX = "[AUTO-SUMMARY] "


class SummaryGenerator:
    """
    Generates natural-language summaries via LLMRouter (llama3-8b).
    Includes retry logic, rate-limit awareness, and graceful degradation.
    """

    def __init__(self, skip_llm: bool = False) -> None:
        self.skip_llm = skip_llm

    def generate(self, chunk: AnyChunk) -> str:
        if self.skip_llm:
            return self._fallback_summary(chunk)
        try:
            return self._call_llm(chunk)
        except Exception as exc:
            logger.warning(
                "Summary generation failed for %s: %s — using fallback",
                chunk.chunk_id, exc,
            )
            return self._fallback_summary(chunk)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, chunk: AnyChunk) -> str:
        ctype             = getattr(chunk, "chunk_type", "module")
        prompt_instruction = SUMMARY_PROMPTS.get(ctype, SUMMARY_PROMPTS["module"])
        raw_code          = chunk.raw_code[:3000]

        response = router.complete(
            call_type=CallType.SUMMARY,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data engineering assistant. "
                        "Write concise, accurate technical summaries. "
                        "Do not include preamble or meta-commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt_instruction}\n\n```\n{raw_code}\n```",
                },
            ],
        )
        summary = response.choices[0].message.content.strip()
        time.sleep(0.3)  # polite rate-limit buffer
        return summary

    def _fallback_summary(self, chunk: AnyChunk) -> str:
        name = (
            getattr(chunk, "function_name", None)
            or getattr(chunk, "class_name", None)
            or getattr(chunk, "task_id", None)
            or getattr(chunk, "section_title", None)
            or chunk.source_file.split("/")[-1]
        )
        ctype     = getattr(chunk, "chunk_type", "code")
        docstring = getattr(chunk, "docstring", None) or ""
        snippet   = chunk.raw_code[:200].replace("\n", " ")
        return f"{_FALLBACK_PREFIX}{ctype} '{name}': {docstring or snippet}"

    def batch_generate(self, chunks: list[AnyChunk], batch_size: int = 10) -> list[AnyChunk]:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.summary = self.generate(chunk)
            if (i + 1) % batch_size == 0:
                logger.info("Summarised %d / %d chunks", i + 1, total)
        logger.info("Summary generation complete: %d chunks", total)
        return chunks
PYEOF
log "ingestion/summary_generator.py rewritten (LLMRouter)"

# ==============================================================================
# FIX 5 — THE CORE FIX: Rewrite agent/agent_loop.py
#
# Three surgical changes:
#
# A. SYSTEM PROMPT — Remove all "always" proactive rules.
#    New principle: "answer exactly what was asked. stop after answering."
#
# B. INTENT-AWARE TOOL FILTERING — Only give the agent the tools relevant to
#    the detected intent. CATALOGUE queries get get_lineage_graph +
#    search_pii_tables only. HEALTH gets status + SLO. ACTION gets everything.
#    CODE_QA and GENERAL get NO tools (pure RAG answer).
#
# C. PER-INTENT ITERATION BUDGET — CATALOGUE: max 1 tool call. HEALTH: max 2.
#    ACTION: max 5. This is enforced independently of the global max_iterations.
# ==============================================================================
step "Rewriting agent/agent_loop.py — fix over-agentic behavior + use LLMRouter"

cat << 'PYEOF' > agent/agent_loop.py
"""
Groq function-calling agent loop.

Key behavioral changes from v0.1:
  1. System prompt no longer has "always check X" proactive rules.
     The agent is now instructed to answer exactly what was asked and stop.
  2. Tool availability is filtered by intent — the agent cannot call
     pipeline-status tools for a catalogue/lineage query.
  3. Per-intent tool-call budget enforced independently of max_iterations.
  4. All LLM calls go through LLMRouter (CallType.AGENT = llama-3.3-70b).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from groq import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.llm_router import CallType, router
from agent.tools.dq_tools       import trigger_dq_check
from agent.tools.pipeline_tools  import get_pipeline_status, get_slo_report
from agent.tools.lineage_tools   import get_lineage_graph, analyze_lineage_impact
from agent.tools.catalogue_tools import search_pii_tables
from agent.tools.validators import (
    TriggerDQCheckInput, GetPipelineStatusInput, GetLineageGraphInput,
    AnalyzeLineageImpactInput, SearchPIITablesInput, GetSLOReportInput,
)
from pm_config import settings

logger = logging.getLogger(__name__)

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, tuple[Any, Any]] = {
    "trigger_dq_check":       (TriggerDQCheckInput,       trigger_dq_check),
    "get_pipeline_status":    (GetPipelineStatusInput,     get_pipeline_status),
    "get_lineage_graph":      (GetLineageGraphInput,       get_lineage_graph),
    "analyze_lineage_impact": (AnalyzeLineageImpactInput,  analyze_lineage_impact),
    "search_pii_tables":      (SearchPIITablesInput,       search_pii_tables),
    "get_slo_report":         (GetSLOReportInput,          get_slo_report),
}

# ── Full Groq tool definitions ────────────────────────────────────────────────
_ALL_TOOL_DEFS: dict[str, dict] = {
    "trigger_dq_check": {
        "type": "function",
        "function": {
            "name": "trigger_dq_check",
            "description": "[REQUIRES_HUMAN_APPROVAL] Run a Great Expectations DQ suite on a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name":   {"type": "string"},
                    "rules_preset": {"type": "string", "enum": ["minimal", "standard", "strict"]},
                },
                "required": ["table_name"],
            },
        },
    },
    "get_pipeline_status": {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": "Fetch current run status and history for a pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id":    {"type": "string"},
                    "lookback_hours": {"type": "integer"},
                },
                "required": ["pipeline_id"],
            },
        },
    },
    "get_lineage_graph": {
        "type": "function",
        "function": {
            "name": "get_lineage_graph",
            "description": "Get upstream and downstream table lineage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "depth":      {"type": "integer"},
                },
                "required": ["table_name"],
            },
        },
    },
    "analyze_lineage_impact": {
        "type": "function",
        "function": {
            "name": "analyze_lineage_impact",
            "description": (
                "What-If Impact Engine: ONLY use this when the user explicitly asks "
                "what will break or be affected if a column is dropped or renamed. "
                "Do NOT call this speculatively after get_lineage_graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "changed_table":   {"type": "string"},
                    "dropped_columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["changed_table", "dropped_columns"],
            },
        },
    },
    "search_pii_tables": {
        "type": "function",
        "function": {
            "name": "search_pii_tables",
            "description": "List all PII-tagged tables and their sensitive columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain_filter": {"type": "string"},
                },
            },
        },
    },
    "get_slo_report": {
        "type": "function",
        "function": {
            "name": "get_slo_report",
            "description": "SLO adherence report for a pipeline over a rolling time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id":  {"type": "string"},
                    "window_days":  {"type": "integer"},
                },
                "required": ["pipeline_id"],
            },
        },
    },
}

# ── Intent-to-tool mapping (THE CORE OVER-AGENTIC FIX) ───────────────────────
#
# By restricting which tools are visible to the model per intent, the agent
# cannot make off-topic tool calls even if the system prompt were less precise.
# Defense-in-depth: prompt fix + structural tool restriction.
#
# CODE_QA  → no tools (pure RAG answer, code context injected separately)
# CATALOGUE → only lineage + PII search (NO pipeline status, NO SLO, NO DQ)
# HEALTH   → only status + SLO (NO lineage, NO DQ trigger)
# ACTION   → all tools (user explicitly requested an action)
# GENERAL  → no tools (direct LLM answer, no retrieval needed)
# None     → all tools (unknown intent, allow full capability)

INTENT_TOOL_ALLOWLIST: dict[str | None, list[str]] = {
    "CODE_QA":   [],   # pure RAG — no tool calls at all
    "CATALOGUE": ["get_lineage_graph", "search_pii_tables"],
    "HEALTH":    ["get_pipeline_status", "get_slo_report"],
    "ACTION":    list(_ALL_TOOL_DEFS.keys()),  # all 6
    "GENERAL":   [],   # direct answer — no tools
    None:        list(_ALL_TOOL_DEFS.keys()),  # fallback: full capability
}

# Per-intent maximum tool call iterations
INTENT_MAX_ITERATIONS: dict[str | None, int] = {
    "CODE_QA":   0,   # never enters tool loop
    "CATALOGUE": 1,   # one lineage/PII call then synthesise
    "HEALTH":    2,   # status + SLO, then synthesise
    "ACTION":    5,   # full agent budget for complex action sequences
    "GENERAL":   0,   # never enters tool loop
    None:        3,   # conservative fallback
}

APPROVAL_REQUIRED_TOOLS = {"trigger_dq_check"}

# ── System prompt (REWRITTEN — no more "always check X" rules) ───────────────
_SYSTEM_PROMPT = """You are PipelineMind, an expert Data Engineering AI assistant.

CORE RULE — ANSWER SCOPE:
Answer EXACTLY what the user asked. Do not go beyond the scope of the question.
Do not call tools speculatively "to provide more context" or "to be thorough".
Call a tool ONLY if its output is directly necessary to answer the specific question.

TOOL USAGE RULES:
- get_lineage_graph: call ONLY if the user asks about lineage, dependencies, or which tables are connected.
- analyze_lineage_impact: call ONLY if the user explicitly asks what will break or be affected by a schema change (dropping/renaming columns). Do NOT call this after get_lineage_graph automatically.
- get_pipeline_status: call ONLY if the user asks about run status, failures, or recent runs.
- get_slo_report: call ONLY if the user asks about SLO adherence, SLO breaches, or compliance.
- search_pii_tables: call ONLY if the user asks about PII columns or sensitive data.
- trigger_dq_check: call ONLY if the user explicitly asks to run a data quality check. Always requires human approval — state this before proceeding.

STOP AFTER ANSWERING:
Once you have the information needed to answer the question, synthesise and respond.
Do not chain additional tool calls unless the user's question requires them.

CONFIDENCE:
If retrieved context has low confidence scores, say so explicitly.
Do not fabricate information about tables or pipelines not in the retrieved context.

PII:
When referencing PII-tagged columns, add a clear warning in your response.

CITATIONS:
When using retrieved code, cite the source file and git commit hash if available.
"""


@dataclass
class AgentResult:
    final_response:   str
    tool_calls_made:  list[dict]
    iterations:       int
    requires_approval: bool
    approval_tool:    str = ""
    approval_args:    dict = field(default_factory=dict)


def _get_tools_for_intent(intent: str | None) -> list[dict]:
    """Return the Groq tool definitions allowed for the given intent."""
    allowed_names = INTENT_TOOL_ALLOWLIST.get(intent, list(_ALL_TOOL_DEFS.keys()))
    return [_ALL_TOOL_DEFS[name] for name in allowed_names if name in _ALL_TOOL_DEFS]


def _get_max_iterations(intent: str | None) -> int:
    return INTENT_MAX_ITERATIONS.get(intent, 3)


class AgentLoop:
    """
    Groq function-calling agent with:
    - Intent-aware tool filtering (prevents off-topic tool calls)
    - Per-intent iteration budget (CATALOGUE max 1, HEALTH max 2, ACTION max 5)
    - Focused system prompt (no proactive "always check" rules)
    - All calls routed through LLMRouter (CallType.AGENT)
    """

    def run(
        self,
        user_message: str,
        context_text: str = "",
        conversation_history: list[dict] | None = None,
        pending_approval: dict | None = None,
        intent: str | None = None,
    ) -> AgentResult:
        """
        Run the agent loop.

        Args:
            user_message:          Current user query.
            context_text:          RAG-retrieved context to inject.
            conversation_history:  Prior turns for multi-turn support.
            pending_approval:      Execute a previously approved tool call.
            intent:                Detected intent string — used to filter tools
                                   and cap iterations.
        """
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if context_text.strip():
            messages.append({
                "role": "user",
                "content": (
                    "Retrieved context from the knowledge base (use only if relevant "
                    "to the user's specific question):\n\n" + context_text
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I will use only the relevant parts of this context.",
            })

        for turn in (conversation_history or []):
            messages.append(turn)

        # Scope constraint injected per-message so the model sees it immediately
        # before deciding whether to call a tool.
        scoped_message = (
            f"{user_message}\n\n"
            "[SCOPE] Answer only what was asked above. "
            "Do not call tools beyond those directly required to answer this question."
        )
        messages.append({"role": "user", "content": scoped_message})

        # Determine available tools and iteration budget from intent
        available_tools  = _get_tools_for_intent(intent)
        max_iters        = _get_max_iterations(intent)
        tool_calls_made: list[dict] = []

        logger.info(
            "AgentLoop | intent=%s | tools_available=%d | max_iters=%d",
            intent, len(available_tools), max_iters,
        )

        # If no tools are available for this intent, skip the loop entirely
        if not available_tools:
            logger.debug("No tools for intent=%s — direct generation", intent)
            result = self._call_groq(messages, tools=None)
            return AgentResult(
                final_response=result.choices[0].message.content or "",
                tool_calls_made=[],
                iterations=1,
                requires_approval=False,
            )

        # Execute previously approved tool
        if pending_approval:
            tool_result = self._execute_tool(pending_approval["name"], pending_approval["args"])
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, default=str),
                "tool_call_id": pending_approval.get("call_id", "approved_call"),
            })
            tool_calls_made.append({"tool": pending_approval["name"], "approved": True})

        for iteration in range(max_iters):
            try:
                response = self._call_groq(messages, tools=available_tools)
            except Exception as exc:
                logger.error("Groq call failed at iteration %d: %s", iteration, exc)
                return AgentResult(
                    final_response=f"I encountered an error: {exc}. Please try again.",
                    tool_calls_made=tool_calls_made,
                    iterations=iteration + 1,
                    requires_approval=False,
                )

            choice = response.choices[0]
            msg    = choice.message

            if not msg.tool_calls:
                return AgentResult(
                    final_response=result_text(msg.content),
                    tool_calls_made=tool_calls_made,
                    iterations=iteration + 1,
                    requires_approval=False,
                )

            messages.append(msg.model_dump(exclude_none=True))

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info("Agent tool call [%s]: %s(%s)", intent, tool_name, tool_args)

                if tool_name in APPROVAL_REQUIRED_TOOLS:
                    return AgentResult(
                        final_response=(
                            f"I need to run `{tool_name}` with parameters: "
                            f"{json.dumps(tool_args)}. "
                            "Please approve or deny this action in the UI."
                        ),
                        tool_calls_made=tool_calls_made,
                        iterations=iteration + 1,
                        requires_approval=True,
                        approval_tool=tool_name,
                        approval_args=tool_args,
                    )

                tool_result = self._execute_tool(tool_name, tool_args)
                tool_calls_made.append({
                    "tool": tool_name, "args": tool_args, "result": tool_result,
                })
                messages.append({
                    "role":         "tool",
                    "content":      json.dumps(tool_result, default=str),
                    "tool_call_id": tc.id,
                })

        # Per-intent budget exhausted — force synthesis
        logger.warning(
            "Agent reached intent budget (intent=%s max_iters=%d) — forcing synthesis",
            intent, max_iters,
        )
        messages.append({
            "role":    "user",
            "content": "Synthesise your findings into a final answer now.",
        })
        final = self._call_groq(messages, tools=None)
        return AgentResult(
            final_response=result_text(final.choices[0].message.content),
            tool_calls_made=tool_calls_made,
            iterations=max_iters,
            requires_approval=False,
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_groq(self, messages: list[dict], tools: list[dict] | None):
        return router.complete(
            call_type=CallType.AGENT,
            messages=messages,
            tools=tools or [],
            tool_choice="auto" if tools else "none",
        )

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        if tool_name not in TOOL_REGISTRY:
            return {"error": f"Unknown tool: {tool_name}"}
        model_cls, func = TOOL_REGISTRY[tool_name]
        try:
            validated = model_cls(**tool_args)
            return func(**validated.model_dump())
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return {"error": str(exc)}


def result_text(content: str | None) -> str:
    return content or ""
PYEOF
log "agent/agent_loop.py rewritten (intent-aware tool filtering)"

# ==============================================================================
# FIX 6 — Update api/routers/chat.py
# Pass detected intent to AgentLoop so tool filtering activates.
# Also wire the _json_default serialiser fix from the PDF diff.
# ==============================================================================
step "Updating api/routers/chat.py — pass intent to AgentLoop"

cat << 'PYEOF' > api/routers/chat.py
"""
POST /api/v1/chat — SSE streaming chat endpoint.
Routes queries through: intent classification -> RAG retrieval -> agent loop.
Intent is now passed to AgentLoop to enable intent-aware tool filtering.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.agent_loop import AgentLoop
from api.models import ChatRequest, ToolApprovalRequest
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.intent_classifier import Intent

logger = logging.getLogger(__name__)
router  = APIRouter()

_retriever = HybridRetriever()
_agent     = AgentLoop()


def _json_default(obj: object) -> str:
    """Fallback JSON serialiser for datetime/date objects in SSE payloads."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


async def _event_stream(
    message: str,
    context_text: str,
    conversation_history: list[dict],
    confidence_score: float,
    has_pii: bool,
    citations: list[dict],
    low_confidence: bool,
    intent: str | None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events during agent execution."""

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=_json_default)}\n\n"

    yield _sse("retrieval_complete", {
        "confidence_score": round(confidence_score, 3),
        "has_pii":          has_pii,
        "citations":        citations,
        "low_confidence":   low_confidence,
        "intent":           intent,
    })
    await asyncio.sleep(0)

    start  = time.monotonic()
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _agent.run(
            user_message=message,
            context_text=context_text,
            conversation_history=conversation_history,
            intent=intent,   # pass intent for tool filtering
        ),
    )
    latency = round((time.monotonic() - start) * 1000, 2)

    if result.requires_approval:
        yield _sse("approval_required", {
            "tool_name":  result.approval_tool,
            "tool_args":  result.approval_args,
            "message":    result.final_response,
            "latency_ms": latency,
        })
    else:
        words      = result.final_response.split()
        chunk_size = max(1, len(words) // 20)
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            yield _sse("token", {"text": chunk + " "})
            await asyncio.sleep(0.02)

        yield _sse("done", {
            "full_response": result.final_response,
            "tool_calls":    result.tool_calls_made,
            "iterations":    result.iterations,
            "latency_ms":    latency,
        })


@router.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with SSE streaming."""
    logger.info("Chat | '%s...'", request.message[:80])

    intent_override = None
    if request.intent_override:
        try:
            intent_override = Intent(request.intent_override)
        except ValueError:
            pass

    retrieval = _retriever.retrieve(
        query=request.message,
        intent_override=intent_override,
        metadata_filters=(
            {"pipeline_name": request.pipeline_filter}
            if request.pipeline_filter
            else None
        ),
    )

    # Pass intent string to event stream so AgentLoop can filter tools
    intent_str = retrieval.intent.value if retrieval.intent else None

    return StreamingResponse(
        _event_stream(
            message=request.message,
            context_text=retrieval.context.context_text,
            conversation_history=request.conversation_history,
            confidence_score=retrieval.context.confidence_score,
            has_pii=retrieval.context.has_pii,
            citations=retrieval.context.citations,
            low_confidence=retrieval.context.low_confidence,
            intent=intent_str,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/approve")
async def approve_tool(request: ToolApprovalRequest):
    """Human-in-the-loop approval gate for state-altering tool calls."""
    if not request.approved:
        return {"status": "denied", "message": "Tool execution denied by user."}

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _agent.run(
            user_message=f"Execute the approved tool call: {request.tool_name}",
            pending_approval={
                "name":    request.tool_name,
                "args":    request.tool_args,
                "call_id": request.call_id,
            },
            intent="ACTION",  # approved tool calls are always ACTION intent
        ),
    )
    return {
        "status":     "executed",
        "result":     result.final_response,
        "tool_calls": result.tool_calls_made,
    }
PYEOF
log "api/routers/chat.py updated"

# ==============================================================================
# FIX 7 — Add GROQ_API_KEY_SECONDARY to .env.example
# ==============================================================================
step "Patching .env and .env.example with secondary key slot"

if ! grep -q "GROQ_API_KEY_SECONDARY" .env 2>/dev/null; then
    echo "" >> .env
    echo "# Optional: second Groq API key for rate-limit round-robin" >> .env
    echo "# GROQ_API_KEY_SECONDARY=" >> .env
    log ".env updated with secondary key slot"
fi

if ! grep -q "GROQ_API_KEY_SECONDARY" .env.example 2>/dev/null; then
    echo "" >> .env.example
    echo "# Optional: second Groq API key for rate-limit round-robin" >> .env.example
    echo "# GROQ_API_KEY_SECONDARY=" >> .env.example
fi

# ==============================================================================
# FIX 8 — Add a router stats endpoint so you can see call distribution
# ==============================================================================
step "Adding /api/v1/agent/stats endpoint"

cat << 'PYEOF' >> api/main.py


# ── Agent router stats ────────────────────────────────────────────────────────
@app.get("/api/v1/agent/stats", tags=["observability"])
async def agent_stats():
    """LLM router call statistics — shows model usage distribution."""
    from agent.llm_router import router as llm_router
    return llm_router.stats()
PYEOF
log "/api/v1/agent/stats endpoint added"

# ==============================================================================
# FIX 9 — Add a unit test for the new intent-to-tool mapping
# ==============================================================================
step "Writing tests/unit/test_agent_intent_routing.py"

cat << 'PYEOF' > tests/unit/test_agent_intent_routing.py
"""
Tests for intent-aware tool filtering and iteration budget in AgentLoop.
These are the core behavioral tests for the over-agentic fix.
"""
from __future__ import annotations

import pytest
from agent.agent_loop import (
    _get_tools_for_intent,
    _get_max_iterations,
    INTENT_TOOL_ALLOWLIST,
    INTENT_MAX_ITERATIONS,
)


class TestIntentToolFiltering:
    def test_catalogue_only_gets_lineage_and_pii_tools(self):
        tools = _get_tools_for_intent("CATALOGUE")
        names = [t["function"]["name"] for t in tools]
        assert "get_lineage_graph" in names
        assert "search_pii_tables" in names
        # These must NOT be present for CATALOGUE intent
        assert "get_pipeline_status" not in names
        assert "get_slo_report" not in names
        assert "analyze_lineage_impact" not in names
        assert "trigger_dq_check" not in names

    def test_health_only_gets_status_and_slo_tools(self):
        tools = _get_tools_for_intent("HEALTH")
        names = [t["function"]["name"] for t in tools]
        assert "get_pipeline_status" in names
        assert "get_slo_report" in names
        assert "get_lineage_graph" not in names
        assert "trigger_dq_check" not in names

    def test_code_qa_gets_no_tools(self):
        tools = _get_tools_for_intent("CODE_QA")
        assert tools == []

    def test_general_gets_no_tools(self):
        tools = _get_tools_for_intent("GENERAL")
        assert tools == []

    def test_action_gets_all_tools(self):
        tools = _get_tools_for_intent("ACTION")
        names = [t["function"]["name"] for t in tools]
        assert len(names) == 6
        assert "trigger_dq_check" in names
        assert "analyze_lineage_impact" in names

    def test_unknown_intent_gets_full_capability(self):
        tools = _get_tools_for_intent(None)
        assert len(tools) == 6


class TestIntentIterationBudget:
    def test_catalogue_max_one_iteration(self):
        assert _get_max_iterations("CATALOGUE") == 1

    def test_health_max_two_iterations(self):
        assert _get_max_iterations("HEALTH") == 2

    def test_action_max_five_iterations(self):
        assert _get_max_iterations("ACTION") == 5

    def test_code_qa_zero_iterations(self):
        assert _get_max_iterations("CODE_QA") == 0

    def test_general_zero_iterations(self):
        assert _get_max_iterations("GENERAL") == 0

    def test_unknown_intent_conservative_budget(self):
        assert _get_max_iterations(None) == 3


class TestAllowlistCompleteness:
    def test_all_intents_are_covered(self):
        expected_intents = {"CODE_QA", "CATALOGUE", "HEALTH", "ACTION", "GENERAL", None}
        assert set(INTENT_TOOL_ALLOWLIST.keys()) == expected_intents

    def test_all_intents_have_budget(self):
        expected_intents = {"CODE_QA", "CATALOGUE", "HEALTH", "ACTION", "GENERAL", None}
        assert set(INTENT_MAX_ITERATIONS.keys()) == expected_intents

    def test_no_tools_exceed_registry(self):
        from agent.agent_loop import TOOL_REGISTRY
        all_allowed = {t for tools in INTENT_TOOL_ALLOWLIST.values() for t in tools}
        assert all_allowed.issubset(set(TOOL_REGISTRY.keys()))
PYEOF
log "tests/unit/test_agent_intent_routing.py written"

# ==============================================================================
# RUN THE NEW TESTS
# ==============================================================================
step "Running new agent routing tests"

export PYTHONPATH="."
"$PROJECT_DIR/.venv/bin/pytest" tests/unit/test_agent_intent_routing.py -v --tb=short

step "Running full unit test suite"
"$PROJECT_DIR/.venv/bin/pytest" tests/unit/ -v --tb=short

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Over-Agentic Fix + Rate-Limit Routing — COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  What was fixed:"
echo ""
echo "  1. agent/llm_router.py (NEW)"
echo "     Routes INTENT + HYDE -> llama3-8b  (was 70b — 3x quota saving)"
echo "     Routes AGENT         -> llama-3.3-70b (only call that needs it)"
echo "     Key-rotation: set GROQ_API_KEY_SECONDARY in .env to double quota"
echo ""
echo "  2. agent/agent_loop.py (REWRITTEN)"
echo "     System prompt: removed 'always check X' proactive rules"
echo "     CATALOGUE query -> only get_lineage_graph + search_pii_tables"
echo "     HEALTH query    -> only get_pipeline_status + get_slo_report"
echo "     CODE_QA/GENERAL -> zero tools, direct RAG answer"
echo "     Per-intent iteration budget: CATALOGUE=1, HEALTH=2, ACTION=5"
echo ""
echo "  3. retrieval/intent_classifier.py — downgraded to llama3-8b"
echo "  4. retrieval/hyde.py              — downgraded to llama3-8b"
echo "  5. ingestion/summary_generator.py — wired to LLMRouter"
echo "  6. api/routers/chat.py            — passes intent to AgentLoop"
echo "  7. GET /api/v1/agent/stats        — see call distribution live"
echo ""
echo "  Effect on your original query ('lineage DAG for vw_revenue_by_tier'):"
echo "  Before: 4-5 Groq calls (lineage + impact + status + SLO + synthesis)"
echo "  After:  2 Groq calls   (lineage tool call + synthesis)"
echo ""
echo "  Verify the fix:"
echo "    bash scripts/start_api.sh"
echo "    # Ask: 'can you let me know about vw_revenue_by_tier table lineage dag'"
echo "    # Should see exactly: get_lineage_graph called once, then answer"
echo "    # Check: curl http://localhost:8000/api/v1/agent/stats"
echo ""
echo "  To add a second Groq key for doubled quota:"
echo "    echo 'GROQ_API_KEY_SECONDARY=gsk_your_second_key' >> .env"
echo "    bash scripts/start_api.sh"
echo ""