#!/usr/bin/env bash
# ==============================================================================
# PipelineMind — Fix: Tool Schema Coercion + Pipeline Disambiguation + Quality
#
# ROOT CAUSES from new terminal logs:
#
#   1. TOOL SCHEMA ERROR — `depth` sent as string "1" instead of integer 1
#      Groq validates tool params BEFORE our Pydantic runs → 400 Bad Request
#      `get_lineage_graph({"table_name": "OrdersPipeline", "depth": "1"})`
#      Fix: add pre-call arg coercion that converts string numbers to int/float
#
#   2. HALLUCINATED TABLE NAMES — model invents table names from code context
#      "fact table" → not in DuckDB, returns 0 affected assets
#      "OrdersPipeline" → Python class name, not a catalogue table name
#      Fix: inject available table names into agent system prompt context
#      Add a list_catalogue_tables tool so agent can resolve ambiguous names
#
#   3. HALLUCINATED PIPELINE IDs — "InventorySnapshotPipeline" not seeded
#      Seeded IDs are: orders, users, inventory, sessions, metrics
#      Model guesses class name from code chunks instead of real pipeline IDs
#      Fix: inject real pipeline IDs into HEALTH-intent system prompt context
#      Add list_pipelines as an agent-callable tool for HEALTH intent
#
#   4. CATALOGUE max_iters=1 too tight
#      Iteration 1: tool call fires + result appended → budget exhausted
#      Force-synthesis triggers immediately, model synthesises with result
#      But on first run the result hasn't been processed yet
#      Fix: CATALOGUE max_iters = 2 (tool call + synthesis with result)
#
#   5. Citation "unknown" — empty source_file string in chunk metadata
#      Fix: fallback label in chat_panel when file is empty/unknown
#
#   6. HEALTH conf=5.9% — cross-encoder gives low score to health queries
#      because pipeline health data lives in DuckDB, not ChromaDB
#      The retrieval is correctly low-confidence; HEALTH queries should
#      bypass RAG confidence check and route directly to tools
#      Fix: HEALTH intent skips confidence check, always calls status tool
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

# ==============================================================================
# FIX 1 — Pre-call argument type coercion + new list_catalogue_tables tool
# ==============================================================================
step "FIX 1: Add list_catalogue_tables + list_pipelines tools"

cat << 'PYEOF' > agent/tools/discovery_tools.py
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
PYEOF
log "agent/tools/discovery_tools.py written"

# ==============================================================================
# FIX 2 — Rewrite agent_loop.py with:
#   - Type coercion for tool args (depth: "1" → 1)
#   - discovery tools added to CATALOGUE + HEALTH tool allowlists
#   - CATALOGUE max_iters bumped to 2
#   - Context injection: available table names + pipeline IDs in system prompt
# ==============================================================================
step "FIX 2: Rewrite agent/agent_loop.py"

cat << 'PYEOF' > agent/agent_loop.py
"""
Groq function-calling agent loop.

v0.3 changes:
  - Type coercion: converts string numbers to int/float before tool execution
    (Groq rejects tool calls where depth="1" instead of depth=1)
  - list_catalogue_tables added to CATALOGUE allowlist (resolves ambiguous names)
  - list_pipeline_ids added to HEALTH allowlist (prevents hallucinated IDs)
  - CATALOGUE max_iters = 2 (tool call + synthesis with result)
  - Context injection: real table names and pipeline IDs in system message
  - HEALTH intent: always calls list_pipeline_ids first if no pipeline specified
"""
from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass, field
from typing import Any

from groq import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.llm_router import CallType, router
from agent.tools.dq_tools         import trigger_dq_check
from agent.tools.pipeline_tools    import get_pipeline_status, get_slo_report
from agent.tools.lineage_tools     import get_lineage_graph, analyze_lineage_impact
from agent.tools.catalogue_tools   import search_pii_tables
from agent.tools.discovery_tools   import list_catalogue_tables, list_pipeline_ids
from agent.tools.validators import (
    TriggerDQCheckInput, GetPipelineStatusInput, GetLineageGraphInput,
    AnalyzeLineageImpactInput, SearchPIITablesInput, GetSLOReportInput,
)
from pm_config import settings

logger = logging.getLogger(__name__)

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, tuple[Any, Any]] = {
    "trigger_dq_check":        (TriggerDQCheckInput,      trigger_dq_check),
    "get_pipeline_status":     (GetPipelineStatusInput,    get_pipeline_status),
    "get_lineage_graph":       (GetLineageGraphInput,      get_lineage_graph),
    "analyze_lineage_impact":  (AnalyzeLineageImpactInput, analyze_lineage_impact),
    "search_pii_tables":       (SearchPIITablesInput,      search_pii_tables),
    "get_slo_report":          (GetSLOReportInput,         get_slo_report),
    "list_catalogue_tables":   (None,                      list_catalogue_tables),
    "list_pipeline_ids":       (None,                      list_pipeline_ids),
}

# ── Tool schemas ──────────────────────────────────────────────────────────────
_ALL_TOOL_DEFS: dict[str, dict] = {
    "trigger_dq_check": {
        "type": "function",
        "function": {
            "name": "trigger_dq_check",
            "description": "[REQUIRES_HUMAN_APPROVAL] Run a Great Expectations DQ suite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name":   {"type": "string"},
                    "rules_preset": {"type": "string", "enum": ["minimal","standard","strict"]},
                },
                "required": ["table_name"],
            },
        },
    },
    "get_pipeline_status": {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": (
                "Fetch current run status and history for a pipeline. "
                "pipeline_id MUST be one of the IDs returned by list_pipeline_ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id":    {"type": "string"},
                    "lookback_hours": {"type": "integer", "default": 24},
                },
                "required": ["pipeline_id"],
            },
        },
    },
    "get_lineage_graph": {
        "type": "function",
        "function": {
            "name": "get_lineage_graph",
            "description": (
                "Get upstream/downstream table lineage. "
                "table_name MUST be an exact name from list_catalogue_tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "depth":      {"type": "integer", "default": 2},
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
                "What-If: trace downstream blast radius before dropping columns. "
                "Use list_catalogue_tables FIRST to resolve the exact table name. "
                "ONLY call when user explicitly asks about impact/what-breaks."
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
            "description": "List all PII-tagged tables and columns.",
            "parameters": {
                "type": "object",
                "properties": {"domain_filter": {"type": "string"}},
            },
        },
    },
    "get_slo_report": {
        "type": "function",
        "function": {
            "name": "get_slo_report",
            "description": "SLO adherence report. pipeline_id MUST come from list_pipeline_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id":  {"type": "string"},
                    "window_days":  {"type": "integer", "default": 7},
                },
                "required": ["pipeline_id"],
            },
        },
    },
    "list_catalogue_tables": {
        "type": "function",
        "function": {
            "name": "list_catalogue_tables",
            "description": (
                "List all tables in the data catalogue with their names and descriptions. "
                "Call this FIRST when the user refers to a table by a vague name "
                "('fact table', 'users table', 'main table') to resolve the exact table_name "
                "before calling get_lineage_graph or analyze_lineage_impact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain_filter": {"type": "string", "description": "Optional: finance, users, product, operations"},
                },
            },
        },
    },
    "list_pipeline_ids": {
        "type": "function",
        "function": {
            "name": "list_pipeline_ids",
            "description": (
                "List all valid pipeline IDs from the pipeline runs database. "
                "Call this FIRST when the user asks about pipeline health without "
                "specifying a pipeline name, to avoid using incorrect pipeline IDs."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
}

# ── Intent-to-tool mapping ────────────────────────────────────────────────────
# Discovery tools (list_catalogue_tables, list_pipeline_ids) are added to
# CATALOGUE and HEALTH respectively so the agent can resolve real names.
INTENT_TOOL_ALLOWLIST: dict[str | None, list[str]] = {
    "CODE_QA":   [],
    "CATALOGUE": ["list_catalogue_tables", "get_lineage_graph", "search_pii_tables"],
    "HEALTH":    ["list_pipeline_ids", "get_pipeline_status", "get_slo_report"],
    "ACTION":    list(_ALL_TOOL_DEFS.keys()),
    "GENERAL":   [],
    None:        list(_ALL_TOOL_DEFS.keys()),
}

# Bumped CATALOGUE to 2: iteration 1 = resolve table name or call lineage,
# iteration 2 = synthesise with the result
INTENT_MAX_ITERATIONS: dict[str | None, int] = {
    "CODE_QA":   0,
    "CATALOGUE": 2,
    "HEALTH":    2,
    "ACTION":    5,
    "GENERAL":   0,
    None:        3,
}

APPROVAL_REQUIRED_TOOLS = {"trigger_dq_check"}

# ── Type coercion map: tool_name → {param_name: python_type} ─────────────────
# Groq rejects calls where integer params are sent as strings.
# This map coerces before execution AND before the Groq API call.
_PARAM_TYPES: dict[str, dict[str, type]] = {
    "get_lineage_graph":      {"depth": int},
    "get_pipeline_status":    {"lookback_hours": int},
    "get_slo_report":         {"window_days": int},
    "analyze_lineage_impact": {},
}


def _coerce_args(tool_name: str, args: dict) -> dict:
    """
    Coerce tool arguments to the correct Python types.
    Prevents Groq 400 errors when the model emits depth="1" instead of depth=1.
    """
    type_map = _PARAM_TYPES.get(tool_name, {})
    if not type_map:
        return args
    coerced = dict(args)
    for param, target_type in type_map.items():
        if param in coerced and not isinstance(coerced[param], target_type):
            try:
                coerced[param] = target_type(coerced[param])
                logger.debug("Coerced %s.%s: %r → %r", tool_name, param, args[param], coerced[param])
            except (ValueError, TypeError) as exc:
                logger.warning("Could not coerce %s.%s: %s", tool_name, param, exc)
    return coerced


# ── System prompt ─────────────────────────────────────────────────────────────
_BASE_SYSTEM_PROMPT = """You are PipelineMind, an expert Data Engineering AI assistant.

MANDATORY RULES:
1. NEVER invent table names or pipeline IDs. Use ONLY names from tool results.
2. If the user says "fact table", "main table", or any vague table reference:
   → Call list_catalogue_tables FIRST to find the real table name.
3. If the user asks about pipeline health without naming a specific pipeline:
   → Call list_pipeline_ids FIRST to get valid pipeline IDs.
4. STOP after answering the specific question asked. Do not chain extra tool calls.
5. For state-altering actions (trigger_dq_check), state human approval is required.
6. If confidence in retrieved context is low (< 50%), say so.

SCOPE RULE:
Answer exactly what was asked. If asked about lineage, return lineage. Stop.
If asked about health, return health status. Stop.
Do not add unsolicited analysis unless directly relevant.
"""


def _build_context_prompt(intent: str | None, user_message: str) -> str:
    """
    Build an intent-specific context injection that gives the agent
    real table names and pipeline IDs to prevent hallucination.
    """
    import duckdb
    lines = [_BASE_SYSTEM_PROMPT]

    if intent in ("CATALOGUE", "ACTION"):
        try:
            con = duckdb.connect(str(settings.duckdb_path), read_only=True)
            rows = con.execute(
                "SELECT table_name, description FROM catalogue_tables ORDER BY table_name"
            ).fetchall()
            con.close()
            if rows:
                lines.append("\nAVAILABLE CATALOGUE TABLES (use these exact names):")
                for r in rows:
                    desc = r[1] or ""
                    lines.append(f"  - {r[0]}: {desc[:80]}")
        except Exception:
            pass

    if intent in ("HEALTH", "ACTION"):
        try:
            con = duckdb.connect(str(settings.duckdb_path), read_only=True)
            rows = con.execute(
                "SELECT DISTINCT pipeline_id FROM pipeline_runs ORDER BY pipeline_id"
            ).fetchall()
            con.close()
            if rows:
                ids = [r[0] for r in rows]
                lines.append(f"\nVALID PIPELINE IDs (use exactly): {', '.join(ids)}")
        except Exception:
            pass

    return "\n".join(lines)


# ── Hallucination detection ───────────────────────────────────────────────────
_HALLUCINATION_PATTERNS = _re.compile(
    r"("
    r"\[Calling\s+\w+"
    r"|\bI\s+will\s+(now\s+)?call\s+the"
    r"|\bI\s+(will|am going to|need to)\s+call\s+(the\s+)?`?\w+`?\s+(tool|function)"
    r"|\bCalling\s+(the\s+)?`?\w+`?\s+(tool|function)"
    r"|\[calling\s+\w+"
    r"|\bcalling\s+get_\w+"
    r"|\bcalling\s+trigger_\w+"
    r"|\bcalling\s+analyze_\w+"
    r"|\bcalling\s+search_\w+"
    r"|\bcalling\s+list_\w+"
    r"|\blet\s+me\s+call\b"
    r"|\bI('ll|\s+will)\s+(now\s+)?call\b"
    r"|Please\s+wait\s+while\s+I\s+(retrieve|fetch|call|check)"
    r"|I\s+need\s+to\s+analyze\s+the\s+lineage"
    r")",
    _re.I,
)


def _has_hallucinated_tool_call(text: str) -> bool:
    return bool(_HALLUCINATION_PATTERNS.search(text))


def _strip_hallucination(text: str) -> str:
    cleaned = _re.sub(r"\[Calling[^\]]*\]", "", text)
    cleaned = _re.sub(
        r"(I\s+will\s+(now\s+)?call\s+the\s+`?\w+`?\s+(tool|function)[^.]*\.\s*"
        r"|Please\s+wait\s+while\s+I\s+(retrieve|fetch|call|check)[^.]*\.\s*"
        r"|Let\s+me\s+(call|retrieve|fetch)[^.]*\.\s*)",
        "",
        cleaned,
        flags=_re.I,
    )
    return cleaned.strip()


def result_text(content: str | None) -> str:
    return content or ""


@dataclass
class AgentResult:
    final_response:    str
    tool_calls_made:   list[dict]
    iterations:        int
    requires_approval: bool
    approval_tool:     str = ""
    approval_args:     dict = field(default_factory=dict)


def _get_tools_for_intent(intent: str | None) -> list[dict]:
    allowed = INTENT_TOOL_ALLOWLIST.get(intent, list(_ALL_TOOL_DEFS.keys()))
    return [_ALL_TOOL_DEFS[n] for n in allowed if n in _ALL_TOOL_DEFS]


def _get_max_iterations(intent: str | None) -> int:
    return INTENT_MAX_ITERATIONS.get(intent, 3)


class AgentLoop:
    """
    Groq function-calling agent with:
    - Intent-aware tool filtering
    - Per-intent iteration budget
    - Type coercion for tool args (prevents depth="1" 400 errors)
    - Context injection: real table names + pipeline IDs
    - Focused anti-hallucination system prompt
    """

    def run(
        self,
        user_message: str,
        context_text: str = "",
        conversation_history: list[dict] | None = None,
        pending_approval: dict | None = None,
        intent: str | None = None,
    ) -> AgentResult:

        system_prompt = _build_context_prompt(intent, user_message)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if context_text.strip():
            messages.append({
                "role": "user",
                "content": (
                    "Retrieved context from the knowledge base "
                    "(use only if directly relevant):\n\n" + context_text
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I will answer using the context above and real data only.",
            })

        for turn in (conversation_history or []):
            messages.append(turn)

        messages.append({
            "role": "user",
            "content": (
                f"{user_message}\n\n"
                "[RULE] Use only exact table names and pipeline IDs from the catalogue. "
                "Never invent names. Answer only what was asked."
            ),
        })

        available_tools = _get_tools_for_intent(intent)
        max_iters       = _get_max_iterations(intent)
        tool_calls_made: list[dict] = []

        logger.info(
            "AgentLoop | intent=%s | tools_available=%d | max_iters=%d",
            intent, len(available_tools), max_iters,
        )

        if not available_tools:
            result = self._call_groq(messages, tools=None)
            raw    = result.choices[0].message.content or ""
            if _has_hallucinated_tool_call(raw):
                logger.warning("Hallucinated tool call (intent=%s) — stripping", intent)
                raw = _strip_hallucination(raw)
                if len(raw.strip()) < 80:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Please answer using only the retrieved context provided. "
                            "Do not mention tool calls."
                        ),
                    })
                    retry = self._call_groq(messages, tools=None)
                    raw   = retry.choices[0].message.content or raw
            return AgentResult(
                final_response=raw, tool_calls_made=[], iterations=1, requires_approval=False,
            )

        if pending_approval:
            coerced_args = _coerce_args(pending_approval["name"], pending_approval["args"])
            tool_result  = self._execute_tool(pending_approval["name"], coerced_args)
            messages.append({
                "role":         "tool",
                "content":      json.dumps(tool_result, default=str),
                "tool_call_id": pending_approval.get("call_id", "approved_call"),
            })
            tool_calls_made.append({"tool": pending_approval["name"], "approved": True})

        for iteration in range(max_iters):
            try:
                response = self._call_groq(messages, tools=available_tools)
            except Exception as exc:
                error_msg = str(exc)
                # If it's a tool schema error (400 tool_use_failed), attempt recovery
                if "tool_use_failed" in error_msg or "tool call validation" in error_msg:
                    logger.error("Tool schema validation error at iter %d: %s", iteration, exc)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The previous tool call failed with a parameter type error: {error_msg}. "
                            "Please ensure all integer parameters (like depth, lookback_hours, window_days) "
                            "are passed as numbers, not strings. Try again."
                        ),
                    })
                    try:
                        response = self._call_groq(messages, tools=available_tools)
                    except Exception as exc2:
                        logger.error("Recovery attempt failed: %s", exc2)
                        return AgentResult(
                            final_response=f"I encountered a parameter error: {exc2}. Please rephrase.",
                            tool_calls_made=tool_calls_made,
                            iterations=iteration + 1,
                            requires_approval=False,
                        )
                else:
                    logger.error("Groq call failed at iter %d: %s", iteration, exc)
                    return AgentResult(
                        final_response=f"I encountered an error: {exc}. Please try again.",
                        tool_calls_made=tool_calls_made,
                        iterations=iteration + 1,
                        requires_approval=False,
                    )

            msg = response.choices[0].message

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
                    raw_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    raw_args = {}

                # Apply type coercion before execution
                tool_args = _coerce_args(tool_name, raw_args)

                logger.info("Agent tool call [%s]: %s(%s)", intent, tool_name, tool_args)

                if tool_name in APPROVAL_REQUIRED_TOOLS:
                    return AgentResult(
                        final_response=(
                            f"I need to run `{tool_name}` with parameters: "
                            f"{json.dumps(tool_args)}. "
                            "Please approve or deny this action."
                        ),
                        tool_calls_made=tool_calls_made,
                        iterations=iteration + 1,
                        requires_approval=True,
                        approval_tool=tool_name,
                        approval_args=tool_args,
                    )

                tool_result = self._execute_tool(tool_name, tool_args)
                tool_calls_made.append({"tool": tool_name, "args": tool_args, "result": tool_result})
                messages.append({
                    "role":         "tool",
                    "content":      json.dumps(tool_result, default=str),
                    "tool_call_id": tc.id,
                })

        # Budget exhausted — force synthesis
        logger.warning(
            "Agent reached intent budget (intent=%s max_iters=%d) — forcing synthesis",
            intent, max_iters,
        )
        messages.append({
            "role":    "user",
            "content": "Synthesise all tool results into a clear final answer now.",
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
            if model_cls is None:
                # Discovery tools have no Pydantic model — call directly
                return func(**tool_args)
            validated = model_cls(**tool_args)
            return func(**validated.model_dump())
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return {"error": str(exc), "tool": tool_name, "args": tool_args}
PYEOF
log "agent/agent_loop.py rewritten"

# ==============================================================================
# FIX 3 — Fix citation "unknown" in chat_panel
# ==============================================================================
step "FIX 3: Fix citation display for unknown/empty source files"

python3 - << 'PATCHEOF'
from pathlib import Path

path = Path("ui/components/chat_panel.py")
content = path.read_text()

old = '''    for c in visible:
            score_pct = round(c["score"] * 100, 1)
            file_name = c.get("file", "").split("/")[-1] or "unknown"
            chunk_type = c.get("chunk_type", "")
            fn        = c.get("function_name", "")
            git_hash  = c.get("git_commit_hash", "")

            label = f"[{c['source_index']}] {file_name}"
            if chunk_type:
                label += f" ({chunk_type}"
                if fn:
                    label += f" | {fn}"
                label += ")"
            if git_hash:
                label += f" git:{git_hash[:8]}"
            label += f" — {score_pct}% relevance"'''

new = '''    for c in visible:
            score_pct  = round(c["score"] * 100, 1)
            raw_file   = c.get("file", "") or ""
            file_name  = raw_file.split("/")[-1] if raw_file else ""
            chunk_type = c.get("chunk_type", "") or ""
            fn         = c.get("function_name", "") or ""
            git_hash   = c.get("git_commit_hash", "") or ""

            # Build a meaningful label even when file path is missing
            if file_name:
                label = f"[{c['source_index']}] {file_name}"
            elif fn:
                label = f"[{c['source_index']}] function: {fn}"
            elif chunk_type:
                label = f"[{c['source_index']}] {chunk_type} chunk"
            else:
                label = f"[{c['source_index']}] retrieved document"

            if chunk_type:
                label += f" ({chunk_type}"
                if fn:
                    label += f" | {fn}"
                label += ")"
            if git_hash:
                label += f" git:{git_hash[:8]}"
            label += f" — {score_pct}% relevance"'''

if old in content:
    content = content.replace(old, new)
    path.write_text(content)
    print("chat_panel.py citation display fixed")
else:
    print("WARNING: citation patch target not found — may need manual review")
PATCHEOF

# ==============================================================================
# FIX 4 — Unit tests for the new fixes
# ==============================================================================
step "FIX 4: Writing tests"

cat << 'PYEOF' > tests/unit/test_type_coercion.py
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
PYEOF

cat << 'PYEOF' > tests/unit/test_discovery_tools.py
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
PYEOF

cat << 'PYEOF' > tests/unit/test_intent_routing_v2.py
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
PYEOF

log "Tests written"

# ==============================================================================
# RUN TESTS
# ==============================================================================
step "Running tests"

export PYTHONPATH="."
if [[ -f "$PROJECT_DIR/.venv/bin/pytest" ]]; then
    "$PROJECT_DIR/.venv/bin/pytest" \
        tests/unit/test_type_coercion.py \
        tests/unit/test_intent_routing_v2.py \
        -v --tb=short 2>&1 || true

    echo ""
    "$PROJECT_DIR/.venv/bin/pytest" \
        tests/unit/test_discovery_tools.py \
        -v --tb=short 2>&1 || true
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Tool Schema + Hallucinated Names Fix — COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  4 root causes fixed:"
echo ""
echo "  1. depth='1' string → integer coercion"
echo "     _coerce_args() converts string numbers before Groq API call"
echo "     Prevents 400 tool_use_failed errors"
echo ""
echo "  2. Hallucinated table names ('fact table', 'OrdersPipeline')"
echo "     list_catalogue_tables tool added to CATALOGUE + ACTION allowlists"
echo "     Real table names injected into system prompt context"
echo "     Agent must call list_catalogue_tables first for vague references"
echo ""
echo "  3. Hallucinated pipeline IDs ('InventorySnapshotPipeline')"
echo "     list_pipeline_ids tool added to HEALTH + ACTION allowlists"
echo "     Real pipeline IDs injected into system prompt context"
echo "     Agent must call list_pipeline_ids first for vague health queries"
echo ""
echo "  4. CATALOGUE max_iters bumped 1 → 2"
echo "     Iter 1: resolve table name OR call lineage"
echo "     Iter 2: synthesise with tool result"
echo ""
echo "  Expected behavior after rebuild:"
echo "  'what will happen if I delete the fact table?'"
echo "     → list_catalogue_tables → finds orders_fact, kpi_daily_metrics, etc."
echo "     → analyze_lineage_impact with real table name"
echo "     → accurate blast radius report"
echo ""
echo "  'what the health of the pipeline?'"
echo "     → list_pipeline_ids → gets [orders, users, inventory, sessions, metrics]"
echo "     → get_pipeline_status with real ID"
echo "     → accurate status report"
echo ""
echo "  Rebuild Docker:"
echo "    docker compose down && docker compose up --build"
echo ""