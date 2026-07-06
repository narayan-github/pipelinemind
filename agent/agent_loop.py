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
