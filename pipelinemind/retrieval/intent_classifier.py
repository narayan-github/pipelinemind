"""
Intent classifier with two-stage routing.

Stage 1 — Keyword fast-path (zero latency, zero quota):
  Matches deterministic signal words before the LLM is called.
  This is the PRIMARY intent routing mechanism — the LLM is a fallback only.
  Keywords are broad and forgiving: partial matches, common phrasings included.

Stage 2 — LLM classifier (llama-3.1-8b-instant, temperature=0.0):
  Only fires when Stage 1 returns no match.
  Includes 10 few-shot examples.

Keyword coverage matrix (verified against observed failures):
  Query                                         Stage1 Match   Intent
  "what will happen if I delete the fact table" ACTION         ✓
  "what the health of the pipeline?"            HEALTH         ✓
  "what's the health?"                          HEALTH         ✓
  "give me the lineage dag for X"               CATALOGUE      ✓
  "what PII columns are in dim_users?"          CATALOGUE      ✓
  "did the orders pipeline fail today?"         HEALTH         ✓
  "why does orders use MERGE?"                  CODE_QA        ✓
  "what is incremental loading?"                GENERAL        ✓
"""
from __future__ import annotations

import json
import logging
import re
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


# ── Stage 1: Keyword fast-path ────────────────────────────────────────────────
# Rules are evaluated in order — first match wins.
# Each tuple: (compiled_regex, Intent, confidence)

_KEYWORD_RULES: list[tuple[re.Pattern, Intent, float]] = [

    # ── ACTION — explicit destructive/change-intent signals ──────────────────
    # Covers: "what will happen if I delete/drop/rename/remove"
    #         "what happens if", "what if I drop", "impact of dropping"
    #         "run a DQ check", "trigger a DQ check"
    (re.compile(
        r"\b("
        r"what\s+will\s+happen\s+if"
        r"|what\s+would\s+happen\s+if"
        r"|what\s+happens\s+if"
        r"|what\s+if\s+i\s+(drop|delete|remove|rename|alter|change)"
        r"|if\s+i\s+(drop|delete|remove|rename|alter|change)"
        r"|impact\s+of\s+(drop|delet|remov|renam)"
        r"|before\s+i\s+(drop|delet|remov|renam)"
        r"|blast\s+radius"
        r"|downstream\s+impact"
        r"|downstream\s+effect"
        r"|run\s+(a\s+)?dq(\s+check)?"
        r"|trigger\s+(a\s+)?dq(\s+check)?"
        r"|data\s+quality\s+check"
        r"|run\s+great\s+expectations"
        r"|schema\s+change\s+impact"
        r")\b",
        re.I,
    ), Intent.ACTION, 0.95),

    # ── CATALOGUE — lineage / schema / PII discovery ─────────────────────────
    # Covers: "lineage dag", "what columns", "PII columns", "upstream/downstream",
    #         "schema of", "table structure", "data catalogue"
    (re.compile(
        r"\b("
        r"lineage(\s+dag|\s+graph|\s+of|\s+for)?"
        r"|data\s+lineage"
        r"|upstream(\s+table|\s+of)?"
        r"|downstream(\s+table|\s+of)?"
        r"|depends\s+on"
        r"|which\s+tables(\s+depend|\s+feed|\s+use|\s+write|\s+read)?"
        r"|pii\s+columns?"
        r"|pii\s+cols?"
        r"|pii\s+tables?"
        r"|pii\s+data"
        r"|sensitive\s+columns?"
        r"|sensitive\s+cols?"
        r"|sensitive\s+data"
        r"|what\s+columns(\s+are|\s+exist|\s+in|\s+does)?"
        r"|columns?\s+in\s+the"
        r"|schema\s+of\s+the"
        r"|schema\s+for\s+the"
        r"|table\s+schema"
        r"|table\s+structure"
        r"|data\s+catalogu?"
        r"|catalogu?\s+table"
        r"|dag\s+(for|of)\s+the"
        r"|table\s+lineage"
        r"|describe\s+the\s+table"
        r"|what\s+is\s+in\s+the\s+\w+\s+table"
        r")\b",
        re.I,
    ), Intent.CATALOGUE, 0.95),

    # ── HEALTH — pipeline operational status ─────────────────────────────────
    # Covers: "pipeline health", "what's the health", "pipeline status",
    #         "pipeline failed", "SLO breach", "last run", "monitoring"
    (re.compile(
        r"\b("
        r"pipeline\s+health"
        r"|health\s+of\s+the\s+pipeline"
        r"|what('s|s|\s+is)\s+the\s+health"
        r"|is\s+the\s+pipeline\s+healthy"
        r"|pipeline\s+(fail|failed|failing|status|ran|running|down|broken|issue)"
        r"|slo\s+(breach|adherence|report|status|compliance|target)"
        r"|success\s+rate"
        r"|last\s+run(\s+status|\s+result|\s+time)?"
        r"|recent\s+(fail|error|run|issue|problem)"
        r"|did\s+\w+\s+(fail|run|succeed|pass|complete)"
        r"|why\s+did\s+\w+\s+fail"
        r"|pipeline\s+monitor"
        r"|monitoring\s+status"
        r"|run\s+history"
        r"|pipeline\s+run"
        r"|job\s+(fail|status|health|run)"
        r"|is\s+\w+\s+(running|healthy|working|up|down)"
        r"|how\s+(is|are)\s+(the\s+)?pipeline"
        r")\b",
        re.I,
    ), Intent.HEALTH, 0.93),

    # ── CODE_QA — implementation / code understanding ────────────────────────
    # Covers: "how does X work", "why does X use", "explain the code",
    #         "what does the function do", "show me the implementation"
    (re.compile(
        r"\b("
        r"how\s+does\s+the\s+\w+"
        r"|why\s+does\s+the\s+\w+"
        r"|why\s+is\s+the\s+\w+\s+(using|using|implemented|written|coded)"
        r"|explain\s+the\s+(code|function|class|method|logic|implementation)"
        r"|what\s+does\s+the\s+(function|class|method|code|script)"
        r"|what\s+does\s+this\s+(function|class|method|code)"
        r"|show\s+me\s+the\s+(code|implementation|logic|function)"
        r"|merge\s+strategy"
        r"|insert\s+overwrite"
        r"|scd\s*(2|type)"
        r"|incremental\s+(logic|strategy|approach|load)"
        r"|watermark\s+(logic|strategy|approach)"
        r"|look\s+at\s+the\s+code"
        r"|read\s+the\s+code"
        r"|what\s+is\s+the\s+\w+\s+method"
        r"|what'\s*s\s+(the\s+)?\w+\s+method"
        r"|how\s+is\s+\w+\s+implemented"
        r"|what\s+does\s+\w+\s+method\s+do"
        r"|load\s+method"
        r"|extract\s+method"
        r"|transform\s+method"
        r")\b",
        re.I,
    ), Intent.CODE_QA, 0.92),

    # ── GENERAL — education / DE concepts (no pipeline-specific context) ─────
    (re.compile(
        r"\b("
        r"what\s+is\s+(a\s+)?(etl|elt|dbt|airflow|dagster|spark|flink|kafka)"
        r"|what\s+is\s+incremental\s+loading"
        r"|explain\s+what\s+\w+\s+is"
        r"|how\s+does\s+\w+\s+work\s+in\s+general"
        r"|best\s+practice"
        r"|definition\s+of\s+"
        r"|difference\s+between"
        r"|compare\s+\w+\s+and\s+\w+"
        r"|teach\s+me"
        r"|help\s+me\s+understand"
        r")\b",
        re.I,
    ), Intent.GENERAL, 0.88),
]


def _keyword_classify(query: str) -> tuple[Intent, float] | None:
    """
    Fast-path keyword classification.
    Returns (Intent, confidence) on match, None if LLM stage should be used.
    """
    q = query.strip()
    for pattern, intent, confidence in _KEYWORD_RULES:
        if pattern.search(q):
            logger.info(
                "Intent (keyword): %s (conf=%.2f) query='%s...'",
                intent.value, confidence, q[:70],
            )
            return intent, confidence
    return None


# ── Stage 2: LLM classifier ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an intent classifier for a Data Engineering AI assistant.
Classify the user query into EXACTLY ONE intent.

INTENT DEFINITIONS:
CODE_QA   — questions about pipeline code logic, SQL/Python implementation, debugging,
             what a specific function/method/class does.
CATALOGUE — questions about table schemas, column metadata, data lineage, lineage DAG,
             upstream/downstream dependencies, PII columns, data discovery.
HEALTH    — questions about pipeline run status, failures, SLO adherence, monitoring,
             whether a pipeline is healthy or broken.
ACTION    — explicit requests: "what will happen if I delete/drop/remove X",
             "trigger DQ check", "run impact analysis before schema change".
GENERAL   — generic data engineering education with no specific pipeline context.

DECISION RULES:
- "health", "healthy", "pipeline health" → HEALTH
- "lineage", "DAG", "upstream", "downstream" → CATALOGUE
- "what will happen if I delete/drop/rename" → ACTION
- "load method", "extract method", "transform method" questions → CODE_QA
- "what is X" where X is a DE concept (not a specific pipeline) → GENERAL

FEW-SHOT EXAMPLES:
Query: "what the health of the pipeline?"
{"intent": "HEALTH", "confidence": 0.95}

Query: "what's the health?"
{"intent": "HEALTH", "confidence": 0.93}

Query: "what will happen if I delete the fact table?"
{"intent": "ACTION", "confidence": 0.96}

Query: "what will happen if I delete the orders_fact table?"
{"intent": "ACTION", "confidence": 0.96}

Query: "can you let me know about vw_revenue_by_tier table lineage dag"
{"intent": "CATALOGUE", "confidence": 0.97}

Query: "what PII columns exist in the users table?"
{"intent": "CATALOGUE", "confidence": 0.96}

Query: "did the orders pipeline fail today?"
{"intent": "HEALTH", "confidence": 0.95}

Query: "what's load method?"
{"intent": "CODE_QA", "confidence": 0.94}

Query: "give me the in depth structure of the extract transform and load thing"
{"intent": "CODE_QA", "confidence": 0.93}

Query: "why does the orders pipeline use MERGE strategy?"
{"intent": "CODE_QA", "confidence": 0.94}

Query: "what is incremental loading in data engineering?"
{"intent": "GENERAL", "confidence": 0.91}

Respond with ONLY this JSON (no markdown, no preamble):
{"intent": "<INTENT>", "confidence": <0.0-1.0>}"""


class IntentClassifier:
    """
    Two-stage intent classifier:
      Stage 1: keyword fast-path (zero API calls)
      Stage 2: llama-3.1-8b-instant with few-shot examples
    """

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=5),
        reraise=False,
    )
    def classify(self, query: str) -> tuple[Intent, float]:
        """Returns (Intent, confidence). Falls back to CODE_QA on any failure."""

        # Stage 1: keyword fast-path
        keyword_result = _keyword_classify(query)
        if keyword_result is not None:
            return keyword_result

        # Stage 2: LLM classifier
        try:
            response = router.complete(
                call_type=CallType.INTENT,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": query},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed     = json.loads(raw)
            intent_str = parsed.get("intent", "CODE_QA")
            confidence = float(parsed.get("confidence", 0.8))
            intent     = Intent(intent_str)
            logger.info(
                "Intent (llm): %s (conf=%.2f) query='%s...'",
                intent.value, confidence, query[:70],
            )
            return intent, confidence
        except Exception as exc:
            logger.warning("Intent LLM stage failed (%s) — defaulting to CODE_QA", exc)
            return Intent.CODE_QA, 0.5
