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

    # Record Prometheus metrics
    try:
        CHAT_REQUESTS_TOTAL.labels(
            intent=intent or "unknown",
            has_pii=str(has_pii),
        ).inc()
        RETRIEVAL_CONFIDENCE.observe(confidence_score)
        if low_confidence:
            LOW_CONFIDENCE_TOTAL.inc()
    except Exception:
        pass

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
