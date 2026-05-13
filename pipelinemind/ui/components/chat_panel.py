"""
Streaming chat panel component.
Connects to the FastAPI SSE endpoint and renders streamed tokens.

Citation display improvements:
  - Scores shown as percentage (sigmoid-normalised, 0-100%)
  - Citations with score < 10% hidden (irrelevant documents)
  - PII warning only shown when a PII_HIGH column is explicitly referenced
  - Negative scores never shown
"""
from __future__ import annotations

import json
import httpx
import streamlit as st

API_BASE          = "http://localhost:8000"
MIN_DISPLAY_SCORE = 0.10   # hide citations below this sigmoid-normalised threshold


def _stream_chat(message: str, history: list[dict]) -> dict:
    """
    Call the FastAPI /api/v1/chat SSE endpoint and collect all events.
    Returns the final event payload.
    """
    full_text        = ""
    result_event     = {}
    approval_event   = {}
    retrieval_event  = {}
    current_event    = ""

    placeholder = st.empty()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/api/v1/chat",
            json={"message": message, "conversation_history": history},
        ) as response:
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "token":
                        full_text += data.get("text", "")
                        placeholder.markdown(full_text + "▌")
                    elif current_event == "retrieval_complete":
                        retrieval_event = data
                    elif current_event == "done":
                        result_event = data
                        placeholder.markdown(full_text)
                    elif current_event == "approval_required":
                        approval_event = data
                        placeholder.markdown(data.get("message", ""))

    return {
        "text":      full_text or approval_event.get("message", ""),
        "done":      result_event,
        "retrieval": retrieval_event,
        "approval":  approval_event,
    }


def _render_citations(citations: list[dict]) -> None:
    """Render citations, hiding irrelevant documents (score < MIN_DISPLAY_SCORE)."""
    visible = [c for c in citations if c.get("score", 0) >= MIN_DISPLAY_SCORE]
    if not visible:
        return

    with st.expander(f"Sources ({len(visible)} relevant)"):
        for c in visible:
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
            label += f" — {score_pct}% relevance"

            # Colour-code by relevance
            if score_pct >= 70:
                st.success(label, icon="✅")
            elif score_pct >= 40:
                st.info(label)
            else:
                st.caption(label)


def render_chat_panel() -> None:
    """Main chat panel with conversation history."""
    st.title("PipelineMind — Data Engineering Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "approval_pending" not in st.session_state:
        st.session_state.approval_pending = None

    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(msg["citations"])
            if msg.get("confidence_score") is not None:
                score = msg["confidence_score"]
                pct   = round(score * 100, 1)
                if score >= 0.7:
                    st.caption(f"Confidence: :green[{pct}%]")
                elif score >= 0.5:
                    st.caption(f"Confidence: :orange[{pct}%]")
                else:
                    st.caption(f"Confidence: :red[{pct}%] — retrieved context may be limited")
            if msg.get("intent"):
                st.caption(f"Intent: `{msg['intent']}`")
            # Tightened PII warning — only for explicit PII_HIGH references
            if msg.get("pii_warning"):
                st.warning(
                    "This response may reference PII columns (email, phone_number, "
                    "date_of_birth). Handle with care.",
                    icon="🔒",
                )

    # Pending approval gate
    if st.session_state.approval_pending:
        from ui.components.approval_gate import render_approval_gate
        ap = st.session_state.approval_pending
        render_approval_gate(
            tool_name=ap["tool_name"],
            tool_args=ap["tool_args"],
            call_id=ap.get("call_id", "pending"),
        )

    # Chat input
    if prompt := st.chat_input("Ask about your pipelines, data catalogue, or health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                result = _stream_chat(prompt, history)
            except Exception as exc:
                st.error(f"Connection error: {exc}")
                return

        msg_record: dict = {"role": "assistant", "content": result["text"]}

        ret = result.get("retrieval", {})
        if ret:
            # Filter citations before storing
            raw_citations = ret.get("citations", [])
            msg_record["citations"]       = [
                c for c in raw_citations if c.get("score", 0) >= MIN_DISPLAY_SCORE
            ]
            msg_record["confidence_score"] = ret.get("confidence_score")
            msg_record["intent"]           = ret.get("intent")
            # PII warning: only if has_pii AND score is meaningful
            top_score = ret.get("confidence_score", 0)
            msg_record["pii_warning"] = ret.get("has_pii", False) and top_score >= 0.5

        if result.get("approval"):
            ap = result["approval"]
            st.session_state.approval_pending = {
                "tool_name": ap.get("tool_name"),
                "tool_args": ap.get("tool_args", {}),
                "call_id":   ap.get("call_id", "pending"),
            }

        st.session_state.messages.append(msg_record)
        st.rerun()
