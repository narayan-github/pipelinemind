"""Human-in-the-loop approval gate component."""
from __future__ import annotations

import httpx
import streamlit as st

from ui.api_client import _API_BASE


def render_approval_gate(tool_name: str, tool_args: dict, call_id: str) -> None:
    st.warning("Agent Action Requires Approval", icon="⚠")
    st.markdown(f"**Tool:** `{tool_name}`")
    st.json(tool_args)

    col_allow, col_deny = st.columns(2)
    with col_allow:
        if st.button("Allow", type="primary", key=f"allow_{call_id}"):
            _submit_approval(tool_name, tool_args, call_id, approved=True)
    with col_deny:
        if st.button("Deny", type="secondary", key=f"deny_{call_id}"):
            _submit_approval(tool_name, tool_args, call_id, approved=False)


def _submit_approval(tool_name: str, tool_args: dict, call_id: str, approved: bool) -> None:
    try:
        resp = httpx.post(
            f"{_API_BASE}/api/v1/chat/approve",
            json={"tool_name": tool_name, "tool_args": tool_args,
                  "call_id": call_id, "approved": approved},
            timeout=60,
        )
        result = resp.json()
        if approved:
            response_text = result.get("result", "") or f"✅ `{tool_name}` executed successfully."
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state["approval_pending"] = None
            st.rerun()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Action denied. No changes were made."})
            st.session_state["approval_pending"] = None
            st.rerun()
    except Exception as exc:
        st.error(f"Approval submission failed: {exc}")
