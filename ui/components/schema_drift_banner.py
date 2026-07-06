"""Schema drift sidebar warning banner."""
from __future__ import annotations

import time
import httpx
import streamlit as st

from ui.api_client import _API_BASE

POLL_INTERVAL = 300


def render_drift_banner() -> None:
    now       = time.time()
    last_poll = st.session_state.get("drift_last_poll", 0)

    if now - last_poll > POLL_INTERVAL or "drift_events" not in st.session_state:
        try:
            resp = httpx.get(f"{_API_BASE}/api/v1/schema-drift", timeout=5)
            data = resp.json()
            st.session_state["drift_events"]    = data.get("drift_events", [])
            st.session_state["drift_last_poll"] = now
        except Exception:
            st.session_state.setdefault("drift_events", [])

    events = st.session_state.get("drift_events", [])
    if events:
        with st.sidebar:
            st.error(f"Schema Drift Detected — {len(events)} table(s) changed")
            for e in events:
                with st.expander(f"Table: {e['table']} ({e.get('severity','?')})"):
                    if e.get("dropped_columns"):
                        st.warning(f"Dropped: {', '.join(e['dropped_columns'])}")
                    if e.get("added_columns"):
                        st.info(f"Added: {', '.join(e['added_columns'])}")
