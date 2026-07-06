"""Pipeline health dashboard component."""
from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from ui.api_client import _API_BASE


def render_health_dashboard() -> None:
    st.header("Pipeline Health Dashboard")

    try:
        resp = httpx.get(f"{_API_BASE}/api/v1/pipelines", timeout=10)
        resp.raise_for_status()
        pipelines = resp.json()
    except Exception as exc:
        st.error(f"Could not reach API ({_API_BASE}): {exc}")
        return

    if not pipelines:
        st.info("No pipeline data available.")
        return

    cols = st.columns(len(pipelines))
    for col, p in zip(cols, pipelines):
        with col:
            st.metric(
                label=p["pipeline_id"],
                value=f"{p['success_rate']}%",
                delta=f"Last: {p['last_status']}",
            )

    st.divider()
    selected = st.selectbox("Drill into pipeline", [p["pipeline_id"] for p in pipelines])
    if selected:
        try:
            status = httpx.get(f"{_API_BASE}/api/v1/pipelines/{selected}/status", timeout=10).json()
            slo    = httpx.get(f"{_API_BASE}/api/v1/pipelines/{selected}/slo",    timeout=10).json()
        except Exception as exc:
            st.error(f"Failed to fetch details: {exc}")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Last Status", status.get("status", "N/A"))
        c2.metric("SLO %",       f"{slo.get('actual_pct', 0)}%")
        c3.metric("Compliant",   "Yes" if slo.get("compliant") else "No")

        if status.get("failures"):
            st.subheader("Recent Failures")
            st.dataframe(pd.DataFrame(status["failures"]))
