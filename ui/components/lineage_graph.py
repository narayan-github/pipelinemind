"""Interactive lineage DAG component."""
from __future__ import annotations

import httpx
import streamlit as st

from ui.api_client import _API_BASE

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False


def render_lineage_graph(table_name: str, depth: int = 2) -> None:
    try:
        resp = httpx.get(
            f"{_API_BASE}/api/v1/catalogue/lineage/{table_name}",
            params={"depth": depth}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        st.error(f"Failed to fetch lineage: {exc}")
        return

    if not AGRAPH_AVAILABLE:
        st.warning("streamlit-agraph not installed. Showing raw lineage data.")
        st.json(data)
        return

    nodes_data = data.get("nodes", [])
    edges_data = data.get("edges", [])
    pii_nodes  = set(data.get("pii_nodes", []))

    nodes = [
        Node(
            id=n["table"], label=n["table"], size=25,
            color="#FF4B4B" if n["table"] in pii_nodes else
                  ("#FFD700" if n["table"] == table_name else "#4B8BFF"),
            title=f"Domain: {n.get('domain','?')} | Rows: {n.get('row_count',0):,}",
        )
        for n in nodes_data
    ]
    edges = [
        Edge(source=e["source"], target=e["target"], label=e.get("transformation", ""))
        for e in edges_data
    ]
    agraph(nodes=nodes, edges=edges,
           config=Config(width=800, height=500, directed=True, physics=True))

    if pii_nodes:
        st.warning(f"PII-tagged nodes: {', '.join(pii_nodes)}", icon="🔒")
