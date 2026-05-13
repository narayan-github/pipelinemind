#!/usr/bin/env bash
# ==============================================================================
# PipelineMind — Docker Networking Fix
# Root cause: Streamlit UI uses hardcoded "http://localhost:8000" which is
# the UI container itself inside Docker. Must use "http://api:8000" (Docker
# service name) when running in containers, "http://localhost:8000" locally.
# Fix: read API_BASE_URL from environment with a sensible default.
# Also fixes Grafana dashboard provisioning (dashboard not found error).
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
# FIX 1 — All UI components: replace hardcoded localhost with env var
# API_BASE_URL defaults to http://localhost:8000 (local dev)
# Docker Compose sets it to http://api:8000 (container networking)
# ==============================================================================
step "FIX 1: Replace hardcoded API URLs in all UI components"

cat << 'PYEOF' > ui/api_client.py
"""
Centralised API base URL resolver.
Reads API_BASE_URL from the environment.

Local dev:  API_BASE_URL not set -> http://localhost:8000
Docker:     API_BASE_URL=http://api:8000 (set via docker-compose.yml)

All UI components import _API_BASE from here — never hardcode the URL.
"""
from __future__ import annotations

import os

_API_BASE: str = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
PYEOF
log "ui/api_client.py written"

# ── chat_panel.py ─────────────────────────────────────────────────────────────
cat << 'PYEOF' > ui/components/chat_panel.py
"""Streaming chat panel component."""
from __future__ import annotations

import json
import httpx
import streamlit as st

from ui.api_client import _API_BASE

MIN_DISPLAY_SCORE = 0.10


def _stream_chat(message: str, history: list[dict]) -> dict:
    full_text       = ""
    result_event    = {}
    approval_event  = {}
    retrieval_event = {}
    current_event   = ""

    placeholder = st.empty()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{_API_BASE}/api/v1/chat",
            json={"message": message, "conversation_history": history},
        ) as response:
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
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
    visible = [c for c in citations if c.get("score", 0) >= MIN_DISPLAY_SCORE]
    if not visible:
        return
    with st.expander(f"Sources ({len(visible)} relevant)"):
        for c in visible:
            score_pct  = round(c["score"] * 100, 1)
            file_name  = c.get("file", "").split("/")[-1] or "unknown"
            chunk_type = c.get("chunk_type", "")
            fn         = c.get("function_name", "")
            git_hash   = c.get("git_commit_hash", "")
            label      = f"[{c['source_index']}] {file_name}"
            if chunk_type:
                label += f" ({chunk_type}" + (f" | {fn}" if fn else "") + ")"
            if git_hash:
                label += f" git:{git_hash[:8]}"
            label += f" — {score_pct}% relevance"
            if score_pct >= 70:
                st.success(label, icon="✅")
            elif score_pct >= 40:
                st.info(label)
            else:
                st.caption(label)


def render_chat_panel() -> None:
    st.title("PipelineMind — Data Engineering Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "approval_pending" not in st.session_state:
        st.session_state.approval_pending = None

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
            if msg.get("pii_warning"):
                st.warning(
                    "This response may reference PII columns. Handle with care.",
                    icon="🔒",
                )

    if st.session_state.approval_pending:
        from ui.components.approval_gate import render_approval_gate
        ap = st.session_state.approval_pending
        render_approval_gate(
            tool_name=ap["tool_name"],
            tool_args=ap["tool_args"],
            call_id=ap.get("call_id", "pending"),
        )

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
            raw_citations = ret.get("citations", [])
            msg_record["citations"]       = [c for c in raw_citations if c.get("score", 0) >= MIN_DISPLAY_SCORE]
            msg_record["confidence_score"] = ret.get("confidence_score")
            msg_record["intent"]           = ret.get("intent")
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
PYEOF
log "ui/components/chat_panel.py fixed"

# ── approval_gate.py ──────────────────────────────────────────────────────────
cat << 'PYEOF' > ui/components/approval_gate.py
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
PYEOF
log "ui/components/approval_gate.py fixed"

# ── health_dashboard.py ────────────────────────────────────────────────────────
cat << 'PYEOF' > ui/components/health_dashboard.py
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
PYEOF
log "ui/components/health_dashboard.py fixed"

# ── lineage_graph.py ──────────────────────────────────────────────────────────
cat << 'PYEOF' > ui/components/lineage_graph.py
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
PYEOF
log "ui/components/lineage_graph.py fixed"

# ── schema_drift_banner.py ────────────────────────────────────────────────────
cat << 'PYEOF' > ui/components/schema_drift_banner.py
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
PYEOF
log "ui/components/schema_drift_banner.py fixed"

# ── Catalogue page ─────────────────────────────────────────────────────────────
cat << 'PYEOF' > ui/pages/03_Catalogue.py
"""Page 3: Data Catalogue Browser"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import pandas as pd
import streamlit as st

from ui.api_client import _API_BASE
from ui.components.lineage_graph       import render_lineage_graph
from ui.components.schema_drift_banner import render_drift_banner

render_drift_banner()
st.header("Data Catalogue Browser")

try:
    tables = httpx.get(f"{_API_BASE}/api/v1/catalogue/tables", timeout=10).json()
except Exception as exc:
    st.error(f"API unavailable ({_API_BASE}): {exc}")
    tables = []

if tables:
    pii_tables = [t for t in tables if t.get("pii_flag")]
    if pii_tables:
        st.warning(f"{len(pii_tables)} table(s) contain PII columns.", icon="🔒")

    selected = st.selectbox("Select a table", [t["table_name"] for t in tables])
    if selected:
        try:
            detail = httpx.get(f"{_API_BASE}/api/v1/catalogue/tables/{selected}", timeout=10).json()
            tbl    = detail.get("table", {})
            cols   = detail.get("columns", [])

            c1, c2, c3 = st.columns(3)
            c1.metric("Domain", tbl.get("domain", "N/A"))
            c2.metric("Rows",   f"{tbl.get('row_count', 0):,}")
            c3.metric("PII",    "Yes" if tbl.get("pii_flag") else "No")

            st.markdown(f"**Description:** {tbl.get('description', 'N/A')}")
            st.dataframe(pd.DataFrame(cols), use_container_width=True)

            st.subheader("Lineage DAG")
            depth = st.slider("Lineage depth", 1, 4, 2)
            render_lineage_graph(selected, depth)
        except Exception as exc:
            st.error(f"Failed to load table detail: {exc}")
PYEOF
log "ui/pages/03_Catalogue.py fixed"

# ==============================================================================
# FIX 2 — docker-compose.yml: set API_BASE_URL + add Grafana provisioning
# ==============================================================================
step "FIX 2: Rewriting docker-compose.yml with API_BASE_URL env var"

# Create Grafana provisioning directory structure
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# Prometheus datasource
cat << 'YAMLEOF' > monitoring/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
YAMLEOF

# Dashboard provisioner
cat << 'YAMLEOF' > monitoring/grafana/provisioning/dashboards/pipelinemind.yml
apiVersion: 1
providers:
  - name: PipelineMind
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
YAMLEOF

# Copy dashboard JSON to the provisioned path
cp monitoring/grafana_dashboard.json monitoring/grafana/dashboards/pipelinemind.json
log "Grafana provisioning config written"

# Full docker-compose.yml
cat << 'DCEOF' > docker-compose.yml
version: "3.9"

x-common-env: &common-env
  env_file: .env
  restart: unless-stopped

services:

  # ── One-shot: seed DuckDB (runs once, exits 0) ──────────────────────────────
  seeder:
    <<: *common-env
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: pipelinemind_seeder
    volumes:
      - pipelinemind_data:/app/data
    environment:
      PYTHONPATH: "."
    command: ["python", "db/seeder.py"]
    restart: "no"

  # ── One-shot: build ChromaDB + BM25 index ───────────────────────────────────
  ingest:
    <<: *common-env
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: pipelinemind_ingest
    volumes:
      - pipelinemind_data:/app/data
    environment:
      PYTHONPATH: "."
    command: [
      "python", "ingestion/ingest_pipeline.py",
      "--repo-path", "./data/pipeline_repo",
      "--sql-path",  "./data/sql",
      "--yaml-path", "./data/dags",
      "--dbt-path",  "./data/dbt_project",
      "--skip-summaries",
      "--force-reindex"
    ]
    restart: "no"
    depends_on:
      seeder:
        condition: service_completed_successfully

  # ── FastAPI backend ──────────────────────────────────────────────────────────
  api:
    <<: *common-env
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: pipelinemind_api
    ports:
      - "8000:8000"
    volumes:
      - pipelinemind_data:/app/data
      - ./logs:/app/logs
    environment:
      PYTHONPATH: "."
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 20s
      timeout: 10s
      retries: 5
      start_period: 60s
    depends_on:
      seeder:
        condition: service_completed_successfully

  # ── Streamlit UI ─────────────────────────────────────────────────────────────
  ui:
    <<: *common-env
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: pipelinemind_ui
    ports:
      - "8501:8501"
    volumes:
      - pipelinemind_data:/app/data
    environment:
      PYTHONPATH: "."
      # KEY FIX: use Docker service name, not localhost
      API_BASE_URL: "http://api:8000"
    depends_on:
      api:
        condition: service_healthy

  # ── Prometheus ───────────────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: pipelinemind_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=7d"
    restart: unless-stopped
    depends_on:
      api:
        condition: service_healthy

  # ── Grafana ───────────────────────────────────────────────────────────────────
  grafana:
    image: grafana/grafana:10.4.0
    container_name: pipelinemind_grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      # Provision datasource + dashboard automatically
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: pipelinemind
      GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH: /var/lib/grafana/dashboards/pipelinemind.json
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  # Named volume so data persists across container restarts
  pipelinemind_data:
  prometheus_data:
  grafana_data:
DCEOF
log "docker-compose.yml rewritten (named volume + API_BASE_URL + Grafana provisioning)"

# ==============================================================================
# FIX 3 — Dockerfiles: ensure PYTHONPATH is baked in as ENV default
# ==============================================================================
step "FIX 3: Updating Dockerfiles with ENV PYTHONPATH"

cat << 'DFAPI' > Dockerfile.api
FROM python:3.11-slim AS base
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --default-timeout=100 --upgrade pip uv

COPY pyproject.toml .
RUN uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        groq chromadb rank-bm25 "sentence-transformers>=3.0.0" \
        tree-sitter tree-sitter-python \
        fastapi "uvicorn[standard]" sse-starlette \
        duckdb "pydantic>=2" pydantic-settings \
        structlog prometheus-client watchdog \
        sqlalchemy pandas numpy scikit-learn \
        tenacity httpx python-dotenv pyyaml \
        great-expectations mcp \
    2>/dev/null || true

COPY . .
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --retries=5 --start-period=60s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop"]
DFAPI

cat << 'DFUI' > Dockerfile.ui
FROM python:3.11-slim AS base
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --default-timeout=100 --upgrade pip uv

COPY pyproject.toml .
RUN uv pip install --system \
        groq chromadb rank-bm25 "sentence-transformers>=3.0.0" \
        streamlit duckdb "pydantic>=2" pydantic-settings \
        structlog httpx python-dotenv pyyaml streamlit-agraph \
    2>/dev/null || true

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
DFUI
log "Dockerfiles updated with ENV PYTHONPATH=/app"

# ==============================================================================
# FIX 4 — Remove the obsolete 'version:' key from docker-compose to silence warning
# ==============================================================================
step "FIX 4: Remove obsolete 'version' key from docker-compose.yml"

python3 - << 'PYEOF'
from pathlib import Path
p = Path("docker-compose.yml")
lines = p.read_text().splitlines()
lines = [l for l in lines if not l.strip().startswith("version:")]
p.write_text("\n".join(lines) + "\n")
print("Removed 'version:' line from docker-compose.yml")
PYEOF

# ==============================================================================
# FIX 5 — scripts: set API_BASE_URL for local dev too (so scripts work either way)
# ==============================================================================
step "FIX 5: Update local scripts to export API_BASE_URL"

cat << 'SHEOF' > scripts/start_ui.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
export API_BASE_URL="http://localhost:8000"
echo "[PM] Starting Streamlit on http://localhost:8501"
echo "[PM] API endpoint: $API_BASE_URL"
streamlit run ui/app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false
SHEOF

cat << 'SHEOF' > scripts/start_api.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."
[[ -f data/pipelinemind.db ]] || python db/seeder.py
echo "[PM] Starting FastAPI on http://localhost:8000"
echo "[PM] API docs: http://localhost:8000/docs"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
SHEOF
chmod +x scripts/*.sh
log "Scripts updated"

# ==============================================================================
# REBUILD AND RESTART DOCKER STACK
# ==============================================================================
step "Stopping existing containers"

if command -v docker &>/dev/null; then
    docker compose down --remove-orphans 2>/dev/null || true
    log "Containers stopped"
else
    die "Docker not found"
fi

step "Rebuilding and starting the full stack"

echo ""
echo "Running: docker compose up --build -d"
echo "(this rebuilds only changed layers — should be fast)"
echo ""

docker compose up --build -d

echo ""
log "Waiting 15 s for containers to initialise..."
sleep 15

step "Health check"

API_OK=false
for i in 1 2 3 4 5; do
    if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        API_OK=true
        break
    fi
    echo "  Waiting for API... attempt $i/5"
    sleep 8
done

if $API_OK; then
    HEALTH=$(curl -s http://localhost:8000/api/v1/health)
    log "API healthy: $HEALTH"
else
    echo ""
    echo "API not yet healthy — check logs: docker compose logs api"
    echo "The containers are starting in the background."
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Docker Networking Fix — COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Root cause:"
echo "  Streamlit UI used hardcoded 'http://localhost:8000' which inside Docker"
echo "  resolves to the UI container itself, not the API container."
echo ""
echo "  Fixes applied:"
echo "  1. ui/api_client.py         New module — reads API_BASE_URL from env"
echo "  2. All UI components        Import _API_BASE from api_client instead of hardcoding"
echo "  3. docker-compose.yml       API_BASE_URL=http://api:8000 set for UI container"
echo "  4. docker-compose.yml       Named volume so data persists across restarts"
echo "  5. docker-compose.yml       Grafana provisioning wired (datasource + dashboard)"
echo "  6. Dockerfile.api/ui        ENV PYTHONPATH=/app baked in"
echo "  7. docker-compose.yml       Removed obsolete 'version:' key (silences warning)"
echo ""
echo "  Access points:"
echo "    Streamlit UI:  http://localhost:8501"
echo "    FastAPI docs:  http://localhost:8000/docs"
echo "    Prometheus:    http://localhost:9090"
echo "    Grafana:       http://localhost:3000   (admin / pipelinemind)"
echo ""
echo "  View live logs:"
echo "    docker compose logs -f ui"
echo "    docker compose logs -f api"
echo ""
echo "  If the UI still shows 'Connection refused', wait 30 s then:"
echo "    docker compose restart ui"
echo ""