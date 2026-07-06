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
