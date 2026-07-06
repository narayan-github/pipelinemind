#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
export PYTHONPATH="."

echo "[PM] PipelineMind RAG Evaluation"
echo "[PM] Checking ChromaDB..."

CHROMA_COUNT=$(python - << 'PYEOF'
import chromadb, sys
sys.path.insert(0, ".")
from pm_config import settings
try:
    c = chromadb.PersistentClient(path=str(settings.chroma_path))
    coll = c.get_or_create_collection("pipelinemind", metadata={"hnsw:space": "cosine"})
    print(coll.count())
except Exception:
    print(0)
PYEOF
)

if [[ "$CHROMA_COUNT" -eq 0 ]]; then
    echo "[WARN] ChromaDB is empty. Running fast ingestion first..."
    bash scripts/ingest_fast.sh
fi

echo "[PM] ChromaDB documents: $CHROMA_COUNT"
echo "[PM] Running ablation study..."
python tests/eval/run_eval.py
