"""
RAG evaluation runner.
Evaluates all three retrieval modes and prints a comparative report.
Run: python tests/eval/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.eval.rag_evaluator import RAGEvaluator, EvalReport
from tests.eval.eval_dataset  import EVAL_DATASET


def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_report(reports: list[EvalReport]) -> None:
    SEP = "─" * 90

    print()
    print("=" * 90)
    print("  PipelineMind RAG Pipeline Evaluation Report")
    print("=" * 90)
    print()
    print(f"  Evaluation set:   {len(EVAL_DATASET)} queries (CODE_QA + CATALOGUE + CONFIG)")
    print(f"  K (cutoff):       {reports[0].k}")
    print()
    print(SEP)
    print(f"  {'Mode':<22} {'MRR@K':>8} {'NDCG@K':>8} {'Recall@K':>10} {'Prec@K':>8} "
          f"{'p50 ms':>8} {'p95 ms':>8} {'N':>4}")
    print(SEP)

    for r in reports:
        print(
            f"  {r.mode:<22} "
            f"{r.mrr:>8.4f} "
            f"{r.ndcg:>8.4f} "
            f"{r.recall:>10.4f} "
            f"{r.precision:>8.4f} "
            f"{r.latency_p50:>8.1f} "
            f"{r.latency_p95:>8.1f} "
            f"{r.n_queries:>4}"
        )

    print(SEP)
    print()

    # Ablation lift table
    if len(reports) >= 2:
        base = reports[0]
        print("  Ablation Study — Lift over dense-only baseline:")
        print(f"  {'Mode':<22} {'MRR lift':>12} {'NDCG lift':>12} {'Recall lift':>13}")
        print("  " + "─" * 60)
        for r in reports[1:]:
            mrr_lift    = (r.mrr    - base.mrr)    / base.mrr    * 100 if base.mrr    else 0
            ndcg_lift   = (r.ndcg   - base.ndcg)   / base.ndcg   * 100 if base.ndcg   else 0
            recall_lift = (r.recall - base.recall)  / base.recall * 100 if base.recall else 0
            print(
                f"  {r.mode:<22} "
                f"{mrr_lift:>+11.1f}% "
                f"{ndcg_lift:>+11.1f}% "
                f"{recall_lift:>+12.1f}%"
            )
        print()

    # Per-query breakdown for lowest-scoring queries
    all_results = [qr for r in reports[-1:] for qr in r.per_query]
    low_scorers = sorted(all_results, key=lambda x: x.mrr)[:3]
    if low_scorers:
        print("  Lowest MRR queries (hybrid+rerank mode) — candidates for prompt tuning:")
        for qr in low_scorers:
            print(f"  [{qr.query_id}] MRR={qr.mrr:.3f}  '{qr.query[:65]}'")
        print()

    # Target assessment
    last = reports[-1]
    print("  Target Assessment (hybrid+rerank):")
    targets = [
        ("MRR@5",      last.mrr,       0.75, ">="),
        ("NDCG@5",     last.ndcg,      0.70, ">="),
        ("Recall@5",   last.recall,    0.80, ">="),
        ("p95 ms",     last.latency_p95, 500, "<="),
    ]
    for name, val, target, op in targets:
        if op == ">=":
            passed = val >= target
        else:
            passed = val <= target
        status = "PASS" if passed else "FAIL"
        bar    = _bar(min(1.0, val / target if target else 1.0))
        print(f"  [{status}] {name:<12} {val:>8.3f}  target{op}{target}  {bar}")
    print()
    print("=" * 90)


def main() -> None:
    print("Initialising RAG evaluator (loading ChromaDB + BM25 index)...")

    try:
        evaluator = RAGEvaluator(k=5)
    except Exception as exc:
        print(f"Failed to initialise evaluator: {exc}")
        print("Make sure the ingestion pipeline has been run: bash scripts/ingest_fast.sh")
        sys.exit(1)

    chroma_count = evaluator.dense.collection.count()
    if chroma_count == 0:
        print("ChromaDB is empty. Run ingestion first: bash scripts/ingest_fast.sh")
        sys.exit(1)

    print(f"ChromaDB: {chroma_count} documents indexed")
    print(f"BM25:     {'available' if evaluator.sparse.available else 'NOT AVAILABLE — run ingest'}")
    print()

    reports: list[EvalReport] = []
    for mode in ["dense_only", "hybrid", "hybrid_rerank"]:
        print(f"Running {mode}...", end=" ", flush=True)
        r = evaluator.evaluate(mode)
        reports.append(r)
        print(f"done (MRR={r.mrr:.4f}  NDCG={r.ndcg:.4f}  n={r.n_queries})")

    _print_report(reports)


if __name__ == "__main__":
    main()
