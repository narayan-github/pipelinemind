"""
RAG evaluation engine.

Metrics computed:
  MRR@K    — Mean Reciprocal Rank: how high the first relevant document ranks
  NDCG@K   — Normalized Discounted Cumulative Gain: quality of full ranking
  Recall@K — Fraction of relevant docs found in top-K
  Precision@K — Fraction of top-K docs that are relevant

Ablation modes:
  dense_only      — ChromaDB HNSW without BM25, without re-ranking
  hybrid          — Dense + BM25 + RRF, without re-ranking
  hybrid_rerank   — Dense + BM25 + RRF + cross-encoder (full pipeline)

For each mode: latency (p50, p95) is measured across the eval set.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, quantiles
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pm_config import settings
from retrieval.chroma_retriever import ChromaRetriever, RetrievedChunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf_fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from retrieval.hyde import HyDEProcessor
from tests.eval.eval_dataset import EvalQuery, EVAL_DATASET

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query_id:        str
    query:           str
    intent:          str
    retrieved_ids:   list[str]
    retrieved_docs:  list[str]
    latency_ms:      float
    relevant_found:  list[bool]      # per position: is position i relevant?
    mrr:             float = 0.0
    ndcg:            float = 0.0
    recall:          float = 0.0
    precision:       float = 0.0


@dataclass
class EvalReport:
    mode:         str
    k:            int
    mrr:          float
    ndcg:         float
    recall:       float
    precision:    float
    latency_p50:  float
    latency_p95:  float
    n_queries:    int
    per_query:    list[QueryResult] = field(default_factory=list)


def _is_relevant(chunk: RetrievedChunk, query: EvalQuery) -> bool:
    """
    Heuristic relevance check: a chunk is relevant if any of the
    query's relevant_files appear in the chunk's source_file path,
    or any relevant_term appears in the chunk's document text.
    """
    doc_lower = (chunk.document + " " + chunk.raw_implementation).lower()
    src_lower  = chunk.source_file.lower()

    for rf in query.relevant_files:
        if rf.lower() in src_lower:
            return True
    for term in query.relevant_terms:
        if term.lower() in doc_lower:
            return True
    return False


def _mrr_at_k(relevant_flags: list[bool]) -> float:
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(relevant_flags: list[bool]) -> float:
    dcg  = sum(1.0 / math.log2(i + 2) for i, r in enumerate(relevant_flags) if r)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(sum(relevant_flags)))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(relevant_flags: list[bool], total_relevant: int) -> float:
    found = sum(relevant_flags)
    return found / total_relevant if total_relevant > 0 else 0.0


def _precision_at_k(relevant_flags: list[bool]) -> float:
    return sum(relevant_flags) / len(relevant_flags) if relevant_flags else 0.0


class RAGEvaluator:
    """
    Runs the evaluation dataset through configurable retrieval pipelines
    and computes MRR@K, NDCG@K, Recall@K, Precision@K, and latency percentiles.
    """

    def __init__(self, k: int = 5) -> None:
        self.k        = k
        self.dense    = ChromaRetriever()
        self.sparse   = BM25Retriever()
        self.reranker = Reranker()
        self.hyde     = HyDEProcessor()

    # ── Retrieval pipelines ───────────────────────────────────────────────────

    def _dense_only(self, query: str) -> list[RetrievedChunk]:
        """Baseline: dense retrieval only, no BM25, no re-ranking."""
        return self.dense.retrieve(query, top_k=self.k)

    def _hybrid(self, query: str) -> list[RetrievedChunk]:
        """Dense + BM25 + RRF, no re-ranking."""
        dense_results  = self.dense.retrieve(query, top_k=self.k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=self.k * 2)
        return reciprocal_rank_fusion(dense_results, sparse_results, top_n=self.k)

    def _hybrid_rerank(self, query: str) -> list[RetrievedChunk]:
        """Full pipeline: Dense + BM25 + RRF + cross-encoder re-ranking."""
        dense_results  = self.dense.retrieve(query, top_k=self.k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=self.k * 2)
        fused          = reciprocal_rank_fusion(dense_results, sparse_results, top_n=self.k * 2)
        return self.reranker.rerank(query, fused, top_k=self.k)

    # ── Evaluation runner ─────────────────────────────────────────────────────

    def _run_single(
        self,
        query_entry: EvalQuery,
        retrieval_fn: Callable[[str], list[RetrievedChunk]],
    ) -> QueryResult:
        t0 = time.perf_counter()
        chunks = retrieval_fn(query_entry.query)
        latency_ms = (time.perf_counter() - t0) * 1000

        relevant_flags = [_is_relevant(c, query_entry) for c in chunks[:self.k]]
        total_relevant = max(1, len(query_entry.relevant_files) + len(query_entry.relevant_terms) // 2)

        return QueryResult(
            query_id=query_entry.query_id,
            query=query_entry.query,
            intent=query_entry.intent,
            retrieved_ids=[c.chunk_id for c in chunks[:self.k]],
            retrieved_docs=[c.document[:80] for c in chunks[:self.k]],
            latency_ms=round(latency_ms, 2),
            relevant_found=relevant_flags,
            mrr=_mrr_at_k(relevant_flags),
            ndcg=_ndcg_at_k(relevant_flags),
            recall=_recall_at_k(relevant_flags, total_relevant),
            precision=_precision_at_k(relevant_flags),
        )

    def evaluate(
        self,
        mode: str,
        dataset: list[EvalQuery] | None = None,
    ) -> EvalReport:
        """
        Run full evaluation for a given mode.

        mode: "dense_only" | "hybrid" | "hybrid_rerank"
        """
        queries = dataset or EVAL_DATASET
        pipeline_map = {
            "dense_only":    self._dense_only,
            "hybrid":        self._hybrid,
            "hybrid_rerank": self._hybrid_rerank,
        }
        if mode not in pipeline_map:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(pipeline_map)}")

        fn = pipeline_map[mode]
        results: list[QueryResult] = []

        logger.warning("Evaluating mode=%s on %d queries (k=%d)", mode, len(queries), self.k)
        for qe in queries:
            # Skip health queries for RAG eval (they use tool calls, not retrieval)
            if qe.intent == "HEALTH":
                continue
            try:
                result = self._run_single(qe, fn)
                results.append(result)
            except Exception as exc:
                logger.error("Query %s failed: %s", qe.query_id, exc)

        if not results:
            return EvalReport(
                mode=mode, k=self.k,
                mrr=0, ndcg=0, recall=0, precision=0,
                latency_p50=0, latency_p95=0, n_queries=0,
            )

        latencies   = [r.latency_ms for r in results]
        sorted_lats = sorted(latencies)
        p50_idx     = int(len(sorted_lats) * 0.50)
        p95_idx     = int(len(sorted_lats) * 0.95)

        return EvalReport(
            mode=mode,
            k=self.k,
            mrr=round(sum(r.mrr for r in results) / len(results), 4),
            ndcg=round(sum(r.ndcg for r in results) / len(results), 4),
            recall=round(sum(r.recall for r in results) / len(results), 4),
            precision=round(sum(r.precision for r in results) / len(results), 4),
            latency_p50=round(sorted_lats[p50_idx], 1),
            latency_p95=round(sorted_lats[min(p95_idx, len(sorted_lats)-1)], 1),
            n_queries=len(results),
            per_query=results,
        )
