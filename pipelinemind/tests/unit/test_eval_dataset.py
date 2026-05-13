"""Unit tests for the evaluation dataset and metric functions."""
from __future__ import annotations

import math
import pytest
from tests.eval.eval_dataset import EVAL_DATASET, EvalQuery
from tests.eval.rag_evaluator import (
    _mrr_at_k, _ndcg_at_k, _recall_at_k, _precision_at_k
)


class TestEvalDataset:
    def test_dataset_non_empty(self):
        assert len(EVAL_DATASET) >= 10

    def test_all_entries_have_required_fields(self):
        for q in EVAL_DATASET:
            assert q.query_id,  f"Missing query_id: {q}"
            assert q.query,     f"Missing query: {q.query_id}"
            assert q.intent in ("CODE_QA", "CATALOGUE", "HEALTH", "ACTION", "GENERAL")
            assert isinstance(q.relevant_files, list)
            assert isinstance(q.relevant_terms, list)

    def test_intent_distribution_covers_all_types(self):
        intents = {q.intent for q in EVAL_DATASET}
        assert "CODE_QA"   in intents
        assert "CATALOGUE" in intents

    def test_catalogue_queries_have_expected_tables(self):
        cat_queries = [q for q in EVAL_DATASET if q.intent == "CATALOGUE"]
        assert any(q.expected_tables for q in cat_queries)


class TestMetricFunctions:
    def test_mrr_first_result_relevant(self):
        assert _mrr_at_k([True, False, False]) == 1.0

    def test_mrr_second_result_relevant(self):
        assert abs(_mrr_at_k([False, True, False]) - 0.5) < 0.001

    def test_mrr_no_relevant(self):
        assert _mrr_at_k([False, False, False]) == 0.0

    def test_ndcg_perfect_ranking(self):
        flags = [True, True, True]
        score = _ndcg_at_k(flags)
        assert abs(score - 1.0) < 0.001

    def test_ndcg_no_relevant(self):
        assert _ndcg_at_k([False, False, False]) == 0.0

    def test_recall_at_k_all_found(self):
        # 2 relevant docs, both in top-3
        score = _recall_at_k([True, False, True], total_relevant=2)
        assert abs(score - 1.0) < 0.001

    def test_recall_at_k_partial(self):
        score = _recall_at_k([True, False, False], total_relevant=2)
        assert abs(score - 0.5) < 0.001

    def test_precision_at_k(self):
        score = _precision_at_k([True, False, True, False])
        assert abs(score - 0.5) < 0.001

    def test_precision_empty(self):
        assert _precision_at_k([]) == 0.0
