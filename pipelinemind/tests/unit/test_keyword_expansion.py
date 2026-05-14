"""
Tests for expanded keyword patterns covering the observed failure cases.
Every query that produced wrong intent in production must have a test here.
"""
from __future__ import annotations
import pytest
from retrieval.intent_classifier import Intent, _keyword_classify


class TestObservedFailures:
    """These are the exact queries that failed in production — must all pass."""

    def test_delete_fact_table_is_action(self):
        result = _keyword_classify("what will happen if I delete the fact table?")
        assert result is not None, "Keyword fast-path must match this query"
        assert result[0] == Intent.ACTION, f"Expected ACTION, got {result[0]}"

    def test_delete_orders_fact_is_action(self):
        result = _keyword_classify("what will happen if I delete the orders_fact table?")
        assert result is not None
        assert result[0] == Intent.ACTION

    def test_pipeline_health_is_health(self):
        result = _keyword_classify("what the health of the pipeline?")
        assert result is not None, "Keyword fast-path must match 'health of the pipeline'"
        assert result[0] == Intent.HEALTH, f"Expected HEALTH, got {result[0]}"

    def test_whats_the_health_is_health(self):
        result = _keyword_classify("what's the health?")
        assert result is not None, "Keyword fast-path must match \"what's the health?\""
        assert result[0] == Intent.HEALTH, f"Expected HEALTH, got {result[0]}"

    def test_whats_load_method_is_code_qa(self):
        result = _keyword_classify("what's load method?")
        assert result is not None
        assert result[0] == Intent.CODE_QA

    def test_extract_transform_load_structure_is_code_qa(self):
        result = _keyword_classify(
            "give me the in dept structure of the extract->transform and load thing"
        )
        # This one is ambiguous — CODE_QA or GENERAL both acceptable
        # but must NOT be HEALTH or CATALOGUE
        if result is not None:
            assert result[0] in (Intent.CODE_QA, Intent.GENERAL)


class TestActionKeywords:
    def test_what_will_happen_if_drop(self):
        assert _keyword_classify("what will happen if I drop user_id?")[0] == Intent.ACTION

    def test_what_will_happen_if_remove(self):
        assert _keyword_classify("what will happen if I remove the column?")[0] == Intent.ACTION

    def test_what_will_happen_if_rename(self):
        assert _keyword_classify("what will happen if I rename the table?")[0] == Intent.ACTION

    def test_what_happens_if(self):
        assert _keyword_classify("what happens if I delete orders_fact?")[0] == Intent.ACTION

    def test_if_i_drop(self):
        assert _keyword_classify("if I drop user_id from stg_users what breaks?")[0] == Intent.ACTION

    def test_if_i_delete(self):
        assert _keyword_classify("if I delete the fact table what happens?")[0] == Intent.ACTION


class TestHealthKeywords:
    def test_pipeline_health(self):
        assert _keyword_classify("check pipeline health")[0] == Intent.HEALTH

    def test_health_of_pipeline(self):
        assert _keyword_classify("health of the pipeline")[0] == Intent.HEALTH

    def test_whats_the_health_short(self):
        assert _keyword_classify("what's the health")[0] == Intent.HEALTH

    def test_pipeline_failed(self):
        assert _keyword_classify("did the orders pipeline fail?")[0] == Intent.HEALTH

    def test_pipeline_status(self):
        assert _keyword_classify("what's the pipeline status?")[0] == Intent.HEALTH

    def test_slo_breach(self):
        assert _keyword_classify("show me SLO breach events")[0] == Intent.HEALTH

    def test_last_run(self):
        assert _keyword_classify("when was the last run of the orders pipeline?")[0] == Intent.HEALTH

    def test_is_pipeline_running(self):
        assert _keyword_classify("is the pipeline running?")[0] == Intent.HEALTH


class TestCatalogueKeywords:
    def test_lineage_dag(self):
        assert _keyword_classify("lineage dag for vw_revenue_by_tier")[0] == Intent.CATALOGUE

    def test_table_lineage(self):
        assert _keyword_classify("show me table lineage for orders_fact")[0] == Intent.CATALOGUE

    def test_pii_columns(self):
        assert _keyword_classify("what PII columns are in dim_users?")[0] == Intent.CATALOGUE

    def test_upstream(self):
        assert _keyword_classify("what tables are upstream of sessions_agg?")[0] == Intent.CATALOGUE

    def test_downstream(self):
        assert _keyword_classify("what depends on orders_fact downstream?")[0] == Intent.CATALOGUE

    def test_what_columns(self):
        assert _keyword_classify("what columns are in the orders_fact table?")[0] == Intent.CATALOGUE


class TestCodeQAKeywords:
    def test_why_does_pipeline_use(self):
        assert _keyword_classify("why does the orders pipeline use MERGE?")[0] == Intent.CODE_QA

    def test_how_does_function_work(self):
        assert _keyword_classify("how does the extract function work?")[0] == Intent.CODE_QA

    def test_load_method(self):
        assert _keyword_classify("what's the load method?")[0] == Intent.CODE_QA

    def test_extract_method(self):
        assert _keyword_classify("explain the extract method")[0] == Intent.CODE_QA
