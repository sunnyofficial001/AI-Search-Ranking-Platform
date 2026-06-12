"""
Integration Test Suite — API Endpoints
========================================
Tests the full HTTP request/response cycle for all major API endpoints.
Uses FastAPI's TestClient (no real server needed).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


# Skip all integration tests if FastAPI app can't be imported
pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI app not importable in this environment")


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_returns_200(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_contains_status(self):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_contains_version(self):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data


class TestSearchEndpoint:
    """Test legacy and v2 search endpoints."""

    def test_search_returns_results(self):
        resp = client.post("/api/v1/search", json={"query": "headphones"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_result_has_required_fields(self):
        resp = client.post("/api/v1/search", json={"query": "laptop"})
        assert resp.status_code == 200
        for item in resp.json()["results"]:
            for field in ["productId", "title", "scores", "finalRank"]:
                assert field in item, f"Missing field: {field}"

    def test_search_scores_are_numeric(self):
        resp = client.post("/api/v1/search", json={"query": "charger"})
        assert resp.status_code == 200
        for item in resp.json()["results"]:
            for score_key, score_val in item["scores"].items():
                assert isinstance(score_val, (int, float)), f"Non-numeric score: {score_key}={score_val}"

    def test_search_final_rank_is_sequential(self):
        resp = client.post("/api/v1/search", json={"query": "wireless"})
        assert resp.status_code == 200
        ranks = sorted(item["finalRank"] for item in resp.json()["results"])
        assert ranks == list(range(1, len(ranks) + 1))

    def test_v2_search_returns_pipeline_metadata(self):
        resp = client.post(
            "/api/v1/v2/search",
            json={
                "query": "headphones",
                "algorithm": "ensemble",
                "page": 1,
                "page_size": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "pipeline_metadata" in data
        assert "evaluation_metrics" in data

    def test_v2_search_all_algorithms(self):
        for algorithm in ["ensemble", "listwise_lambdamart", "pairwise_ranknet"]:
            resp = client.post("/api/v1/v2/search", json={"query": "speaker", "algorithm": algorithm})
            assert resp.status_code == 200, f"Algorithm {algorithm} failed"

    def test_v2_search_with_personalization(self):
        resp = client.post(
            "/api/v1/v2/search",
            json={
                "query": "electronics",
                "algorithm": "ensemble",
                "user_id": "user-1",
                "personalization_weight": 0.3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user-1"


class TestRecommendEndpoint:
    """Test recommendation endpoints."""

    def test_collaborative_filtering_returns_items(self):
        resp = client.post(
            "/api/v1/recommend",
            json={
                "type": "collaborative",
                "userId": "user-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0

    def test_content_based_returns_items(self):
        resp = client.post(
            "/api/v1/recommend",
            json={
                "type": "content",
                "userId": "user-1",
                "productId": "prod-3",
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0

    def test_hybrid_returns_items(self):
        resp = client.post(
            "/api/v1/recommend",
            json={
                "type": "hybrid",
                "userId": "user-2",
                "productId": "prod-1",
                "hybridWeight": 0.6,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) > 0

    def test_v2_session_based_recommend(self):
        resp = client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "session_based",
                "user_id": "user-1",
                "session_items": ["prod-1", "prod-5"],
                "top_k": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "metadata" in data

    def test_v2_two_tower_recommend(self):
        resp = client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "two_tower",
                "user_id": "user-3",
                "top_k": 6,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) > 0

    def test_v2_cold_start_recommend(self):
        resp = client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "cold_start",
                "top_k": 5,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["items"]) > 0

    def test_v2_item2vec_recommend(self):
        resp = client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "item2vec",
                "item_id": "prod-3",
                "top_k": 4,
            },
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        # Should not include the seed item
        assert "prod-3" not in [i["item_id"] for i in items]

    def test_v2_recommend_has_diversity_metadata(self):
        resp = client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "hybrid",
                "user_id": "user-1",
                "apply_diversity": True,
            },
        )
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert "intra_list_diversity" in meta


class TestFeatureStoreEndpoints:
    """Test feature store API."""

    def test_feature_catalog_non_empty(self):
        resp = client.get("/api/v1/feature-store/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_features"] > 0
        assert len(data["catalog"]) > 0

    def test_feature_catalog_has_required_fields(self):
        resp = client.get("/api/v1/feature-store/catalog")
        for feature in resp.json()["catalog"]:
            for field in ["name", "type", "description", "source"]:
                assert field in feature

    def test_compute_features_returns_vector(self):
        resp = client.post("/api/v1/feature-store/compute?query=headphones&doc_id=prod-3")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert data["feature_count"] > 0
        assert "validation" in data

    def test_compute_features_invalid_doc(self):
        resp = client.post("/api/v1/feature-store/compute?query=test&doc_id=nonexistent-999")
        assert resp.status_code == 404


class TestDriftEndpoints:
    """Test drift detection API."""

    def test_drift_report_returns_features(self):
        resp = client.get("/api/v1/drift/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert "overall_status" in data
        assert data["overall_status"] in ("stable", "warning", "critical")

    def test_drift_alerts_returns_list(self):
        resp = client.get("/api/v1/drift/alerts")
        assert resp.status_code == 200
        assert "alerts" in resp.json()

    def test_drift_record_observation(self):
        resp = client.post("/api/v1/drift/record?feature_name=bm25_score&value=2.3")
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True


class TestABTestingEndpoints:
    """Test A/B testing API."""

    def test_list_experiments_non_empty(self):
        resp = client.get("/api/v1/ab-tests")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0

    def test_analyze_experiment(self):
        resp = client.get("/api/v1/ab-tests/exp-ranking-001/analysis")
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis" in data
        assert "recommendation" in data

    def test_get_variant_assignment_deterministic(self):
        resp1 = client.get("/api/v1/ab-tests/exp-ranking-001/assign?user_id=stable-user-42")
        resp2 = client.get("/api/v1/ab-tests/exp-ranking-001/assign?user_id=stable-user-42")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["variant_id"] == resp2.json()["variant_id"]

    def test_create_experiment(self):
        resp = client.post(
            "/api/v1/ab-tests",
            json={
                "name": "Integration Test Experiment",
                "description": "Created by integration test",
                "primary_metric": "ctr",
                "guardrail_metrics": ["latency_p99"],
                "min_sample_size": 500,
                "variants": [
                    {
                        "id": "ctrl",
                        "name": "Control",
                        "type": "control",
                        "traffic_fraction": 0.5,
                    },
                    {
                        "id": "trt",
                        "name": "Treatment",
                        "type": "treatment",
                        "traffic_fraction": 0.5,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "experiment_id" in data
        assert data["status"] == "running"


class TestModelRegistryEndpoints:
    """Test model registry API."""

    def test_list_models_returns_models(self):
        resp = client.get("/api/v1/model-registry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0

    def test_filter_by_stage(self):
        resp = client.get("/api/v1/model-registry?stage=production")
        assert resp.status_code == 200
        for model in resp.json()["models"]:
            assert model["stage"] == "production"

    def test_get_specific_model(self):
        resp = client.get("/api/v1/model-registry/model-lambdamart-v1.3.0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "model-lambdamart-v1.3.0"

    def test_get_nonexistent_model_returns_404(self):
        resp = client.get("/api/v1/model-registry/model-does-not-exist-9999")
        assert resp.status_code == 404

    def test_compare_models(self):
        resp = client.get("/api/v1/model-registry/compare/model-lambdamart-v1.3.0/model-ranknet-v0.9.1")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_winner" in data
        assert "metrics_comparison" in data

    def test_trigger_retraining(self):
        resp = client.post(
            "/api/v1/model-registry/retrain",
            json={
                "model_name": "LambdaMART-LTR",
                "trigger_type": "integration_test",
                "reason": "Testing retraining trigger from integration tests",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "trigger_id" in data
        assert data["model_name"] == "LambdaMART-LTR"


class TestMetricsEndpoints:
    """Test observability endpoints."""

    def test_metrics_returns_data(self):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "sla_report" in data
        assert "feature_cache" in data

    def test_prometheus_format_is_text(self):
        resp = client.get("/api/v1/metrics/prometheus")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        # Should contain Prometheus lines
        assert "# HELP" in resp.text or "# TYPE" in resp.text or resp.text == ""


class TestDataPipelineEndpoints:
    """Test data pipeline trigger endpoints."""

    def test_pipeline_dry_run(self):
        resp = client.post(
            "/api/v1/pipeline/run",
            json={
                "pipeline_name": "feature_computation",
                "batch_size": 50,
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert "estimated_rows" in data

    def test_pipeline_actual_run(self):
        resp = client.post(
            "/api/v1/pipeline/run",
            json={
                "pipeline_name": "feature_computation",
                "batch_size": 10,
                "dry_run": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["rows_processed"] > 0

    def test_pipeline_lineage(self):
        resp = client.get("/api/v1/pipeline/lineage")
        assert resp.status_code == 200
        assert "lineage" in resp.json()


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    def test_search_analytics(self):
        resp = client.get("/api/v1/analytics/search")
        assert resp.status_code == 200
        data = resp.json()
        assert "search" in data
        assert "recommendations" in data
        assert "sla" in data

    def test_ranking_analytics(self):
        resp = client.get("/api/v1/analytics/ranking")
        assert resp.status_code == 200
        data = resp.json()
        assert "algorithm_comparison" in data
        assert len(data["algorithm_comparison"]) >= 3

    def test_ranking_analytics_has_ndcg(self):
        resp = client.get("/api/v1/analytics/ranking")
        for algo in resp.json()["algorithm_comparison"]:
            assert "ndcg_at_10" in algo
            assert 0.0 <= algo["ndcg_at_10"] <= 1.0
