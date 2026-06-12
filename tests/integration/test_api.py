import unittest

from fastapi.testclient import TestClient

from backend.main import app


class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    def test_search_retrieval(self):
        payload = {
            "query": "smart speaker Alexa Dot",
            "weights": {"bm25": 0.40, "cosine": 0.20, "ctr": 0.15, "popularity": 0.15},
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], payload["query"])
        self.assertIn("expandedQuery", data)
        self.assertGreater(len(data["results"]), 0)

        # Check model features registry structure
        first_item = data["results"][0]
        self.assertIn("productId", first_item)
        self.assertIn("features", first_item)
        self.assertIn("bm25", first_item["features"])

    def test_recommendation_filtering(self):
        payload = {"userId": "user-1", "productId": "prod-1", "type": "content"}
        response = self.client.post("/api/v1/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "content")
        self.assertGreater(len(data["results"]), 0)

    def test_explain_shap(self):
        payload = {"productId": "prod-3", "query": "noise canceling headphones"}
        response = self.client.post("/api/v1/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["productId"], payload["productId"])
        self.assertIn("baseValue", data)
        self.assertIn("contributions", data)
        self.assertGreater(len(data["contributions"]), 0)

    def test_train_experiments_pipeline(self):
        payload = {
            "algorithm": "listwise_lambdamart",
            "nEstimators": 10,
            "learningRate": 0.05,
        }
        response = self.client.post("/api/v1/rank/train", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["algorithm"], "listwise_lambdamart")
        self.assertIn("runId", data)
        self.assertIn("metrics", data)
        self.assertIn("ndcg5", data["metrics"])
