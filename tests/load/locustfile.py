"""
Load Testing — AI Search Platform
====================================
Locust-based load test covering search, recommendations, and ML endpoints.
Simulates realistic user behavior with weighted task distribution.

Run:
    locust -f tests/load/locustfile.py \
           --host=http://localhost:8000 \
           --users=100 \
           --spawn-rate=10 \
           --run-time=5m \
           --headless \
           --html=load-report.html

Scenarios:
  - SearchUser: heavy read workload (70% of users)
  - RecommendUser: recommendation requests (20% of users)
  - MLOpsUser: drift/metrics/registry polling (10% of users)
"""

import random

from locust import HttpUser, between, events, tag, task
from locust.exception import RescheduleTask

# ─────────────────────────────────────────────
# Test Data
# ─────────────────────────────────────────────

SEARCH_QUERIES = [
    "noise cancelling headphones",
    "wireless speaker alexa",
    "gaming laptop rtx",
    "running shoes nike",
    "slim fit jeans",
    "usb-c fast charger",
    "lightweight backpack",
    "fiction books bestseller",
    "bluetooth earbuds",
    "minimalist sneakers",
]

ALGORITHMS = ["ensemble", "listwise_lambdamart", "pairwise_ranknet"]

USER_IDS = [f"user-{i}" for i in range(1, 6)]

ITEM_IDS = [f"prod-{i}" for i in range(1, 11)]

REC_TYPES = ["session_based", "two_tower", "cold_start", "item2vec", "hybrid"]

SESSION_SEQUENCES = [
    ["prod-1", "prod-5"],
    ["prod-3", "prod-6"],
    ["prod-2", "prod-7"],
    ["prod-9", "prod-5"],
]

FEATURE_NAMES = ["bm25_score", "ctr_7d", "freshness_score", "popularity_score"]


# ─────────────────────────────────────────────
# User: Search-heavy (70% weight)
# ─────────────────────────────────────────────


class SearchUser(HttpUser):
    """Simulates a user primarily doing search queries."""

    weight = 70
    wait_time = between(0.5, 2.0)

    @task(40)
    @tag("search", "v1", "critical")
    def legacy_search(self):
        """V1 search endpoint — baseline ranking."""
        query = random.choice(SEARCH_QUERIES)
        with self.client.post(
            "/api/v1/search",
            json={"query": query},
            name="/api/v1/search",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("results"):
                    resp.failure("Empty results returned")
                elif len(data["results"]) == 0:
                    resp.failure("Zero results for query")
                else:
                    resp.success()
            elif resp.status_code == 429:
                resp.failure("Rate limited")
                raise RescheduleTask()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(35)
    @tag("search", "v2", "critical")
    def v2_search_ensemble(self):
        """V2 multi-stage ensemble ranking."""
        query = random.choice(SEARCH_QUERIES)
        user_id = random.choice(USER_IDS + [None])
        payload = {
            "query": query,
            "algorithm": "ensemble",
            "page": 1,
            "page_size": 10,
            "diversity_lambda": round(random.uniform(0.1, 0.5), 2),
        }
        if user_id:
            payload["user_id"] = user_id
            payload["personalization_weight"] = 0.2

        with self.client.post(
            "/api/v1/v2/search",
            json=payload,
            name="/api/v1/v2/search [ensemble]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "pipeline_metadata" not in data:
                    resp.failure("Missing pipeline_metadata")
                else:
                    latency = data.get("latency_ms", 0)
                    if latency > 500:
                        resp.failure(f"Latency {latency}ms exceeds SLA")
                    else:
                        resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(15)
    @tag("search", "v2")
    def v2_search_algorithms(self):
        """Test different ranking algorithms."""
        algo = random.choice(ALGORITHMS)
        query = random.choice(SEARCH_QUERIES)
        with self.client.post(
            "/api/v1/v2/search",
            json={"query": query, "algorithm": algo, "page_size": 5},
            name=f"/api/v1/v2/search [{algo}]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(10)
    @tag("features")
    def compute_features(self):
        """Feature store — compute feature vector."""
        query = random.choice(SEARCH_QUERIES).split()[0]
        doc_id = random.choice(ITEM_IDS)
        with self.client.post(
            f"/api/v1/feature-store/compute?query={query}&doc_id={doc_id}",
            name="/api/v1/feature-store/compute",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ─────────────────────────────────────────────
# User: Recommendation-heavy (20% weight)
# ─────────────────────────────────────────────


class RecommendUser(HttpUser):
    """Simulates a user browsing recommendations."""

    weight = 20
    wait_time = between(1.0, 3.0)

    @task(30)
    @tag("recommend", "v1")
    def legacy_collaborative(self):
        user_id = random.choice(USER_IDS)
        with self.client.post(
            "/api/v1/recommend",
            json={"type": "collaborative", "userId": user_id},
            name="/api/v1/recommend [collaborative]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(25)
    @tag("recommend", "v2", "session")
    def v2_session_recommend(self):
        user_id = random.choice(USER_IDS)
        session = random.choice(SESSION_SEQUENCES)
        with self.client.post(
            "/api/v1/v2/recommend",
            json={
                "rec_type": "session_based",
                "user_id": user_id,
                "session_items": session,
                "top_k": 6,
                "apply_diversity": True,
            },
            name="/api/v1/v2/recommend [session]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "items" not in data:
                    resp.failure("Missing items in response")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(20)
    @tag("recommend", "v2", "two-tower")
    def v2_two_tower_recommend(self):
        user_id = random.choice(USER_IDS)
        with self.client.post(
            "/api/v1/v2/recommend",
            json={"rec_type": "two_tower", "user_id": user_id, "top_k": 6},
            name="/api/v1/v2/recommend [two_tower]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(15)
    @tag("recommend", "v2", "item2vec")
    def v2_item2vec_recommend(self):
        item_id = random.choice(ITEM_IDS)
        with self.client.post(
            "/api/v1/v2/recommend",
            json={"rec_type": "item2vec", "item_id": item_id, "top_k": 5},
            name="/api/v1/v2/recommend [item2vec]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(10)
    @tag("recommend", "cold-start")
    def cold_start_recommend(self):
        with self.client.post(
            "/api/v1/v2/recommend",
            json={"rec_type": "cold_start", "top_k": 6},
            name="/api/v1/v2/recommend [cold_start]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ─────────────────────────────────────────────
# User: MLOps / Monitoring (10% weight)
# ─────────────────────────────────────────────


class MLOpsUser(HttpUser):
    """Simulates ML engineers polling dashboards and monitoring."""

    weight = 10
    wait_time = between(5.0, 15.0)

    @task(25)
    @tag("health")
    def health_check(self):
        with self.client.get(
            "/api/v1/health",
            name="/api/v1/health",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(20)
    @tag("metrics")
    def metrics_poll(self):
        with self.client.get(
            "/api/v1/metrics",
            name="/api/v1/metrics",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(15)
    @tag("drift")
    def drift_report(self):
        with self.client.get(
            "/api/v1/drift/report",
            name="/api/v1/drift/report",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(15)
    @tag("ab-tests")
    def ab_experiments_list(self):
        with self.client.get(
            "/api/v1/ab-tests",
            name="/api/v1/ab-tests",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(10)
    @tag("model-registry")
    def model_registry_list(self):
        with self.client.get(
            "/api/v1/model-registry",
            name="/api/v1/model-registry",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(10)
    @tag("analytics")
    def ranking_analytics(self):
        with self.client.get(
            "/api/v1/analytics/ranking",
            name="/api/v1/analytics/ranking",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(5)
    @tag("drift")
    def drift_record_observation(self):
        feature = random.choice(FEATURE_NAMES)
        value = round(random.uniform(0.0, 3.0), 4)
        with self.client.post(
            f"/api/v1/drift/record?feature_name={feature}&value={value}",
            name="/api/v1/drift/record",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ─────────────────────────────────────────────
# Locust Event Hooks — Custom SLA Reporting
# ─────────────────────────────────────────────

SLA_LIMITS_MS = {
    "/api/v1/search": 200,
    "/api/v1/v2/search [ensemble]": 300,
    "/api/v1/v2/recommend [session]": 150,
    "/api/v1/v2/recommend [two_tower]": 100,
    "/api/v1/health": 50,
    "/api/v1/metrics": 100,
}

sla_violations = []


@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    **kwargs,
):
    """Track SLA violations across all requests."""
    if exception:
        return
    limit = SLA_LIMITS_MS.get(name)
    if limit and response_time > limit:
        sla_violations.append(
            {
                "endpoint": name,
                "response_time_ms": round(response_time, 2),
                "sla_limit_ms": limit,
                "violation_ms": round(response_time - limit, 2),
            }
        )


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print SLA violation summary at end of test."""
    if sla_violations:
        print(f"\n{'=' * 60}")
        print(f"SLA VIOLATIONS: {len(sla_violations)}")
        by_endpoint = {}
        for v in sla_violations:
            ep = v["endpoint"]
            by_endpoint.setdefault(ep, []).append(v["violation_ms"])
        for ep, violations in by_endpoint.items():
            avg = sum(violations) / len(violations)
            print(f"  {ep}: {len(violations)} violations, avg +{avg:.1f}ms over SLA")
        print(f"{'=' * 60}\n")
    else:
        print("\n✓ No SLA violations detected\n")
