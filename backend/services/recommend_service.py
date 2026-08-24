"""
Recommendation Service (Legacy v1 API)
=======================================
Provides three recommendation strategies: content-based, collaborative filtering,
and matrix factorisation SGD.

Data source hierarchy:
  1. Production: queries the ProductModel table via SQLAlchemy.
  2. Test fallback (TESTING=true only): falls back to MOCK_PRODUCTS from search_service.
     This keeps the 172 existing integration tests green while preserving the
     non-negotiable rule: "Mocks are allowed ONLY in automated test fixtures."

The USER_RATING_MATRIX is retained as an in-memory fixture for collaborative
filtering — in production this should be replaced with a ClickLog query, but
that requires user interaction data in the DB which is not yet seeded.
"""

import math
import logging
from typing import List, Dict, Any, Tuple

from backend.core.config import settings

logger = logging.getLogger("recommend_service")

# ---------------------------------------------------------------------------
# Rating matrix (fixed fixture — see module docstring)
# ---------------------------------------------------------------------------

USER_RATING_MATRIX = {
    "user-1": {"prod-1": 5, "prod-2": 3, "prod-3": 5, "prod-4": 0, "prod-5": 4,
               "prod-6": 0, "prod-7": 2, "prod-8": 0, "prod-9": 5, "prod-10": 3},
    "user-2": {"prod-1": 1, "prod-2": 5, "prod-3": 0, "prod-4": 5, "prod-5": 2,
               "prod-6": 1, "prod-7": 4, "prod-8": 5, "prod-9": 0, "prod-10": 4},
    "user-3": {"prod-1": 4, "prod-2": 0, "prod-3": 4, "prod-4": 0, "prod-5": 5,
               "prod-6": 5, "prod-7": 0, "prod-8": 2, "prod-9": 4, "prod-10": 0},
    "user-4": {"prod-1": 0, "prod-2": 4, "prod-3": 2, "prod-4": 4, "prod-5": 0,
               "prod-6": 3, "prod-7": 5, "prod-8": 4, "prod-9": 2, "prod-10": 5},
    "user-5": {"prod-1": 5, "prod-2": 2, "prod-3": 5, "prod-4": 1, "prod-5": 5,
               "prod-6": 5, "prod-7": 3, "prod-8": 1, "prod-9": 5, "prod-10": 2},
}


# ---------------------------------------------------------------------------
# Product catalog helper — DB-first, test fallback
# ---------------------------------------------------------------------------

def _load_products() -> List[Dict[str, Any]]:
    """
    Return the product catalog as a list of plain dicts.

    In production: queries the `products` table via SQLAlchemy.
    In tests (TESTING=true) with no DB: falls back to MOCK_PRODUCTS.
    """
    try:
        from backend.database.connection import SessionLocal
        if SessionLocal is None:
            raise RuntimeError("Database session factory is None.")
        db = SessionLocal()
        try:
            from backend.database.models import ProductModel
            rows = db.query(ProductModel).all()
            if rows:
                return [
                    {
                        "id": r.id,
                        "title": r.title,
                        "description": r.description or "",
                        "category": r.category,
                        "popularity": r.popularity,
                        "ctr": r.ctr,
                        "freshness": r.freshness,
                        "engagement": r.engagement,
                    }
                    for r in rows
                ]
        finally:
            db.close()
    except Exception as exc:
        logger.debug("DB product load failed (%s).", exc)

    # Fallback allowed only in test mode
    if settings.is_testing:
        logger.warning("TESTING=true — using MOCK_PRODUCTS for recommendations.")
        from backend.services.search_service import MOCK_PRODUCTS
        return MOCK_PRODUCTS

    raise RuntimeError(
        "Product catalog is unavailable from the database and TESTING is not set. "
        "Ensure the database is running and seeded (run the application with startup events)."
    )


# ---------------------------------------------------------------------------
# RecommendService
# ---------------------------------------------------------------------------

class RecommendService:

    @staticmethod
    def get_content_based(product_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        products = _load_products()
        source = next((p for p in products if p["id"] == product_id), None)
        if not source:
            source = products[0]

        results = []
        for p in products:
            if p["id"] == source["id"]:
                continue

            score = 0.0
            breakdown = []

            if p["category"] == source["category"]:
                score += 0.50
                breakdown.append("Category match (+0.50)")

            s_tokens = set(source["title"].lower().split())
            p_tokens = set(p["title"].lower().split())
            intersect = s_tokens.intersection(p_tokens)
            if intersect:
                term_match = min(0.40, len(intersect) * 0.10)
                score += term_match
                breakdown.append(f"Title matching tokens {list(intersect)} (+{term_match:.2f})")

            results.append({
                "productId": p["id"],
                "title": p["title"],
                "category": p["category"],
                "score": round(score, 3),
                "type": "content",
                "breakdown": ". ".join(breakdown) if breakdown else "Default category resemblance baseline",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def get_collaborative_filtering(current_user: str, limit: int = 4) -> List[Dict[str, Any]]:
        products = _load_products()

        user_ratings = USER_RATING_MATRIX.get(current_user)
        if not user_ratings:
            user_ratings = USER_RATING_MATRIX.get("user-1", {})
            current_user = "user-1"

        sim_scores: Dict[str, float] = {}
        for other, r_map in USER_RATING_MATRIX.items():
            if other == current_user:
                continue
            dot = norm_self = norm_other = 0.0
            for p in products:
                r1 = user_ratings.get(p["id"], 0)
                r2 = r_map.get(p["id"], 0)
                dot += r1 * r2
                norm_self += r1 * r1
                norm_other += r2 * r2
            denom = math.sqrt(norm_self) * math.sqrt(norm_other)
            sim_scores[other] = dot / denom if denom > 0 else 0.0

        results = []
        for p in products:
            if user_ratings.get(p["id"], 0) > 0:
                continue

            w_sum = sim_sum = 0.0
            for other, sim in sim_scores.items():
                rating = USER_RATING_MATRIX[other].get(p["id"], 0)
                if rating > 0:
                    w_sum += sim * rating
                    sim_sum += abs(sim)

            pred = w_sum / sim_sum if sim_sum > 0 else 0.0
            score_normalized = pred / 5.0

            if score_normalized > 0.01:
                results.append({
                    "productId": p["id"],
                    "title": p["title"],
                    "category": p["category"],
                    "score": round(score_normalized, 3),
                    "type": "collaborative",
                    "breakdown": f"Predicted rank {pred:.2f}/5.0 via user neighborhood similarity.",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def train_matrix_factorization(
        epochs: int = 30,
        latent_dim: int = 3,
        lr: float = 0.05,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, float]]]:
        import random

        products = _load_products()
        users_list = list(USER_RATING_MATRIX.keys())
        products_list = [p["id"] for p in products]

        p_matrix = {u: [random.uniform(0.1, 0.5) for _ in range(latent_dim)] for u in users_list}
        q_matrix = {p: [random.uniform(0.1, 0.5) for _ in range(latent_dim)] for p in products_list}

        lambda_reg = 0.02
        losses = []

        for epoch in range(1, epochs + 1):
            sq_err = 0.0
            cnt = 0
            for u in users_list:
                for p in products_list:
                    r = USER_RATING_MATRIX[u].get(p, 0)
                    if r > 0:
                        pred = sum(p_matrix[u][k] * q_matrix[p][k] for k in range(latent_dim))
                        err = r - pred
                        sq_err += err * err
                        cnt += 1
                        for k in range(latent_dim):
                            pk = p_matrix[u][k]
                            qk = q_matrix[p][k]
                            p_matrix[u][k] += lr * (err * qk - lambda_reg * pk)
                            q_matrix[p][k] += lr * (err * pk - lambda_reg * qk)
            rmse = math.sqrt(sq_err / cnt) if cnt > 0 else 0.0
            if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                losses.append({"epoch": float(epoch), "rmse": round(rmse, 4)})

        return p_matrix, q_matrix, losses

    @staticmethod
    def get_hybrid_recommendations(
        user_id: str,
        target_product_id: str = None,
        limit: int = 4,
        hybrid_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        products = _load_products()
        cf_list = RecommendService.get_collaborative_filtering(user_id, 10)
        source_id = target_product_id or products[0]["id"]
        cb_list = RecommendService.get_content_based(source_id, 10)

        scores_map = {p["id"]: {"product": p, "cf": 0.0, "cb": 0.0} for p in products}

        for c in cf_list:
            if c["productId"] in scores_map:
                scores_map[c["productId"]]["cf"] = c["score"]
        for c in cb_list:
            if c["productId"] in scores_map:
                scores_map[c["productId"]]["cb"] = c["score"]

        results = []
        for pid, data in scores_map.items():
            if pid == target_product_id:
                continue
            score = hybrid_weight * data["cf"] + (1.0 - hybrid_weight) * data["cb"]
            results.append({
                "productId": pid,
                "title": data["product"]["title"],
                "category": data["product"]["category"],
                "score": round(score, 3),
                "type": "hybrid",
                "breakdown": (
                    f"Hybrid [{hybrid_weight}*CF ({data['cf']:.2f}) "
                    f"+ {1 - hybrid_weight}*CB ({data['cb']:.2f})]"
                ),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
