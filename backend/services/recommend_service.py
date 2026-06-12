import math
from typing import Any, Dict, List, Tuple

from backend.services.search_service import MOCK_PRODUCTS

# Sparse matrix structure
USER_RATING_MATRIX = {
    "user-1": {
        "prod-1": 5,
        "prod-2": 3,
        "prod-3": 5,
        "prod-4": 0,
        "prod-5": 4,
        "prod-6": 0,
        "prod-7": 2,
        "prod-8": 0,
        "prod-9": 5,
        "prod-10": 3,
    },
    "user-2": {
        "prod-1": 1,
        "prod-2": 5,
        "prod-3": 0,
        "prod-4": 5,
        "prod-5": 2,
        "prod-6": 1,
        "prod-7": 4,
        "prod-8": 5,
        "prod-9": 0,
        "prod-10": 4,
    },
    "user-3": {
        "prod-1": 4,
        "prod-2": 0,
        "prod-3": 4,
        "prod-4": 0,
        "prod-5": 5,
        "prod-6": 5,
        "prod-7": 0,
        "prod-8": 2,
        "prod-9": 4,
        "prod-10": 0,
    },
    "user-4": {
        "prod-1": 0,
        "prod-2": 4,
        "prod-3": 2,
        "prod-4": 4,
        "prod-5": 0,
        "prod-6": 3,
        "prod-7": 5,
        "prod-8": 4,
        "prod-9": 2,
        "prod-10": 5,
    },
    "user-5": {
        "prod-1": 5,
        "prod-2": 2,
        "prod-3": 5,
        "prod-4": 1,
        "prod-5": 5,
        "prod-6": 5,
        "prod-7": 3,
        "prod-8": 1,
        "prod-9": 5,
        "prod-10": 2,
    },
}


class RecommendService:
    @staticmethod
    def get_content_based(product_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        source = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
        if not source:
            source = MOCK_PRODUCTS[0]

        results = []
        for p in MOCK_PRODUCTS:
            if p["id"] == source["id"]:
                continue

            score = 0.0
            breakdown = []

            if p["category"] == source["category"]:
                score += 0.50
                breakdown.append("Category match (+0.50)")

            # Compute intersection text overlap
            s_tokens = set(source["title"].lower().split())
            p_tokens = set(p["title"].lower().split())
            intersect = s_tokens.intersection(p_tokens)
            if intersect:
                term_match = min(0.40, len(intersect) * 0.10)
                score += term_match
                breakdown.append(f"Title matching tokens {list(intersect)} (+{term_match:.2f})")

            results.append(
                {
                    "productId": p["id"],
                    "title": p["title"],
                    "category": p["category"],
                    "score": round(score, 3),
                    "type": "content",
                    "breakdown": ". ".join(breakdown) if breakdown else "Default category resemblance baseline",
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def get_collaborative_filtering(current_user: str, limit: int = 4) -> List[Dict[str, Any]]:
        user_ratings = USER_RATING_MATRIX.get(current_user)
        if not user_ratings:
            user_ratings = USER_RATING_MATRIX["user-1"]
            current_user = "user-1"

        # Compute user-to-user cosine similarities
        sim_scores = {}
        for other, r_map in USER_RATING_MATRIX.items():
            if other == current_user:
                continue
            dot = 0.0
            norm_self = 0.0
            norm_other = 0.0
            for p in MOCK_PRODUCTS:
                r1 = user_ratings.get(p["id"], 0)
                r2 = r_map.get(p["id"], 0)
                dot += r1 * r2
                norm_self += r1 * r1
                norm_other += r2 * r2
            sim_scores[other] = (
                dot / (math.sqrt(norm_self) * math.sqrt(norm_other)) if (norm_self > 0 and norm_other > 0) else 0.0
            )

        # Calculate predicted product score rates
        results = []
        for p in MOCK_PRODUCTS:
            if user_ratings.get(p["id"], 0) > 0:
                continue

            w_sum = 0.0
            sim_sum = 0.0
            for other, sim in sim_scores.items():
                rating = USER_RATING_MATRIX[other].get(p["id"], 0)
                if rating > 0:
                    w_sum += sim * rating
                    sim_sum += abs(sim)

            pred = w_sum / sim_sum if sim_sum > 0 else 0.0
            score_normalized = pred / 5.0

            if score_normalized > 0.01:
                results.append(
                    {
                        "productId": p["id"],
                        "title": p["title"],
                        "category": p["category"],
                        "score": round(score_normalized, 3),
                        "type": "collaborative",
                        "breakdown": f"Predicted rank {pred:.2f}/5.0 via user neighborhood similarity.",
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def train_matrix_factorization(
        epochs: int = 30, latent_dim: int = 3, lr: float = 0.05
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, float]]]:
        users_list = list(USER_RATING_MATRIX.keys())
        products_list = [p["id"] for p in MOCK_PRODUCTS]

        # Latent dimensions
        import random

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

                        # Apply gradients updates
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
        cf_list = RecommendService.get_collaborative_filtering(user_id, 10)
        source_id = target_product_id or MOCK_PRODUCTS[0]["id"]
        cb_list = RecommendService.get_content_based(source_id, 10)

        scores_map = {}
        for p in MOCK_PRODUCTS:
            scores_map[p["id"]] = {"product": p, "cf": 0.0, "cb": 0.0}

        for c in cf_list:
            scores_map[c["productId"]]["cf"] = c["score"]
        for c in cb_list:
            scores_map[c["productId"]]["cb"] = c["score"]

        results = []
        for pid, data in scores_map.items():
            if pid == target_product_id:
                continue

            score = hybrid_weight * data["cf"] + (1.0 - hybrid_weight) * data["cb"]
            results.append(
                {
                    "productId": pid,
                    "title": data["product"]["title"],
                    "category": data["product"]["category"],
                    "score": round(score, 3),
                    "type": "hybrid",
                    "breakdown": f"Hybrid matching [{hybrid_weight}*CF ({data['cf']:.2f}) + {1 - hybrid_weight}*CB ({data['cb']:.2f})]",
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
