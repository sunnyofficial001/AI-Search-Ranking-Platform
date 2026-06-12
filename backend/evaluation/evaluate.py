"""
Evaluation Module
=================
Provides both the legacy static Evaluator API (used by unit tests)
and the new query-grouped evaluation that uses ranking_metrics.py.
"""

import math
from typing import Dict, List


class Evaluator:
    """Static methods for computing IR metrics. Used by unit tests and the API."""

    # ── DCG / NDCG ────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_dcg(relevances: List[float], k: int) -> float:
        """DCG@k using the standard gain formula: sum((2^rel - 1) / log2(i+2))."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (2**rel - 1) / math.log2(i + 2)
        return dcg

    @staticmethod
    def calculate_ndcg(ranked: List[float], ideal: List[float], k: int) -> float:
        """NDCG@k = DCG@k / IDCG@k. Returns 0.0 if IDCG = 0."""
        idcg = Evaluator.calculate_dcg(sorted(ideal, reverse=True), k)
        if idcg == 0:
            return 0.0
        return Evaluator.calculate_dcg(ranked, k) / idcg

    # ── MAP ───────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_map(ranked: List[float], num_relevant: int) -> float:
        """Mean Average Precision given a ranked list and total relevant count."""
        if num_relevant == 0:
            return 0.0
        precision_sum = 0.0
        hits = 0
        for i, rel in enumerate(ranked):
            if rel > 0:
                hits += 1
                precision_sum += hits / (i + 1)
        return precision_sum / num_relevant

    # ── MRR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_mrr(ranked: List[float]) -> float:
        """Mean Reciprocal Rank — reciprocal of rank of first relevant document."""
        for i, rel in enumerate(ranked):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0

    # ── Precision / Recall ────────────────────────────────────────────────────

    @staticmethod
    def calculate_precision_at_k(ranked: List[float], k: int) -> float:
        """Precision@k — fraction of top-k results that are relevant."""
        top_k = ranked[:k]
        return sum(1 for r in top_k if r > 0) / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(ranked: List[float], num_relevant: int, k: int) -> float:
        """Recall@k — fraction of relevant documents found in top-k."""
        if num_relevant == 0:
            return 0.0
        top_k = ranked[:k]
        return sum(1 for r in top_k if r > 0) / num_relevant

    # ── Full Suite ────────────────────────────────────────────────────────────

    @staticmethod
    def run_full_suite(
        ranked_relevances: List[float], ideal_relevances: List[float], qids: list = None
    ) -> Dict[str, float]:
        """Compute NDCG@5, NDCG@10, MAP, MRR, P@5, R@5 for a single query."""
        num_relevant = sum(1 for r in ideal_relevances if r > 0)

        return {
            "ndcg5": round(Evaluator.calculate_ndcg(ranked_relevances, ideal_relevances, 5), 4),
            "ndcg10": round(Evaluator.calculate_ndcg(ranked_relevances, ideal_relevances, 10), 4),
            "map": round(Evaluator.calculate_map(ranked_relevances, num_relevant), 4),
            "mrr": round(Evaluator.calculate_mrr(ranked_relevances), 4),
            "precision5": round(Evaluator.calculate_precision_at_k(ranked_relevances, 5), 4),
            "recall5": round(Evaluator.calculate_recall_at_k(ranked_relevances, num_relevant, 5), 4),
        }
