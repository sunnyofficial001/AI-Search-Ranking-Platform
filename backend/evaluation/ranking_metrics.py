"""
Production-Grade Ranking Metrics
=================================
Query-grouped evaluation metrics for Learning-to-Rank.
All metrics are averaged over queries, matching the MSLR-WEB10K evaluation protocol.

Implemented:
- NDCG@k  (k=1,3,5,10)
- MAP      (Mean Average Precision)
- MRR      (Mean Reciprocal Rank)
- Precision@k  (k=10)
- Recall@k     (k=10)
"""

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k for a single ranked list."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2.0**rel - 1.0) / math.log2(i + 2.0)
    return dcg


def _ndcg_at_k(ranked_rels: List[int], k: int) -> float:
    """Compute NDCG@k for a single query."""
    ideal = sorted(ranked_rels, reverse=True)
    dcg = _dcg_at_k(ranked_rels, k)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0.0 else 0.0


def _average_precision(ranked_rels: List[int], threshold: int = 1) -> float:
    """
    Compute Average Precision for a single query.
    Documents with relevance >= threshold are considered relevant.
    """
    relevant_count = 0
    precision_sum = 0.0
    total_relevant = sum(1 for r in ranked_rels if r >= threshold)
    if total_relevant == 0:
        return 0.0
    for i, rel in enumerate(ranked_rels):
        if rel >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / total_relevant


def _reciprocal_rank(ranked_rels: List[int], threshold: int = 1) -> float:
    """Compute Reciprocal Rank for a single query."""
    for i, rel in enumerate(ranked_rels):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def _precision_at_k(ranked_rels: List[int], k: int, threshold: int = 1) -> float:
    """Compute Precision@k for a single query."""
    cutoff = ranked_rels[:k]
    if not cutoff:
        return 0.0
    return sum(1 for r in cutoff if r >= threshold) / len(cutoff)


def _recall_at_k(ranked_rels: List[int], k: int, threshold: int = 1) -> float:
    """Compute Recall@k for a single query."""
    total_relevant = sum(1 for r in ranked_rels if r >= threshold)
    if total_relevant == 0:
        return 0.0
    cutoff = ranked_rels[:k]
    return sum(1 for r in cutoff if r >= threshold) / total_relevant


def evaluate_per_query(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    qids: np.ndarray,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
    relevance_threshold: int = 1,
) -> Dict[str, float]:
    """
    Evaluate ranking metrics across all queries.

    Args:
        y_true: Ground truth relevance labels (shape N,).
        y_pred: Model-predicted scores (shape N,). Higher = more relevant.
        qids: Query IDs corresponding to each row (shape N,).
        k_values: Cutoff depths for NDCG, Precision, Recall.
        relevance_threshold: Min relevance label to be considered relevant.

    Returns:
        dict of metric_name → mean_value across all queries.
    """
    unique_qids = np.unique(qids)
    n_queries = len(unique_qids)

    # Per-query accumulator
    ndcg_sums = {k: 0.0 for k in k_values}
    ap_sum = 0.0
    rr_sum = 0.0
    prec_sums = {k: 0.0 for k in k_values}
    rec_sums = {k: 0.0 for k in k_values}

    for qid in unique_qids:
        mask = qids == qid
        q_true = y_true[mask].tolist()
        q_pred = y_pred[mask].tolist()

        # Rank documents by predicted score (descending)
        ranked_indices = sorted(range(len(q_pred)), key=lambda i: q_pred[i], reverse=True)
        ranked_rels = [q_true[i] for i in ranked_indices]

        for k in k_values:
            ndcg_sums[k] += _ndcg_at_k(ranked_rels, k)
            prec_sums[k] += _precision_at_k(ranked_rels, k, relevance_threshold)
            rec_sums[k] += _recall_at_k(ranked_rels, k, relevance_threshold)

        ap_sum += _average_precision(ranked_rels, relevance_threshold)
        rr_sum += _reciprocal_rank(ranked_rels, relevance_threshold)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"NDCG@{k}"] = round(ndcg_sums[k] / n_queries, 5)
        metrics[f"P@{k}"] = round(prec_sums[k] / n_queries, 5)
        metrics[f"R@{k}"] = round(rec_sums[k] / n_queries, 5)

    metrics["MAP"] = round(ap_sum / n_queries, 5)
    metrics["MRR"] = round(rr_sum / n_queries, 5)
    metrics["n_queries"] = int(n_queries)

    return metrics


def compute_all_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    qids: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute and optionally print the full evaluation suite for one model.
    """
    metrics = evaluate_per_query(y_true, y_pred, qids, k_values=(1, 3, 5, 10))

    if verbose:
        logger.info(f"\n{'=' * 55}")
        logger.info(f"  {model_name} — Test Set Evaluation ({metrics['n_queries']:,} queries)")
        logger.info(f"{'=' * 55}")
        logger.info(f"  NDCG@1  = {metrics['NDCG@1']:.5f}")
        logger.info(f"  NDCG@3  = {metrics['NDCG@3']:.5f}")
        logger.info(f"  NDCG@5  = {metrics['NDCG@5']:.5f}")
        logger.info(f"  NDCG@10 = {metrics['NDCG@10']:.5f}")
        logger.info(f"  MAP     = {metrics['MAP']:.5f}")
        logger.info(f"  MRR     = {metrics['MRR']:.5f}")
        logger.info(f"  P@10    = {metrics['P@10']:.5f}")
        logger.info(f"  R@10    = {metrics['R@10']:.5f}")
        logger.info(f"{'=' * 55}\n")

    return metrics


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format a comparison table of metrics across models.

    Args:
        results: dict of model_name → metrics_dict.
    Returns:
        Markdown table string.
    """
    headers = [
        "Model",
        "NDCG@1",
        "NDCG@3",
        "NDCG@5",
        "NDCG@10",
        "MAP",
        "MRR",
        "P@10",
        "R@10",
    ]
    col_w = [30, 8, 8, 8, 9, 8, 8, 8, 8]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
    rows = [header_row, sep]

    for model_name, m in results.items():
        vals = [
            model_name.ljust(col_w[0]),
            fmt(m.get("NDCG@1", 0)).ljust(col_w[1]),
            fmt(m.get("NDCG@3", 0)).ljust(col_w[2]),
            fmt(m.get("NDCG@5", 0)).ljust(col_w[3]),
            fmt(m.get("NDCG@10", 0)).ljust(col_w[4]),
            fmt(m.get("MAP", 0)).ljust(col_w[5]),
            fmt(m.get("MRR", 0)).ljust(col_w[6]),
            fmt(m.get("P@10", 0)).ljust(col_w[7]),
            fmt(m.get("R@10", 0)).ljust(col_w[8]),
        ]
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join(rows)
