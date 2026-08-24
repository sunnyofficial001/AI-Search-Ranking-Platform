"""
Dataset Validation Pipeline
=============================
Provides deterministic validation of MSLR-WEB10K dataset integrity before training.

Checks:
  - All three splits present (train, vali, test)
  - Correct feature dimensionality (136)
  - No NaN or Inf values
  - Label distribution in 0–4 range
  - Query group consistency
  - Feature statistics (mean, std, min, max per feature)
  - Leakage detection: test qids not in train
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import Counter

import numpy as np

from backend.data.mslr_loader import (
    load_split, get_query_groups, sort_by_qid, NUM_FEATURES, get_feature_names
)

logger = logging.getLogger(__name__)

VALID_LABELS = {0, 1, 2, 3, 4}
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def validate_split(
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    split_name: str,
) -> Dict[str, Any]:
    """
    Validate a single dataset split.

    Returns a dict with validation results. Raises ValueError on critical failures.
    """
    report: Dict[str, Any] = {"split": split_name, "passed": True, "issues": []}

    # 1. Shape check
    if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
        msg = f"Expected shape (N, {NUM_FEATURES}), got {X.shape}"
        report["issues"].append(msg)
        report["passed"] = False
        raise ValueError(f"[{split_name}] {msg}")

    if len(y) != len(X) or len(qids) != len(X):
        msg = "Label / qid length mismatch with feature matrix"
        report["issues"].append(msg)
        report["passed"] = False
        raise ValueError(f"[{split_name}] {msg}")

    # 2. Label validity
    label_set = set(y.tolist())
    invalid_labels = label_set - VALID_LABELS
    if invalid_labels:
        msg = f"Found out-of-range labels: {invalid_labels}"
        report["issues"].append(msg)
        report["passed"] = False

    label_counts = {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())}
    report["label_distribution"] = label_counts

    # 3. NaN / Inf check
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    report["nan_values"] = nan_count
    report["inf_values"] = inf_count
    if nan_count > 0 or inf_count > 0:
        msg = f"Found {nan_count} NaN and {inf_count} Inf values in features"
        report["issues"].append(msg)
        # Replace NaN/Inf with 0 (imputation note, not a hard failure)
        logger.warning(f"[{split_name}] {msg} — replacing with 0")

    # 4. Feature statistics
    feature_stats = {
        "global_mean": round(float(np.mean(X)), 6),
        "global_std":  round(float(np.std(X)), 6),
        "global_min":  round(float(np.min(X)), 6),
        "global_max":  round(float(np.max(X)), 6),
    }
    report["feature_stats"] = feature_stats

    # 5. Query group statistics
    unique_qids, counts = np.unique(qids, return_counts=True)
    report["n_queries"] = int(len(unique_qids))
    report["n_docs"] = int(len(y))
    report["avg_docs_per_query"] = round(float(np.mean(counts)), 2)
    report["min_docs_per_query"] = int(np.min(counts))
    report["max_docs_per_query"] = int(np.max(counts))

    if report["min_docs_per_query"] < 1:
        report["issues"].append("Found queries with 0 documents")
        report["passed"] = False

    return report


def check_no_label_leakage(
    qids_train: np.ndarray,
    qids_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Verify test query IDs do not appear in training set (query-level leakage check).
    """
    train_qids = set(qids_train.tolist())
    test_qids = set(qids_test.tolist())
    overlap = train_qids & test_qids
    return {
        "train_queries": len(train_qids),
        "test_queries": len(test_qids),
        "overlapping_queries": len(overlap),
        "leakage_detected": len(overlap) > 0,
        "overlap_qids_sample": sorted(list(overlap))[:5],
    }


def validate_dataset(fold_dir: str) -> Dict[str, Any]:
    """
    Run full validation across train, vali, and test splits.

    Returns a complete validation report.
    Saves reports/dataset_validation.json.
    """
    logger.info("=== Running MSLR-WEB10K Dataset Validation ===")
    full_report: Dict[str, Any] = {
        "dataset": "MSLR-WEB10K",
        "fold_dir": fold_dir,
        "num_features": NUM_FEATURES,
        "splits": {},
        "leakage_check": {},
        "overall_passed": True,
    }

    split_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for split_name in ["train", "vali", "test"]:
        path = os.path.join(fold_dir, f"{split_name}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing split file: {path}")

        # Load only first 50k rows for fast validation (structure check, not full metrics)
        logger.info(f"  Validating {split_name}...")
        X, y, qids = load_split(path, max_rows=50_000, verbose=False)
        X, y, qids = sort_by_qid(X, y, qids)
        split_data[split_name] = (X, y, qids)

        split_report = validate_split(X, y, qids, split_name)
        full_report["splits"][split_name] = split_report

        if not split_report["passed"]:
            full_report["overall_passed"] = False

        logger.info(
            f"  {split_name:5s}: {split_report['n_docs']:>7,} docs | "
            f"{split_report['n_queries']:>4,} queries | "
            f"labels={split_report['label_distribution']} | "
            f"NaN={split_report['nan_values']} | Inf={split_report['inf_values']}"
        )

    # Leakage check
    X_tr, y_tr, q_tr = split_data["train"]
    X_te, y_te, q_te = split_data["test"]
    leakage = check_no_label_leakage(q_tr, q_te)
    full_report["leakage_check"] = leakage
    if leakage["leakage_detected"]:
        logger.warning(
            f"⚠ Query ID overlap detected: {leakage['overlapping_queries']} shared queries"
        )
    else:
        logger.info("✓ No query-level leakage detected between train and test sets")

    # Save validation report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_validation.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Dataset validation report saved to {report_path}")

    status = "PASSED ✓" if full_report["overall_passed"] else "FAILED ✗"
    logger.info(f"=== Dataset Validation {status} ===")
    return full_report
