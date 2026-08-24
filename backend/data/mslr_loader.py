"""
MSLR-WEB10K Data Loader
========================
Parses the real LETOR/SVMLight-format MSLR-WEB10K Fold1 dataset.

File format per line:
    <relevance_label> qid:<qid> <feat_id>:<feat_val> ... # <comment>

Features: 136 per document (indices 1–136)
Labels: 0-4 graded relevance
"""

import os
import logging
from typing import Generator, Tuple, List, Dict, Any
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger(__name__)

# MSLR-WEB10K Fold1 paths (relative to project root)
FOLD1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Fold1")

NUM_FEATURES = 136

# Human-readable feature names for MSLR-WEB10K (136 features)
# Reference: https://www.microsoft.com/en-us/research/project/mslr/
FEATURE_NAMES: Dict[int, str] = {
    1:  "TF_body",         2:  "TF_anchor",       3:  "TF_title",
    4:  "TF_url",          5:  "TF_whole_doc",
    6:  "IDF_body",        7:  "IDF_anchor",       8:  "IDF_title",
    9:  "IDF_url",         10: "IDF_whole_doc",
    11: "TF_IDF_body",     12: "TF_IDF_anchor",    13: "TF_IDF_title",
    14: "TF_IDF_url",      15: "TF_IDF_whole_doc",
    16: "doc_len_body",    17: "doc_len_anchor",    18: "doc_len_title",
    19: "doc_len_url",     20: "doc_len_whole_doc",
    21: "TF_normalized_body",      22: "TF_normalized_anchor",
    23: "TF_normalized_title",     24: "TF_normalized_url",
    25: "TF_normalized_whole_doc",
    26: "sum_TF_body",       27: "sum_TF_anchor",     28: "sum_TF_title",
    29: "sum_TF_url",        30: "sum_TF_whole_doc",
    31: "min_TF_body",       32: "min_TF_anchor",     33: "min_TF_title",
    34: "min_TF_url",        35: "min_TF_whole_doc",
    36: "max_TF_body",       37: "max_TF_anchor",     38: "max_TF_title",
    39: "max_TF_url",        40: "max_TF_whole_doc",
    41: "mean_TF_body",      42: "mean_TF_anchor",    43: "mean_TF_title",
    44: "mean_TF_url",       45: "mean_TF_whole_doc",
    46: "covered_query_term_ratio_body",    47: "covered_query_term_ratio_anchor",
    48: "covered_query_term_ratio_title",   49: "covered_query_term_ratio_url",
    50: "covered_query_term_ratio_whole_doc",
    51: "covered_query_term_num_body",      52: "covered_query_term_num_anchor",
    53: "covered_query_term_num_title",     54: "covered_query_term_num_url",
    55: "covered_query_term_num_whole_doc",
    56: "stream_length_body",    57: "stream_length_anchor",
    58: "stream_length_title",   59: "stream_length_url",
    60: "stream_length_whole_doc",
    61: "idf_sum_body",          62: "idf_sum_anchor",     63: "idf_sum_title",
    64: "idf_sum_url",           65: "idf_sum_whole_doc",
    66: "idf_min_body",          67: "idf_min_anchor",     68: "idf_min_title",
    69: "idf_min_url",           70: "idf_min_whole_doc",
    71: "idf_max_body",          72: "idf_max_anchor",     73: "idf_max_title",
    74: "idf_max_url",           75: "idf_max_whole_doc",
    76: "idf_mean_body",         77: "idf_mean_anchor",    78: "idf_mean_title",
    79: "idf_mean_url",          80: "idf_mean_whole_doc",
    81: "idf_variance_body",     82: "idf_variance_anchor",83: "idf_variance_title",
    84: "idf_variance_url",      85: "idf_variance_whole_doc",
    86: "tf_idf_sum_body",       87: "tf_idf_sum_anchor",  88: "tf_idf_sum_title",
    89: "tf_idf_sum_url",        90: "tf_idf_sum_whole_doc",
    91: "tf_idf_min_body",       92: "tf_idf_min_anchor",  93: "tf_idf_min_title",
    94: "tf_idf_min_url",        95: "tf_idf_min_whole_doc",
    96: "tf_idf_max_body",       97: "tf_idf_max_anchor",  98: "tf_idf_max_title",
    99: "tf_idf_max_url",        100:"tf_idf_max_whole_doc",
    101:"tf_idf_mean_body",      102:"tf_idf_mean_anchor", 103:"tf_idf_mean_title",
    104:"tf_idf_mean_url",       105:"tf_idf_mean_whole_doc",
    106:"bool_query_body",       107:"bool_query_anchor",  108:"bool_query_title",
    109:"bool_query_url",        110:"bool_query_whole_doc",
    111:"LM_body",               112:"LM_anchor",          113:"LM_title",
    114:"LM_url",                115:"LM_whole_doc",
    116:"LM_linear_body",        117:"LM_linear_anchor",   118:"LM_linear_title",
    119:"LM_linear_url",         120:"LM_linear_whole_doc",
    121:"LM_dir_body",           122:"LM_dir_anchor",      123:"LM_dir_title",
    124:"LM_dir_url",            125:"LM_dir_whole_doc",
    126:"BM25_body",             127:"BM25_anchor",        128:"BM25_title",
    129:"BM25_url",              130:"BM25_whole_doc",
    131:"LMIR_ABS_body",         132:"LMIR_ABS_anchor",    133:"LMIR_ABS_title",
    134:"LMIR_ABS_url",          135:"LMIR_ABS_whole_doc",
    136:"SiteMap_quality",
}


def get_feature_names() -> List[str]:
    """Return ordered list of 136 feature names (0-indexed for numpy)."""
    return [FEATURE_NAMES.get(i + 1, f"f{i+1}") for i in range(NUM_FEATURES)]


def _parse_line(line: str) -> Tuple[int, int, List[float]]:
    """
    Parse a single SVMLight/LETOR format line.
    Returns (relevance_label, qid, feature_vector[136]).
    """
    # Strip comment
    line = line.split(" #")[0].strip()
    if not line:
        raise ValueError("Empty line")

    parts = line.split()
    label = int(parts[0])
    qid_str = parts[1]  # "qid:12345"
    qid = int(qid_str.split(":")[1])

    fvec = [0.0] * NUM_FEATURES
    for token in parts[2:]:
        if ":" not in token:
            continue
        idx_str, val_str = token.split(":", 1)
        idx = int(idx_str) - 1   # 0-indexed
        if 0 <= idx < NUM_FEATURES:
            fvec[idx] = float(val_str)

    return label, qid, fvec


def stream_lines(filepath: str) -> Generator[Tuple[int, int, List[float]], None, None]:
    """
    Memory-efficient generator over SVMLight lines.
    Yields (label, qid, feature_vector) per document.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield _parse_line(line)
            except Exception as e:
                logger.warning(f"Skipping malformed line: {e}")
                continue


def load_split(
    filepath: str,
    max_rows: int = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single MSLR Fold1 split (train / vali / test).

    Args:
        filepath: Path to .txt file (LETOR SVMLight format).
        max_rows: If set, load only first N rows (for HPO speed).
        verbose: Print progress.

    Returns:
        X: np.ndarray of shape (N, 136), float32
        y: np.ndarray of shape (N,), int32 — relevance labels 0-4
        qids: np.ndarray of shape (N,), int32 — query IDs
    """
    assert os.path.isfile(filepath), f"Dataset file not found: {filepath}"

    labels_list: List[int] = []
    qids_list: List[int] = []
    features_list: List[List[float]] = []

    for i, (label, qid, fvec) in enumerate(stream_lines(filepath)):
        labels_list.append(label)
        qids_list.append(qid)
        features_list.append(fvec)

        if verbose and (i + 1) % 100_000 == 0:
            logger.info(f"  Loaded {i+1:,} rows from {os.path.basename(filepath)}...")

        if max_rows is not None and (i + 1) >= max_rows:
            break

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    qids = np.array(qids_list, dtype=np.int32)

    assert X.shape[1] == NUM_FEATURES, (
        f"Expected {NUM_FEATURES} features, got {X.shape[1]}"
    )

    if verbose:
        n_queries = len(np.unique(qids))
        label_dist = Counter(labels_list)
        logger.info(
            f"  Split loaded: {len(y):,} docs | {n_queries:,} queries | "
            f"label distribution: {dict(sorted(label_dist.items()))}"
        )

    return X, y, qids


def get_query_groups(qids: np.ndarray) -> np.ndarray:
    """
    Convert array of qids into group sizes array for LightGBM.
    IMPORTANT: qids must be sorted (contiguous groups).
    Returns array of group sizes (one per unique query).
    """
    _, counts = np.unique(qids, return_counts=True)
    return counts.astype(np.int32)


def sort_by_qid(X: np.ndarray, y: np.ndarray, qids: np.ndarray):
    """Sort X, y, qids by qid (required by LightGBM lambdarank)."""
    sort_idx = np.argsort(qids, kind="stable")
    return X[sort_idx], y[sort_idx], qids[sort_idx]


def load_all_splits(
    fold_dir: str = None,
    hpo_subset_rows: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load train, vali, and test splits from MSLR-WEB10K Fold1.

    Args:
        fold_dir: Path to Fold1 directory. Defaults to data/Fold1.
        hpo_subset_rows: If set, only load this many rows for HPO phase.
        verbose: Print progress information.

    Returns:
        dict with keys: train, vali, test — each a tuple (X, y, qids).
        Also includes 'stats' with dataset summary.
    """
    if fold_dir is None:
        root_full_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "MSLR-WEB10K", "Fold1")
        )
        if os.path.exists(root_full_dir) and os.path.isfile(os.path.join(root_full_dir, "train.txt")):
            fold_dir = root_full_dir
        else:
            # Allow demo/synthetic fallback ONLY in automated test environments.
            # In all other contexts (development, production, training runs) the
            # real dataset is required — fail explicitly per the platform rule:
            # "If a dependency/service/model is unavailable, FAIL EXPLICITLY."
            import os as _os
            _testing = _os.getenv("TESTING", "").lower() in ("1", "true", "yes")
            if _testing:
                logger.warning(
                    f"Real MSLR-WEB10K dataset not found at {root_full_dir}. "
                    "Falling back to synthetic demo data (TESTING=true — test-only path)."
                )
                fold_dir = os.path.normpath(FOLD1_DIR)
            else:
                raise RuntimeError(
                    f"MSLR-WEB10K Fold1 dataset not found at: {root_full_dir}\n"
                    "Production and development execution require the real dataset.\n"
                    "Download it from: https://www.microsoft.com/en-us/research/project/mslr/\n"
                    "and place train.txt / vali.txt / test.txt in the above directory.\n"
                    "To suppress this error in tests, set TESTING=true."
                )

    splits = {}
    for split_name in ["train", "vali", "test"]:
        path = os.path.join(fold_dir, f"{split_name}.txt")
        assert os.path.isfile(path), f"Missing dataset file: {path}"

        max_rows = hpo_subset_rows if (split_name == "train" and hpo_subset_rows) else None
        if split_name != "train":
            max_rows = None  # Always use full vali and test sets

        logger.info(f"Loading {split_name} split from {path}...")
        X, y, qids = load_split(path, max_rows=max_rows, verbose=verbose)
        # Sort by qid for LightGBM
        X, y, qids = sort_by_qid(X, y, qids)
        splits[split_name] = (X, y, qids)

    # Dataset statistics
    X_tr, y_tr, q_tr = splits["train"]
    X_va, y_va, q_va = splits["vali"]
    X_te, y_te, q_te = splits["test"]

    stats = {
        "dataset": "MSLR-WEB10K Fold1",
        "num_features": NUM_FEATURES,
        "train": {
            "n_docs": int(len(y_tr)),
            "n_queries": int(len(np.unique(q_tr))),
            "label_distribution": {
                str(k): int(v) for k, v in sorted(Counter(y_tr.tolist()).items())
            },
        },
        "vali": {
            "n_docs": int(len(y_va)),
            "n_queries": int(len(np.unique(q_va))),
            "label_distribution": {
                str(k): int(v) for k, v in sorted(Counter(y_va.tolist()).items())
            },
        },
        "test": {
            "n_docs": int(len(y_te)),
            "n_queries": int(len(np.unique(q_te))),
            "label_distribution": {
                str(k): int(v) for k, v in sorted(Counter(y_te.tolist()).items())
            },
        },
    }
    splits["stats"] = stats

    logger.info("\n=== MSLR-WEB10K Dataset Statistics ===")
    for split_name in ["train", "vali", "test"]:
        s = stats[split_name]
        logger.info(
            f"  {split_name:5s}: {s['n_docs']:>8,} docs | {s['n_queries']:>5,} queries | "
            f"labels: {s['label_distribution']}"
        )

    return splits
