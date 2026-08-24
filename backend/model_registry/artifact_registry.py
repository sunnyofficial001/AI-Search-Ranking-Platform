"""
Model Registry — Metadata Persistence & Version Management
============================================================
Tracks trained model artifacts with their hyperparameters, evaluation metrics,
feature schema, dataset provenance, and training timestamps.

Every time a model is trained:
  1. Its artifact is saved to disk.
  2. A metadata record is written to models/registry.json.
  3. The champion model pointer is updated.

Loading always validates that the artifact file exists and the feature schema matches.
"""

import os
import json
import hashlib
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Model artifact directory
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"

NUM_FEATURES = 136  # MSLR-WEB10K feature count

ARTIFACT_MAP = {
    "pointwise_xgb":     "pointwise_xgboost.json",
    "pairwise_ranknet":  "pairwise_ranknet.pt",
    "listwise_lambdamart": "listwise_lambdamart.txt",
}


def _git_commit_hash() -> str:
    """Return current short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(MODELS_DIR.parent)
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file for integrity verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry() -> Dict[str, Any]:
    """Load the registry JSON, initializing it if missing."""
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Registry file corrupted ({e}), reinitializing.")
    return {"champion": {}, "models": {}}


def _save_registry(registry: Dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    model_key: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    evaluation_metrics: Dict[str, float],
    dataset_info: Dict[str, Any],
    artifact_path: Optional[str] = None,
    num_features: int = NUM_FEATURES,
) -> Dict[str, Any]:
    """
    Register a trained model in the artifact registry.

    Args:
        model_key: Unique identifier, e.g. 'pointwise_xgb', 'listwise_lambdamart'.
        algorithm: Human-readable algorithm name.
        hyperparameters: Dict of training hyperparameters.
        evaluation_metrics: Dict of evaluation metric values (NDCG@10, MAP, etc.).
        dataset_info: Dict with dataset name, fold, and split sizes.
        artifact_path: Path to model file. Defaults to standard MODELS_DIR location.
        num_features: Feature count for schema validation.

    Returns:
        The created metadata record.
    """
    if artifact_path is None:
        filename = ARTIFACT_MAP.get(model_key)
        if not filename:
            raise ValueError(f"Unknown model_key '{model_key}'. Add it to ARTIFACT_MAP.")
        artifact_path = str(MODELS_DIR / filename)

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {artifact_path}. "
            f"Train the model before registering."
        )

    record = {
        "model_key": model_key,
        "algorithm": algorithm,
        "version": datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "artifact_path": artifact_path,
        "artifact_sha256": _file_sha256(artifact_path),
        "artifact_size_bytes": os.path.getsize(artifact_path),
        "num_features": num_features,
        "feature_schema_version": "mslr_web10k_136",
        "hyperparameters": hyperparameters,
        "evaluation_metrics": evaluation_metrics,
        "dataset_info": dataset_info,
    }

    registry = _load_registry()
    if "models" not in registry:
        registry["models"] = {}
    registry["models"][model_key] = record

    # Update champion if this model has the highest NDCG@10
    champion = registry.get("champion", {})
    current_best = champion.get("ndcg10", -1.0)
    this_ndcg10 = evaluation_metrics.get("NDCG@10", evaluation_metrics.get("ndcg10", 0.0))
    if this_ndcg10 > current_best:
        registry["champion"] = {
            "model_key": model_key,
            "algorithm": algorithm,
            "ndcg10": this_ndcg10,
            "version": record["version"],
        }

    _save_registry(registry)
    logger.info(f"Registered model '{model_key}' v{record['version']} → NDCG@10={this_ndcg10:.5f}")
    return record


def validate_artifact(model_key: str) -> bool:
    """
    Validate that the registered artifact exists and its SHA-256 checksum matches.
    Returns True if valid, raises RuntimeError otherwise.
    """
    registry = _load_registry()
    record = registry.get("models", {}).get(model_key)
    if not record:
        raise KeyError(f"Model '{model_key}' is not in the registry.")

    artifact_path = record["artifact_path"]
    if not os.path.exists(artifact_path):
        raise RuntimeError(f"Artifact file missing: {artifact_path}")

    expected_sha = record.get("artifact_sha256")
    if expected_sha:
        actual_sha = _file_sha256(artifact_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for '{model_key}'! "
                f"Expected {expected_sha[:12]}... got {actual_sha[:12]}... "
                f"The artifact may have been corrupted or overwritten."
            )

    expected_features = record.get("num_features", NUM_FEATURES)
    if expected_features != NUM_FEATURES:
        raise RuntimeError(
            f"Feature schema mismatch: registry has {expected_features} features, "
            f"inference pipeline expects {NUM_FEATURES}."
        )

    logger.info(f"Artifact validation PASSED for '{model_key}'.")
    return True


def get_all_registered_models() -> List[Dict[str, Any]]:
    """Return all registered model records from the registry."""
    registry = _load_registry()
    return list(registry.get("models", {}).values())


def get_champion() -> Optional[Dict[str, Any]]:
    """Return the current champion model metadata."""
    registry = _load_registry()
    champ = registry.get("champion", {})
    if not champ:
        return None
    model_key = champ.get("model_key")
    return registry.get("models", {}).get(model_key)
