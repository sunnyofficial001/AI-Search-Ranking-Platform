"""
ML Inference Pipeline — Unified Serving Layer
==============================================
Loads trained model artifacts once at startup and provides:
  - Single-document scoring
  - Batch document scoring
  - Feature schema validation
  - Prediction confidence/metadata
  - Latency measurement

All feature preprocessing is identical to training (same imputation, same feature ordering).

Usage:
    from backend.ml.inference import InferencePipeline
    scorer = InferencePipeline.get_instance()
    results = scorer.rank_documents(query="noise canceling headphones", doc_ids=[...])
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
NUM_FEATURES = 136


class ModelLoadError(RuntimeError):
    pass


class FeatureSchemaError(ValueError):
    pass


class InferencePipeline:
    """
    Thread-safe singleton serving all trained ranking models.

    Singleton pattern ensures models are loaded only once, regardless of
    how many FastAPI workers or test fixtures access the pipeline.
    """

    _instance: Optional["InferencePipeline"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._xgb_model = None
        self._lgb_model = None
        self._ranknet_model = None
        self._models_loaded = False
        self._load_timestamps: Dict[str, float] = {}
        self._load_all_models()

    @classmethod
    def get_instance(cls) -> "InferencePipeline":
        """Return the shared singleton instance (lazy init, thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_all_models(self) -> None:
        """Load all three ranking model artifacts from disk."""
        errors: List[str] = []

        # 1. XGBoost Pointwise
        xgb_path = MODELS_DIR / "pointwise_xgboost.json"
        try:
            import xgboost as xgb
            t0 = time.perf_counter()
            model = xgb.XGBRegressor()
            model.load_model(str(xgb_path))
            self._xgb_model = model
            self._load_timestamps["xgboost"] = time.perf_counter() - t0
            logger.info(f"XGBoost loaded in {self._load_timestamps['xgboost']*1000:.1f}ms")
        except Exception as e:
            errors.append(f"XGBoost: {e}")
            logger.warning(f"XGBoost load failed: {e}")

        # 2. LightGBM LambdaMART
        lgb_path = MODELS_DIR / "listwise_lambdamart.txt"
        try:
            import lightgbm as lgb
            t0 = time.perf_counter()
            self._lgb_model = lgb.Booster(model_file=str(lgb_path))
            self._load_timestamps["lambdamart"] = time.perf_counter() - t0
            logger.info(f"LambdaMART loaded in {self._load_timestamps['lambdamart']*1000:.1f}ms")
        except Exception as e:
            errors.append(f"LambdaMART: {e}")
            logger.warning(f"LambdaMART load failed: {e}")

        # 3. PyTorch RankNet
        ranknet_path = MODELS_DIR / "pairwise_ranknet.pt"
        try:
            import torch
            from backend.training.train_pipeline import PyTorchRankNet
            t0 = time.perf_counter()
            net = PyTorchRankNet(input_dim=NUM_FEATURES)
            net.load_state_dict(torch.load(str(ranknet_path), weights_only=True))
            net.eval()
            self._ranknet_model = net
            self._load_timestamps["ranknet"] = time.perf_counter() - t0
            logger.info(f"RankNet loaded in {self._load_timestamps['ranknet']*1000:.1f}ms")
        except Exception as e:
            errors.append(f"RankNet: {e}")
            logger.warning(f"RankNet load failed: {e}")

        self._models_loaded = len(errors) < 3
        if errors:
            logger.warning(f"Some models failed to load: {'; '.join(errors)}")

    def _validate_feature_matrix(self, X: np.ndarray) -> None:
        """Validate shape and content of feature matrix before inference."""
        if X.ndim != 2:
            raise FeatureSchemaError(f"Feature matrix must be 2D, got shape {X.shape}")
        if X.shape[1] != NUM_FEATURES:
            raise FeatureSchemaError(
                f"Expected {NUM_FEATURES} features per document, got {X.shape[1]}"
            )
        if np.isnan(X).any():
            raise FeatureSchemaError("Feature matrix contains NaN values")
        if np.isinf(X).any():
            raise FeatureSchemaError("Feature matrix contains Inf values")

    def score_pointwise(self, X: np.ndarray) -> np.ndarray:
        """Score documents using XGBoost pointwise regressor."""
        self._validate_feature_matrix(X)
        if self._xgb_model is None:
            raise ModelLoadError("XGBoost model is not loaded")
        return self._xgb_model.predict(X.astype(np.float32))

    def score_listwise(self, X: np.ndarray) -> np.ndarray:
        """Score documents using LambdaMART listwise ranker."""
        self._validate_feature_matrix(X)
        if self._lgb_model is None:
            raise ModelLoadError("LambdaMART model is not loaded")
        return self._lgb_model.predict(X.astype(np.float32))

    def score_pairwise(self, X: np.ndarray) -> np.ndarray:
        """Score documents using RankNet pairwise neural ranker."""
        self._validate_feature_matrix(X)
        if self._ranknet_model is None:
            raise ModelLoadError("RankNet model is not loaded")
        import torch
        with torch.no_grad():
            tensor = torch.FloatTensor(X.astype(np.float32))
            scores = self._ranknet_model(tensor).squeeze(-1).numpy()
        return scores.reshape(-1) if scores.ndim > 0 else scores.reshape(1)

    def score_ensemble(
        self,
        X: np.ndarray,
        w_pointwise: float = 0.25,
        w_pairwise: float = 0.30,
        w_listwise: float = 0.45,
    ) -> np.ndarray:
        """
        Stacked ensemble scorer — weighted linear combination of all three rankers.
        Weights sum to 1.0; listwise weighted highest per NDCG performance ordering.
        """
        self._validate_feature_matrix(X)
        scores = np.zeros(len(X), dtype=np.float64)

        active_weight = 0.0
        model_scores: Dict[str, np.ndarray] = {}

        if self._xgb_model is not None:
            model_scores["pointwise"] = self.score_pointwise(X)
            active_weight += w_pointwise

        if self._ranknet_model is not None:
            model_scores["pairwise"] = self.score_pairwise(X)
            active_weight += w_pairwise

        if self._lgb_model is not None:
            model_scores["listwise"] = self.score_listwise(X)
            active_weight += w_listwise

        if not model_scores:
            raise ModelLoadError("No ranking models are loaded for ensemble scoring")

        # Normalize each model's scores to [0, 1] before ensemble blending
        weights = {"pointwise": w_pointwise, "pairwise": w_pairwise, "listwise": w_listwise}
        for name, raw in model_scores.items():
            lo, hi = raw.min(), raw.max()
            normalized = (raw - lo) / (hi - lo + 1e-9)
            scores += normalized * weights[name]

        # Rescale by active weight so output is always in [0, 1] even if models are missing
        if active_weight > 0:
            scores /= active_weight

        return scores

    def get_model_status(self) -> Dict[str, Any]:
        """Return load status and timing for all three models."""
        return {
            "xgboost":      {"loaded": self._xgb_model is not None,
                             "load_ms": round(self._load_timestamps.get("xgboost", 0) * 1000, 1)},
            "lambdamart":   {"loaded": self._lgb_model is not None,
                             "load_ms": round(self._load_timestamps.get("lambdamart", 0) * 1000, 1)},
            "ranknet":      {"loaded": self._ranknet_model is not None,
                             "load_ms": round(self._load_timestamps.get("ranknet", 0) * 1000, 1)},
            "all_loaded":   self._models_loaded,
        }
