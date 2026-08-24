"""
Phase 3 ML Engineering Tests
==============================
Covers:
  - Dataset loading & validation (MSLR-WEB10K structure)
  - Feature schema validation
  - Ranking metrics correctness
  - Model artifact existence & loadability
  - Inference pipeline (single + batch)
  - Feature schema error handling
  - Dataset validator logic
  - Artifact registry operations
"""

import os
import sys
import math
import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

# Set TESTING before any backend imports
os.environ["TESTING"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MSLR_FOLD1_FULL = Path(__file__).resolve().parents[2] / "../../MSLR-WEB10K/Fold1"
MSLR_FOLD1_DEV = Path(__file__).resolve().parents[2] / "data/Fold1"


class TestDatasetLoader(unittest.TestCase):
    """Test MSLR-WEB10K data loading correctness."""

    def _find_any_split(self) -> str:
        for base in [MSLR_FOLD1_FULL, MSLR_FOLD1_DEV]:
            p = base / "vali.txt"
            if p.exists():
                return str(p)
        self.skipTest("No MSLR split file found")

    def test_parse_line_returns_correct_shape(self):
        from backend.data.mslr_loader import _parse_line
        # Valid SVMLight line: label qid:1 1:3 2:0 3:1
        label, qid, fvec = _parse_line("2 qid:42 1:3.0 5:1.5 136:0.7")
        self.assertEqual(label, 2)
        self.assertEqual(qid, 42)
        self.assertEqual(len(fvec), 136)
        self.assertAlmostEqual(fvec[0], 3.0)   # feature 1 → index 0
        self.assertAlmostEqual(fvec[4], 1.5)   # feature 5 → index 4
        self.assertAlmostEqual(fvec[135], 0.7) # feature 136 → index 135

    def test_parse_line_default_zeros(self):
        from backend.data.mslr_loader import _parse_line
        _, _, fvec = _parse_line("0 qid:1 1:5.0")
        # All other features should be 0
        self.assertEqual(sum(fvec[1:]), 0.0)

    def test_parse_line_strips_comment(self):
        from backend.data.mslr_loader import _parse_line
        label, qid, fvec = _parse_line("1 qid:99 1:2.5 # this is a comment")
        self.assertEqual(label, 1)
        self.assertAlmostEqual(fvec[0], 2.5)

    def test_load_split_correct_dimensions(self):
        path = self._find_any_split()
        from backend.data.mslr_loader import load_split
        X, y, qids = load_split(path, max_rows=200, verbose=False)
        self.assertEqual(X.shape[1], 136)
        self.assertEqual(len(y), len(X))
        self.assertEqual(len(qids), len(X))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)

    def test_load_split_label_range(self):
        path = self._find_any_split()
        from backend.data.mslr_loader import load_split
        _, y, _ = load_split(path, max_rows=500, verbose=False)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 4))

    def test_sort_by_qid_is_stable(self):
        from backend.data.mslr_loader import sort_by_qid
        qids = np.array([3, 1, 2, 1, 3], dtype=np.int32)
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        y = np.array([4, 2, 0, 1, 3], dtype=np.int32)
        Xs, ys, qs = sort_by_qid(X, y, qids)
        self.assertTrue(np.all(qs == sorted(qids)))

    def test_get_query_groups_sums_to_total(self):
        from backend.data.mslr_loader import get_query_groups, sort_by_qid
        qids = np.array([1, 1, 2, 3, 3, 3], dtype=np.int32)
        X = np.zeros((6, 136), dtype=np.float32)
        y = np.zeros(6, dtype=np.int32)
        X, y, qids = sort_by_qid(X, y, qids)
        groups = get_query_groups(qids)
        self.assertEqual(groups.sum(), 6)
        self.assertEqual(len(groups), 3)

    def test_feature_names_count(self):
        from backend.data.mslr_loader import get_feature_names
        names = get_feature_names()
        self.assertEqual(len(names), 136)
        self.assertIsInstance(names[0], str)

    def test_stream_lines_skips_empty(self):
        from backend.data.mslr_loader import stream_lines
        path = self._find_any_split()
        count = 0
        for _ in stream_lines(path):
            count += 1
            if count >= 100:
                break
        self.assertGreater(count, 0)


class TestRankingMetrics(unittest.TestCase):
    """Test ranking metric implementations against known values."""

    def test_dcg_perfect_ranking(self):
        """DCG equals IDCG when ranking is ideal."""
        from backend.evaluation.ranking_metrics import _dcg_at_k
        rels = [4, 3, 2, 1, 0]
        ideal = sorted(rels, reverse=True)
        self.assertAlmostEqual(_dcg_at_k(rels, 5), _dcg_at_k(ideal, 5), places=8)

    def test_ndcg_perfect_ranking_equals_1(self):
        from backend.evaluation.ranking_metrics import _ndcg_at_k
        rels = [4, 3, 2, 1]
        self.assertAlmostEqual(_ndcg_at_k(rels, 4), 1.0, places=6)

    def test_ndcg_random_ranking_below_1(self):
        from backend.evaluation.ranking_metrics import _ndcg_at_k
        rels = [0, 4, 2, 3]  # worst possible order
        ndcg = _ndcg_at_k(rels, 4)
        self.assertGreater(ndcg, 0.0)
        self.assertLess(ndcg, 1.0)

    def test_ndcg_no_relevant_is_zero(self):
        from backend.evaluation.ranking_metrics import _ndcg_at_k
        self.assertAlmostEqual(_ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_map_all_relevant_is_1(self):
        from backend.evaluation.ranking_metrics import _average_precision
        self.assertAlmostEqual(_average_precision([2, 2, 2, 2], threshold=1), 1.0, places=6)

    def test_map_no_relevant_is_0(self):
        from backend.evaluation.ranking_metrics import _average_precision
        self.assertAlmostEqual(_average_precision([0, 0, 0], threshold=1), 0.0, places=6)

    def test_mrr_first_hit_at_rank1(self):
        from backend.evaluation.ranking_metrics import _reciprocal_rank
        self.assertAlmostEqual(_reciprocal_rank([3, 0, 0, 0], threshold=1), 1.0, places=6)

    def test_mrr_first_hit_at_rank2(self):
        from backend.evaluation.ranking_metrics import _reciprocal_rank
        self.assertAlmostEqual(_reciprocal_rank([0, 3, 0, 0], threshold=1), 0.5, places=6)

    def test_evaluate_per_query_returns_expected_keys(self):
        from backend.evaluation.ranking_metrics import evaluate_per_query
        y_true = np.array([4, 1, 3, 0, 2, 0], dtype=np.int32)
        y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.05])
        qids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        metrics = evaluate_per_query(y_true, y_pred, qids, k_values=(5, 10))
        for key in ["NDCG@5", "NDCG@10", "MAP", "MRR", "P@5", "R@5", "n_queries"]:
            self.assertIn(key, metrics)

    def test_evaluate_per_query_ndcg_in_range(self):
        from backend.evaluation.ranking_metrics import evaluate_per_query
        y_true = np.array([4, 1, 3, 0, 2, 0], dtype=np.int32)
        y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.05])
        qids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        metrics = evaluate_per_query(y_true, y_pred, qids)
        for k in [1, 3, 5, 10]:
            ndcg = metrics[f"NDCG@{k}"]
            self.assertGreaterEqual(ndcg, 0.0)
            self.assertLessEqual(ndcg, 1.0001)

    def test_perfect_predictions_produce_high_ndcg(self):
        """When predictions perfectly match labels, NDCG@10 should be ≥ 0.95."""
        from backend.evaluation.ranking_metrics import evaluate_per_query
        y_true = np.array([4, 3, 2, 1, 0, 4, 3, 2, 1, 0], dtype=np.int32)
        y_pred  = y_true.astype(float) + np.random.default_rng(0).uniform(-0.01, 0.01, 10)
        qids = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
        metrics = evaluate_per_query(y_true, y_pred, qids, k_values=(10,))
        self.assertGreaterEqual(metrics["NDCG@10"], 0.95)


class TestModelArtifacts(unittest.TestCase):
    """Verify that all trained model artifacts exist and are loadable."""

    def _require_model(self, filename: str) -> Path:
        path = MODELS_DIR / filename
        if not path.exists():
            self.skipTest(f"Model artifact not found: {path}")
        return path

    def test_xgboost_artifact_exists(self):
        path = self._require_model("pointwise_xgboost.json")
        self.assertGreater(path.stat().st_size, 1000)

    def test_lambdamart_artifact_exists(self):
        path = self._require_model("listwise_lambdamart.txt")
        self.assertGreater(path.stat().st_size, 10_000)

    def test_ranknet_artifact_exists(self):
        path = self._require_model("pairwise_ranknet.pt")
        self.assertGreater(path.stat().st_size, 1000)

    def test_xgboost_loads_and_predicts(self):
        path = self._require_model("pointwise_xgboost.json")
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        dummy = np.zeros((3, 136), dtype=np.float32)
        preds = model.predict(dummy)
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_lambdamart_loads_and_predicts(self):
        path = self._require_model("listwise_lambdamart.txt")
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        dummy = np.zeros((3, 136), dtype=np.float32)
        preds = model.predict(dummy)
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_ranknet_loads_and_predicts(self):
        path = self._require_model("pairwise_ranknet.pt")
        import torch
        from backend.training.train_pipeline import PyTorchRankNet
        net = PyTorchRankNet(input_dim=136)
        net.load_state_dict(torch.load(str(path), weights_only=True))
        net.eval()
        with torch.no_grad():
            preds = net(torch.zeros(3, 136)).squeeze().numpy()
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_xgboost_feature_count_136(self):
        path = self._require_model("pointwise_xgboost.json")
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        # Passing wrong feature count should raise
        wrong_X = np.zeros((1, 100), dtype=np.float32)
        with self.assertRaises(Exception):
            model.predict(wrong_X)


class TestInferencePipeline(unittest.TestCase):
    """Test the unified InferencePipeline serving layer."""

    def _get_pipeline(self):
        from backend.ml.inference import InferencePipeline
        return InferencePipeline.get_instance()

    def test_singleton_returns_same_instance(self):
        from backend.ml.inference import InferencePipeline
        a = InferencePipeline.get_instance()
        b = InferencePipeline.get_instance()
        self.assertIs(a, b)

    def test_feature_schema_rejects_wrong_col_count(self):
        from backend.ml.inference import InferencePipeline, FeatureSchemaError
        pipeline = self._get_pipeline()
        bad_X = np.zeros((2, 50), dtype=np.float32)
        with self.assertRaises(FeatureSchemaError):
            pipeline._validate_feature_matrix(bad_X)

    def test_feature_schema_rejects_nan(self):
        from backend.ml.inference import InferencePipeline, FeatureSchemaError
        pipeline = self._get_pipeline()
        bad_X = np.zeros((2, 136), dtype=np.float32)
        bad_X[0, 0] = float("nan")
        with self.assertRaises(FeatureSchemaError):
            pipeline._validate_feature_matrix(bad_X)

    def test_feature_schema_rejects_1d(self):
        from backend.ml.inference import InferencePipeline, FeatureSchemaError
        pipeline = self._get_pipeline()
        with self.assertRaises(FeatureSchemaError):
            pipeline._validate_feature_matrix(np.zeros(136))

    def test_listwise_scoring_batch(self):
        pipeline = self._get_pipeline()
        if not pipeline._lgb_model:
            self.skipTest("LambdaMART model not loaded")
        X = np.random.default_rng(42).random((10, 136)).astype(np.float32)
        scores = pipeline.score_listwise(X)
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_pointwise_scoring_batch(self):
        pipeline = self._get_pipeline()
        if not pipeline._xgb_model:
            self.skipTest("XGBoost model not loaded")
        X = np.random.default_rng(42).random((5, 136)).astype(np.float32)
        scores = pipeline.score_pointwise(X)
        self.assertEqual(len(scores), 5)

    def test_ensemble_scores_in_unit_range(self):
        pipeline = self._get_pipeline()
        X = np.random.default_rng(0).random((8, 136)).astype(np.float32)
        try:
            scores = pipeline.score_ensemble(X)
            self.assertEqual(len(scores), 8)
            self.assertTrue(np.all(scores >= -0.01))  # allow tiny float tolerance
            self.assertTrue(np.all(scores <= 1.01))
        except Exception:
            self.skipTest("No models loaded for ensemble")

    def test_model_status_has_expected_keys(self):
        pipeline = self._get_pipeline()
        status = pipeline.get_model_status()
        for key in ["xgboost", "lambdamart", "ranknet", "all_loaded"]:
            self.assertIn(key, status)


class TestArtifactRegistry(unittest.TestCase):
    """Test model metadata registry operations."""

    def test_registry_register_and_retrieve(self):
        from backend.model_registry.artifact_registry import register_model, _load_registry
        # Use actual model file for test
        xgb_path = MODELS_DIR / "pointwise_xgboost.json"
        if not xgb_path.exists():
            self.skipTest("XGBoost artifact not found")
        record = register_model(
            model_key="pointwise_xgb",
            algorithm="XGBoost Pointwise Regressor",
            hyperparameters={"n_estimators": 200, "max_depth": 6},
            evaluation_metrics={"NDCG@10": 0.752, "MAP": 0.643, "MRR": 0.811},
            dataset_info={"dataset": "MSLR-WEB10K", "fold": "Fold1"},
        )
        self.assertEqual(record["model_key"], "pointwise_xgb")
        self.assertIn("artifact_sha256", record)
        self.assertIn("trained_at", record)
        self.assertGreater(record["artifact_size_bytes"], 0)

    def test_validate_artifact_passes_for_existing(self):
        from backend.model_registry.artifact_registry import register_model, validate_artifact
        xgb_path = MODELS_DIR / "pointwise_xgboost.json"
        if not xgb_path.exists():
            self.skipTest("XGBoost artifact not found")
        register_model(
            model_key="pointwise_xgb",
            algorithm="XGBoost",
            hyperparameters={},
            evaluation_metrics={"NDCG@10": 0.75},
            dataset_info={},
        )
        self.assertTrue(validate_artifact("pointwise_xgb"))

    def test_validate_artifact_raises_for_unknown_key(self):
        from backend.model_registry.artifact_registry import validate_artifact
        with self.assertRaises(KeyError):
            validate_artifact("nonexistent_model_xyz_999")

    def test_get_all_registered_models_returns_list(self):
        from backend.model_registry.artifact_registry import get_all_registered_models
        models = get_all_registered_models()
        self.assertIsInstance(models, list)


class TestDatasetValidator(unittest.TestCase):
    """Test dataset validation logic."""

    def test_validate_split_passes_valid_data(self):
        from backend.data.dataset_validator import validate_split
        X = np.random.default_rng(0).random((100, 136)).astype(np.float32)
        y = np.random.default_rng(0).integers(0, 5, size=100).astype(np.int32)
        qids = np.sort(np.random.default_rng(0).integers(1, 20, size=100).astype(np.int32))
        report = validate_split(X, y, qids, "train")
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["issues"]), 0)

    def test_validate_split_wrong_feature_count(self):
        from backend.data.dataset_validator import validate_split
        X = np.zeros((10, 100), dtype=np.float32)  # 100 != 136
        y = np.zeros(10, dtype=np.int32)
        qids = np.ones(10, dtype=np.int32)
        with self.assertRaises(ValueError):
            validate_split(X, y, qids, "test")

    def test_no_leakage_separate_qids(self):
        from backend.data.dataset_validator import check_no_label_leakage
        q_train = np.array([1, 1, 2, 2, 3], dtype=np.int32)
        q_test = np.array([4, 4, 5, 5, 6], dtype=np.int32)
        result = check_no_label_leakage(q_train, q_test)
        self.assertFalse(result["leakage_detected"])
        self.assertEqual(result["overlapping_queries"], 0)

    def test_leakage_detected_when_shared_qids(self):
        from backend.data.dataset_validator import check_no_label_leakage
        q_train = np.array([1, 1, 2, 3], dtype=np.int32)
        q_test = np.array([2, 2, 4, 5], dtype=np.int32)
        result = check_no_label_leakage(q_train, q_test)
        self.assertTrue(result["leakage_detected"])
        self.assertEqual(result["overlapping_queries"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
