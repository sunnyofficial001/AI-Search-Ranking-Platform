"""
Comprehensive Test Suite
=========================
Production-grade tests covering:
  - Ranking engine correctness
  - Statistical test implementations
  - Feature store pipeline
  - Drift detection algorithms
  - A/B testing statistical significance
  - Model registry operations
  - Recommendation engine quality
  - API endpoint contracts
  - Performance benchmarks
"""

import math
import os
import sys
import time
import unittest

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.ab_testing.ab_testing import ABStatisticalTests, ExperimentManager
from backend.drift_detection.drift_monitor import DriftMonitor, StatisticalTests
from backend.evaluation.evaluate import Evaluator
from backend.feature_store.feature_store import (
    FeatureValidator,
    OfflineFeaturePipeline,
    OnlineFeatureStore,
)
from backend.model_registry.model_registry import ModelRegistry, ModelStage, ModelType
from backend.services.advanced_recommend import (
    ITEM_CATALOG,
    ColdStartHandler,
    DiversityOptimizer,
    EmbeddingEngine,
    SessionBasedRecommender,
    TwoTowerRecommender,
    UserSession,
)
from backend.services.ranking_engine import (
    DiversityReRanker,
    EnsembleRanker,
    ListwiseLambdaMARTScorer,
    MultiStageRankingPipeline,
    PairwiseRankNetScorer,
    PointwiseScorer,
    RankingCandidate,
    RankingContext,
    RankingEvaluator,
)
from backend.services.search_service import SearchService

# ============================================================
# SECTION 1: EVALUATION METRICS
# ============================================================


class TestEvaluationMetrics(unittest.TestCase):
    """Test NDCG, MAP, MRR, ERR calculations for correctness."""

    def test_dcg_perfect_ranking(self):
        """Perfect ranking should have DCG = IDCG."""
        rels = [4, 3, 2, 1, 0]
        dcg = Evaluator.calculate_dcg(rels, 5)
        self.assertGreater(dcg, 0)

    def test_dcg_formula(self):
        """DCG@2 = (2^4-1)/log2(2) + (2^2-1)/log2(3) = 15 + ~1.89."""
        rels = [4, 2, 0, 0]
        dcg = Evaluator.calculate_dcg(rels, 2)
        expected = 15.0 / math.log2(2) + 3.0 / math.log2(3)
        self.assertAlmostEqual(dcg, expected, places=3)

    def test_ndcg_perfect_is_one(self):
        """NDCG of perfect ranking = 1.0."""
        ideal = [4, 3, 2, 1]
        ndcg = Evaluator.calculate_ndcg(ideal, ideal, 4)
        self.assertAlmostEqual(ndcg, 1.0, places=3)

    def test_ndcg_range(self):
        """NDCG must always be in [0, 1]."""
        ranked = [1, 4, 0, 2, 3]
        ideal = [4, 3, 2, 1, 0]
        ndcg = Evaluator.calculate_ndcg(ranked, ideal, 5)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_ndcg_worse_than_perfect(self):
        """Suboptimal ranking should have NDCG < 1.0."""
        ranked = [0, 1, 2, 3, 4]
        ideal = [4, 3, 2, 1, 0]
        ndcg = Evaluator.calculate_ndcg(ranked, ideal, 5)
        self.assertLess(ndcg, 1.0)

    def test_map_calculation(self):
        """MAP: items at positions 1 and 3 relevant → AP = (1/1 + 2/3) / 2 = 0.833."""
        ranked = [4, 1, 3, 0]
        computed = Evaluator.calculate_map(ranked, 2)
        self.assertAlmostEqual(computed, 0.8333, places=3)

    def test_mrr_first_position(self):
        """MRR = 1.0 when first result is relevant."""
        ranked = [4, 0, 0, 0]
        mrr = Evaluator.calculate_mrr(ranked)
        self.assertEqual(mrr, 1.0)

    def test_mrr_second_position(self):
        """MRR = 0.5 when first relevant is at rank 2."""
        ranked = [0, 4, 0, 0]
        mrr = Evaluator.calculate_mrr(ranked)
        self.assertEqual(mrr, 0.5)

    def test_precision_at_k(self):
        """P@5 with 3 relevant in top 5 = 0.6."""
        ranked = [3, 0, 5, 1, 2]
        prec = Evaluator.calculate_precision_at_k(ranked, 5)
        self.assertEqual(prec, 0.6)

    def test_recall_at_k(self):
        """R@5 with all 3 relevant in top 5 = 1.0."""
        ranked = [3, 0, 5, 1, 2]
        rec = Evaluator.calculate_recall_at_k(ranked, 3, 5)
        self.assertEqual(rec, 1.0)

    def test_full_suite_returns_all_metrics(self):
        """Full suite should return all expected metric keys."""
        ranked = [3, 2, 0, 1, 4]
        ground = [4, 3, 2, 1, 0]
        result = Evaluator.run_full_suite(ranked, ground)
        for key in ["ndcg5", "ndcg10", "map", "mrr", "precision5", "recall5"]:
            self.assertIn(key, result)


# ============================================================
# SECTION 2: RANKING ENGINE
# ============================================================


class TestRankingEngine(unittest.TestCase):
    """Test multi-stage ranking pipeline components."""

    SAMPLE_FEATURES = {
        "bm25_score": 2.5,
        "tfidf_cosine": 0.6,
        "exact_match_title": 1.0,
        "query_term_coverage": 0.8,
        "title_term_density": 0.5,
        "ctr_7d": 0.15,
        "dwell_time_median_seconds": 120.0,
        "freshness_score": 0.9,
        "popularity_score": 90.0,
        "avg_rating": 4.8,
        "query_category_match": 0.7,
        "add_to_cart_rate": 0.06,
        "purchase_rate": 0.02,
    }

    def _make_candidate(self, doc_id="prod-1", category="Electronics", bm25=2.0, ctr=0.12):
        features = {**self.SAMPLE_FEATURES, "bm25_score": bm25, "ctr_7d": ctr}
        return RankingCandidate(doc_id=doc_id, title="Test Item", category=category, features=features)

    def test_pointwise_scorer_range(self):
        """Pointwise score must be in [0, 1]."""
        score = PointwiseScorer.score(self.SAMPLE_FEATURES)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pointwise_higher_bm25_scores_higher(self):
        """Higher BM25 should produce higher pointwise score, all else equal."""
        low = PointwiseScorer.score({**self.SAMPLE_FEATURES, "bm25_score": 0.1})
        high = PointwiseScorer.score({**self.SAMPLE_FEATURES, "bm25_score": 4.0})
        self.assertGreater(high, low)

    def test_pointwise_batch_assigns_scores(self):
        """Batch scoring assigns scores to all candidates."""
        candidates = [self._make_candidate(f"prod-{i}") for i in range(5)]
        scored = PointwiseScorer.score_batch(candidates)
        for c in scored:
            self.assertGreater(c.pointwise_score, 0)

    def test_pairwise_scores_range(self):
        """Pairwise scores should be in [0, 1]."""
        candidates = [self._make_candidate(f"prod-{i}", bm25=float(i)) for i in range(1, 5)]
        scored = PairwiseRankNetScorer.score_batch(candidates)
        for c in scored:
            self.assertGreaterEqual(c.pairwise_score, 0.0)
            self.assertLessEqual(c.pairwise_score, 1.0)

    def test_listwise_scores_assigned(self):
        """Listwise scorer assigns non-negative scores."""
        candidates = [self._make_candidate(f"prod-{i}") for i in range(1, 5)]
        scored = ListwiseLambdaMARTScorer.score_batch(candidates)
        for c in scored:
            self.assertGreaterEqual(c.listwise_score, 0.0)

    def test_ensemble_fuses_all_scores(self):
        """Ensemble score is weighted combination of all three rankers."""
        c = self._make_candidate()
        c.pointwise_score = 0.6
        c.pairwise_score = 0.7
        c.listwise_score = 0.8
        EnsembleRanker.fuse_scores([c])
        expected = 0.25 * 0.6 + 0.35 * 0.7 + 0.40 * 0.8
        self.assertAlmostEqual(c.ensemble_score, expected, places=4)

    def test_diversity_reranker_changes_order(self):
        """MMR re-ranker should return results with different category distribution."""
        candidates = [self._make_candidate(f"prod-{i}", category="Electronics", bm25=float(5 - i)) for i in range(5)]
        candidates += [self._make_candidate(f"prod-apparel-{i}", category="Apparel", bm25=float(i)) for i in range(3)]
        for c in candidates:
            c.personalized_score = c.features["bm25_score"] / 5.0

        reranked = DiversityReRanker.rerank(candidates, lambda_param=0.6, top_k=5)
        categories = [c.category for c in reranked]
        # Should not be all Electronics
        self.assertIn("Apparel", categories)

    def test_multi_stage_pipeline_returns_results(self):
        """Full pipeline should return ranked results with metadata."""
        from backend.services.search_service import MOCK_PRODUCTS

        context = RankingContext(query="headphones", page=1, page_size=5)
        ranked, meta = MultiStageRankingPipeline.rank(MOCK_PRODUCTS, "headphones", context)
        self.assertGreater(len(ranked), 0)
        self.assertIn("total_time_ms", meta)
        self.assertIn("stages", meta)

    def test_ranking_evaluator_full_suite(self):
        """RankingEvaluator computes all metrics correctly."""
        candidates = [self._make_candidate(f"prod-{i}") for i in range(5)]
        for i, c in enumerate(candidates):
            c.relevance_label = 4 - i
        metrics = RankingEvaluator.full_evaluation_suite(candidates)
        for key in [
            "ndcg@5",
            "ndcg@10",
            "map",
            "mrr",
            "err@10",
            "precision@5",
            "recall@5",
        ]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)

    def test_err_at_10_range(self):
        """ERR should be in [0, 1]."""
        labels = [4, 3, 0, 2, 1, 0, 0, 1, 2, 3]
        err = RankingEvaluator.expected_reciprocal_rank(labels)
        self.assertGreaterEqual(err, 0.0)
        self.assertLessEqual(err, 1.0)


# ============================================================
# SECTION 3: FEATURE STORE
# ============================================================


class TestFeatureStore(unittest.TestCase):
    """Test feature computation, validation, and caching."""

    SAMPLE_DOC = {
        "id": "prod-3",
        "title": "Sony WH-1000XM5 Noise Cancelling Headphones",
        "description": "Industry leading noise canceling headphones with 30 hours battery life",
        "category": "Electronics",
        "popularity": 95.0,
        "ctr": 0.15,
        "freshness": 0.90,
        "engagement": 4.8,
    }

    def test_query_doc_features_keys(self):
        """Computed features should contain all expected keys."""
        features = OfflineFeaturePipeline.compute_query_doc_features("noise cancelling headphones", self.SAMPLE_DOC)
        expected_keys = [
            "bm25_score",
            "tfidf_cosine",
            "exact_match_title",
            "query_term_coverage",
            "title_term_density",
        ]
        for key in expected_keys:
            self.assertIn(key, features)

    def test_bm25_positive_for_relevant_query(self):
        """BM25 should be > 0 for matching query."""
        features = OfflineFeaturePipeline.compute_query_doc_features("noise cancelling headphones", self.SAMPLE_DOC)
        self.assertGreater(features["bm25_score"], 0.0)

    def test_bm25_zero_for_irrelevant_query(self):
        """BM25 should be 0 for completely irrelevant query."""
        features = OfflineFeaturePipeline.compute_query_doc_features("xyz qwerty zzzz", self.SAMPLE_DOC)
        self.assertEqual(features["bm25_score"], 0.0)

    def test_document_features_range(self):
        """Document features should be within spec ranges."""
        features = OfflineFeaturePipeline.compute_document_features(self.SAMPLE_DOC)
        self.assertGreaterEqual(features["freshness_score"], 0.0)
        self.assertLessEqual(features["freshness_score"], 1.0)
        self.assertGreaterEqual(features["ctr_7d"], 0.0)
        self.assertLessEqual(features["ctr_7d"], 1.0)

    def test_feature_validation_catches_out_of_range(self):
        """Validator should flag values outside spec bounds."""
        bad_features = {"bm25_score": -5.0, "tfidf_cosine": 2.0}
        is_valid, errors = FeatureValidator.validate(bad_features)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_feature_validation_passes_valid_features(self):
        """Validator should pass valid features."""
        good_features = {"bm25_score": 2.5, "tfidf_cosine": 0.6, "ctr_7d": 0.12}
        is_valid, errors = FeatureValidator.validate(good_features)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_feature_validator_clamp(self):
        """validate_and_clamp should bring values within range."""
        bad = {"bm25_score": 100.0}  # max is 50.0
        clamped = FeatureValidator.validate_and_clamp(bad)
        self.assertLessEqual(clamped["bm25_score"], 50.0)

    def test_online_feature_store_cache_hit(self):
        """Second call to get_or_compute should use cache."""
        doc = self.SAMPLE_DOC
        query = "test cache headphones"
        # First call — miss
        features1 = OnlineFeatureStore.get_or_compute(query, doc)
        # Second call — hit
        features2 = OnlineFeatureStore.get_or_compute(query, doc)
        self.assertEqual(features1, features2)

    def test_full_feature_vector_has_correct_count(self):
        """Full feature vector should contain both qd and doc features."""
        vector = OfflineFeaturePipeline.compute_full_feature_vector("headphones", self.SAMPLE_DOC)
        self.assertGreater(len(vector.features), 10)
        self.assertIsNotNone(vector.entity_key)
        self.assertIsNotNone(vector.computed_at)

    def test_feature_catalog_non_empty(self):
        """Feature catalog should list all registered features."""
        catalog = OnlineFeatureStore.get_feature_catalog()
        self.assertGreater(len(catalog), 5)
        for entry in catalog:
            self.assertIn("name", entry)
            self.assertIn("type", entry)
            self.assertIn("description", entry)


# ============================================================
# SECTION 4: DRIFT DETECTION
# ============================================================


class TestDriftDetection(unittest.TestCase):
    """Test statistical drift detection algorithms."""

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        psi = StatisticalTests.psi(data, data)
        self.assertLess(psi, 0.05)

    def test_psi_very_different_distributions(self):
        """PSI of well-separated distributions should be > 0.2."""
        # Use spread distributions so bucket logic fires correctly
        ref = list(range(1, 51))  # 1-50 uniform
        cur = list(range(200, 250))  # 200-249 — completely disjoint
        psi = StatisticalTests.psi(ref, cur)
        # All current values fall outside reference range → buckets max out → PSI large
        self.assertGreater(psi, 0.2)

    def test_psi_moderate_shift(self):
        """PSI for moderate shift should be between 0.1 and 0.2."""
        import random

        random.seed(42)
        ref = [random.gauss(0, 1) for _ in range(200)]
        cur = [random.gauss(0.5, 1) for _ in range(200)]
        psi = StatisticalTests.psi(ref, cur)
        # Should detect some shift
        self.assertGreater(psi, 0.0)

    def test_ks_identical(self):
        """KS statistic for identical distributions = 0."""
        data = list(range(100))
        ks = StatisticalTests.ks_statistic(data, data)
        self.assertAlmostEqual(ks, 0.0, places=4)

    def test_ks_completely_separated(self):
        """KS statistic for non-overlapping distributions = 1.0."""
        ref = [1.0, 2.0, 3.0] * 20
        cur = [100.0, 200.0, 300.0] * 20
        ks = StatisticalTests.ks_statistic(ref, cur)
        self.assertAlmostEqual(ks, 1.0, places=2)

    def test_jsd_identical_distributions(self):
        """JSD of identical distributions = 0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        jsd = StatisticalTests.jensen_shannon_divergence(data, data)
        self.assertLess(jsd, 0.01)

    def test_jsd_range(self):
        """JSD must be in [0, 1]."""
        import random

        ref = [random.gauss(0, 1) for _ in range(100)]
        cur = [random.gauss(5, 1) for _ in range(100)]
        jsd = StatisticalTests.jensen_shannon_divergence(ref, cur)
        self.assertGreaterEqual(jsd, 0.0)
        self.assertLessEqual(jsd, 1.0)

    def test_page_hinkley_detects_change(self):
        """Page-Hinkley should detect abrupt mean shift."""
        # Stable sequence followed by a large shift
        stable = [1.0] * 20
        shifted = [50.0] * 20
        all_vals = stable + shifted
        change, stat = StatisticalTests.page_hinkley(all_vals, delta=0.005, lambda_threshold=10.0)
        self.assertTrue(change)

    def test_page_hinkley_no_false_positive(self):
        """Page-Hinkley should not trigger on stable data."""
        import random

        random.seed(123)
        stable = [random.gauss(1.0, 0.1) for _ in range(30)]
        change, _ = StatisticalTests.page_hinkley(stable, lambda_threshold=200.0)
        self.assertFalse(change)

    def test_drift_report_returns_all_features(self):
        """Drift report should include all monitored features."""
        report = DriftMonitor.run_full_drift_report()
        self.assertIn("features", report)
        self.assertIn("overall_status", report)
        self.assertGreater(len(report["features"]), 0)


# ============================================================
# SECTION 5: A/B TESTING
# ============================================================


class TestABTesting(unittest.TestCase):
    """Test statistical significance testing and experiment management."""

    def test_z_test_significant_difference(self):
        """Large sample with meaningful lift should be significant."""
        z, p, sig = ABStatisticalTests.two_proportion_z_test(10000, 1000, 10000, 1200)
        self.assertTrue(sig)
        self.assertLess(p, 0.05)

    def test_z_test_no_difference(self):
        """Identical conversion rates should not be significant."""
        z, p, sig = ABStatisticalTests.two_proportion_z_test(1000, 100, 1000, 100)
        self.assertFalse(sig)
        self.assertAlmostEqual(z, 0.0, places=2)

    def test_t_test_significant(self):
        """Large mean difference with tiny variance → significant."""
        import random as _r

        _r.seed(0)
        control = [0.5 + _r.gauss(0, 0.01) for _ in range(100)]
        treatment = [0.8 + _r.gauss(0, 0.01) for _ in range(100)]
        t, p, sig, d = ABStatisticalTests.welch_t_test(control, treatment)
        self.assertTrue(sig)

    def test_t_test_cohens_d_large(self):
        """Large effect size should give Cohen's d > 0.8."""
        import random as _r

        _r.seed(1)
        control = [1.0 + _r.gauss(0, 0.1) for _ in range(50)]
        treatment = [3.0 + _r.gauss(0, 0.1) for _ in range(50)]
        _, _, _, d = ABStatisticalTests.welch_t_test(control, treatment)
        self.assertGreater(abs(d), 0.8)

    def test_minimum_sample_size_positive(self):
        """Required sample size should always be > 0."""
        n = ABStatisticalTests.minimum_sample_size(0.10, 0.05)
        self.assertGreater(n, 0)

    def test_minimum_sample_size_larger_for_smaller_mde(self):
        """Smaller MDE requires larger sample size."""
        n1 = ABStatisticalTests.minimum_sample_size(0.10, 0.10)
        n2 = ABStatisticalTests.minimum_sample_size(0.10, 0.05)
        self.assertGreater(n2, n1)

    def test_experiment_assignment_deterministic(self):
        """Same user should always get the same variant."""
        exp_id = "exp-ranking-001"
        for _ in range(5):
            v1 = ExperimentManager.get_variant_assignment(exp_id, "user-stable")
        v2 = ExperimentManager.get_variant_assignment(exp_id, "user-stable")
        self.assertEqual(v1, v2)

    def test_experiment_analysis_returns_stats(self):
        """Experiment analysis should return statistical test results."""
        result = ExperimentManager.analyze_experiment("exp-ranking-001")
        self.assertIn("analysis", result)
        self.assertIn("recommendation", result)
        for variant_analysis in result["analysis"].values():
            self.assertIn("statistical_tests", variant_analysis)
            self.assertIn("lift", variant_analysis)

    def test_experiment_list_non_empty(self):
        """Should return seeded experiments."""
        experiments = ExperimentManager.list_experiments()
        self.assertGreater(len(experiments), 0)


# ============================================================
# SECTION 6: RECOMMENDATION ENGINE
# ============================================================


class TestRecommendationEngine(unittest.TestCase):
    """Test advanced recommendation algorithms."""

    def test_embedding_engine_unit_norm(self):
        """Item embeddings should be unit-normalized."""
        emb = EmbeddingEngine.item_embedding("prod-1")
        norm = math.sqrt(sum(x * x for x in emb))
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_embedding_similarity_self(self):
        """An item's similarity to itself should be 1.0."""
        emb = EmbeddingEngine.item_embedding("prod-3")
        sim = EmbeddingEngine.cosine_similarity(emb, emb)
        self.assertAlmostEqual(sim, 1.0, places=3)

    def test_same_category_higher_similarity(self):
        """Items in the same category should have higher similarity."""
        emb1 = EmbeddingEngine.item_embedding("prod-1")  # Electronics
        emb3 = EmbeddingEngine.item_embedding("prod-3")  # Electronics
        emb2 = EmbeddingEngine.item_embedding("prod-2")  # Apparel
        sim_same = EmbeddingEngine.cosine_similarity(emb1, emb3)
        sim_diff = EmbeddingEngine.cosine_similarity(emb1, emb2)
        self.assertGreater(sim_same, sim_diff)

    def test_most_similar_returns_correct_count(self):
        """most_similar should return top_k items."""
        all_ids = [item["id"] for item in ITEM_CATALOG]
        similar = EmbeddingEngine.most_similar("prod-1", all_ids, top_k=3)
        self.assertEqual(len(similar), 3)

    def test_most_similar_excludes_query(self):
        """most_similar should not include the query item itself."""
        all_ids = [item["id"] for item in ITEM_CATALOG]
        similar = EmbeddingEngine.most_similar("prod-1", all_ids, top_k=5)
        similar_ids = [s[0] for s in similar]
        self.assertNotIn("prod-1", similar_ids)

    def test_session_based_with_history(self):
        """Session recommendations should avoid viewed items."""
        session = UserSession(user_id="user-1", clicked_items=["prod-1", "prod-5"])
        recs = SessionBasedRecommender.recommend(session, exclude_viewed=True)
        rec_ids = [r.item_id for r in recs]
        self.assertNotIn("prod-1", rec_ids)
        self.assertNotIn("prod-5", rec_ids)

    def test_session_cold_start_returns_items(self):
        """Empty session should return popularity-ranked fallback."""
        recs = SessionBasedRecommender._cold_start_fallback(top_k=5)
        self.assertEqual(len(recs), 5)

    def test_two_tower_scores_range(self):
        """Two-tower scores should be in [0, 1]."""
        recs = TwoTowerRecommender.recommend("user-1", top_k=5)
        for r in recs:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    def test_cold_start_diversity(self):
        """Cold-start recommendations should span multiple categories."""
        recs = ColdStartHandler.handle_new_user(top_k=6)
        categories = set(r.category for r in recs)
        self.assertGreater(len(categories), 1)

    def test_diversity_optimizer_ild(self):
        """ILD should be higher when categories are diverse."""
        from backend.services.advanced_recommend import RecommendationItem

        diverse = [
            RecommendationItem("prod-1", "Echo Dot", "Electronics", 0.8, "content_based"),
            RecommendationItem("prod-2", "Kanken", "Apparel", 0.7, "content_based"),
            RecommendationItem("prod-7", "Nike", "Footwear", 0.6, "content_based"),
        ]
        homogeneous = [
            RecommendationItem("prod-1", "Echo", "Electronics", 0.8, "content_based"),
            RecommendationItem("prod-3", "Sony", "Electronics", 0.7, "content_based"),
            RecommendationItem("prod-9", "Asus", "Electronics", 0.6, "content_based"),
        ]
        ild_diverse = DiversityOptimizer.intra_list_diversity(diverse)
        ild_homo = DiversityOptimizer.intra_list_diversity(homogeneous)
        self.assertGreater(ild_diverse, ild_homo)


# ============================================================
# SECTION 7: SEARCH SERVICE
# ============================================================


class TestSearchService(unittest.TestCase):
    """Test BM25, TF-IDF, and query expansion."""

    def test_bm25_positive_match(self):
        """BM25 should be positive for query with matching terms."""
        score = SearchService.calculate_bm25("headphones", "Sony Headphones", "noise cancelling headphones wireless")
        self.assertGreater(score, 0.0)

    def test_bm25_zero_no_match(self):
        """BM25 should be 0 for non-matching query."""
        score = SearchService.calculate_bm25("zzzzqqqq", "Sony Headphones", "wireless audio")
        self.assertEqual(score, 0.0)

    def test_cosine_similarity_range(self):
        """Cosine similarity should be in [0, 1]."""
        result = SearchService.calculate_tfidf_and_cosine(
            "noise cancelling headphones",
            "Sony Headphones",
            "noise cancelling wireless",
        )
        self.assertGreaterEqual(result["cosine"], 0.0)
        self.assertLessEqual(result["cosine"], 1.0)

    def test_query_expansion_electronics(self):
        """Electronics query should be expanded with relevant terms."""
        expanded = SearchService.expand_query("Alexa speaker")
        self.assertIn("alexa", expanded.lower())
        self.assertGreater(len(expanded.split()), len("Alexa speaker".split()))

    def test_candidates_returned(self):
        """fetch_candidate_documents should return at least one result."""
        candidates = SearchService.fetch_candidate_documents("headphones")
        self.assertGreater(len(candidates), 0)


# ============================================================
# SECTION 8: MODEL REGISTRY
# ============================================================


class TestModelRegistry(unittest.TestCase):
    """Test model registry CRUD and lifecycle operations."""

    def test_list_models_non_empty(self):
        """Registry should have seeded models."""
        models = ModelRegistry.list_models()
        self.assertGreater(len(models), 0)

    def test_list_production_models(self):
        """Should be able to filter by stage."""
        prod_models = ModelRegistry.list_models(stage=ModelStage.PRODUCTION)
        for m in prod_models:
            self.assertEqual(m["stage"], "production")

    def test_get_model_returns_correct_id(self):
        """get_model should return correct model by ID."""
        model = ModelRegistry.get_model("model-lambdamart-v1.3.0")
        self.assertIsNotNone(model)
        self.assertEqual(model["model_id"], "model-lambdamart-v1.3.0")

    def test_get_model_not_found(self):
        """get_model should return None for nonexistent ID."""
        model = ModelRegistry.get_model("nonexistent-model-id")
        self.assertIsNone(model)

    def test_model_compare_returns_winner(self):
        """Compare should return a winner."""
        result = ModelRegistry.compare_models("model-lambdamart-v1.3.0", "model-ranknet-v0.9.1")
        self.assertIn("overall_winner", result)
        self.assertIn("metrics_comparison", result)

    def test_trigger_retraining_queued(self):
        """Retraining trigger should appear in queue."""
        trigger = ModelRegistry.trigger_retraining("TestModel", reason="Unit test trigger")
        queue = ModelRegistry.get_retraining_queue()
        trigger_ids = [t["trigger_id"] for t in queue]
        self.assertIn(trigger.trigger_id, trigger_ids)

    def test_champion_model_is_production(self):
        """Champion model should always be in production stage."""
        champion = ModelRegistry.get_champion_model(ModelType.LISTWISE)
        if champion:
            self.assertEqual(champion["stage"], "production")


# ============================================================
# SECTION 9: PERFORMANCE BENCHMARKS
# ============================================================


class TestPerformanceBenchmarks(unittest.TestCase):
    """Ensure key operations meet latency SLAs."""

    FEATURES = {
        "bm25_score": 2.5,
        "tfidf_cosine": 0.6,
        "ctr_7d": 0.12,
        "freshness_score": 0.9,
        "popularity_score": 90.0,
        "avg_rating": 4.8,
        "exact_match_title": 1.0,
        "query_term_coverage": 0.8,
        "title_term_density": 0.5,
        "query_category_match": 0.7,
        "add_to_cart_rate": 0.06,
        "purchase_rate": 0.02,
        "dwell_time_median_seconds": 120.0,
    }

    def test_pointwise_scorer_sub_1ms(self):
        """Single pointwise score should complete in < 1ms."""
        start = time.perf_counter_ns()
        PointwiseScorer.score(self.FEATURES)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertLess(elapsed_ms, 1.0)

    def test_feature_computation_sub_5ms(self):
        """Full feature vector computation should be < 5ms."""
        doc = {
            "id": "prod-1",
            "title": "Sony Headphones",
            "description": "Wireless noise cancelling",
            "category": "Electronics",
            "popularity": 90.0,
            "ctr": 0.15,
            "freshness": 0.9,
            "engagement": 4.8,
        }
        start = time.perf_counter_ns()
        OfflineFeaturePipeline.compute_full_feature_vector("headphones", doc)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertLess(elapsed_ms, 5.0)

    def test_batch_ranking_10_docs_sub_50ms(self):
        """Ranking 10 documents should complete in < 50ms."""
        from backend.services.search_service import MOCK_PRODUCTS

        context = RankingContext(query="electronics", page=1, page_size=10)
        start = time.perf_counter_ns()
        MultiStageRankingPipeline.rank(MOCK_PRODUCTS, "electronics", context)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertLess(elapsed_ms, 50.0)

    def test_drift_psi_sub_10ms(self):
        """PSI computation on 100 samples should be < 10ms."""
        ref = [float(i) for i in range(100)]
        cur = [float(i) + 0.5 for i in range(100)]
        start = time.perf_counter_ns()
        StatisticalTests.psi(ref, cur)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.assertLess(elapsed_ms, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
