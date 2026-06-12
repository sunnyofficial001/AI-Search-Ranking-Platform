import unittest

from backend.evaluation.evaluate import Evaluator


class TestRetrievalMetrics(unittest.TestCase):
    def test_dcg_calculation(self):
        # Relevance scores
        relevances = [4, 2, 0, 0]
        # DCG@2 = (2^4 - 1)/log2(2) + (2^2 - 1)/log2(3) = 15/1 + 3/1.58496 = 15 + 1.89278 = 16.89278
        computed_dcg = Evaluator.calculate_dcg(relevances, 2)
        self.assertAlmostEqual(computed_dcg, 16.89278, places=3)

    def test_ndcg_calculation(self):
        ranked = [4, 1, 2, 0]
        ideal = [4, 2, 1, 0]

        computed_ndcg = Evaluator.calculate_ndcg(ranked, ideal, 3)
        self.assertTrue(0.0 <= computed_ndcg <= 1.0)
        self.assertGreater(computed_ndcg, 0.5)

    def test_map_calculation(self):
        # Index pos: 0=relevant, 1=irrelevant, 2=relevant, 3=irrelevant
        # Precision@1 = 1/1 = 1.0
        # Precision@3 = 2/3 = 0.666
        # AP = (1.0 + 0.6666)/2 = 0.8333
        ranked = [4, 1, 3, 0]  # 4 and 3 have score >= 2 (relevant)
        computed_map = Evaluator.calculate_map(ranked, 2)
        self.assertAlmostEqual(computed_map, 0.8333, places=3)

    def test_mrr_calculation(self):
        # First relevant item is at index 1 (rank 2)
        # MRR = 1/2 = 0.5
        ranked = [0, 4, 2, 0]
        mrr = Evaluator.calculate_mrr(ranked)
        self.assertEqual(mrr, 0.5)

    def test_precision_recall_k(self):
        ranked = [3, 0, 5, 1, 2]  # relevant at 0, 2, 4 (3 items)
        prec_5 = Evaluator.calculate_precision_at_k(ranked, 5)
        rec_5 = Evaluator.calculate_recall_at_k(ranked, 3, 5)

        self.assertEqual(prec_5, 0.6)  # 3/5
        self.assertEqual(rec_5, 1.0)  # 3/3
