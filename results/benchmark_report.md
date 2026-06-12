# Performance Benchmark & MSLR-WEB10K Evaluation Report

*(Generated on 2026-06-07 09:15:30)*

This report presents performance metrics for the Pointwise, Pairwise, and Listwise rankers trained on the real **MSLR-WEB10K** dataset.

## 1. Machine Learning Ranking Quality Evaluation

| Model | Strategy | Target Function |
|-------|----------|-----------------|
| **XGBoost** | Pointwise | Regression MSE |
| **RankNet** | Pairwise | Cross-Entropy |
| **LambdaMART** | Listwise | NDCG@10 (Lambda gradients) |

### Results on Test Set

| Model                          | NDCG@1   | NDCG@3   | NDCG@5   | NDCG@10   | MAP      | MRR      | P@10     | R@10     |
| ------------------------------ | -------- | -------- | -------- | --------- | -------- | -------- | -------- | -------- |
| Pointwise (XGBoost)            | 0.4238   | 0.4161   | 0.4230   | 0.4441    | 0.6099   | 0.8269   | 0.6628   | 0.1863   |
| Pairwise (RankNet)             | 0.1595   | 0.1743   | 0.1880   | 0.2107    | 0.4682   | 0.5996   | 0.4445   | 0.1135   |
| Listwise (LambdaMART)          | 0.4369   | 0.4257   | 0.4327   | 0.4512    | 0.6061   | 0.8302   | 0.6578   | 0.1848   |

## 2. Dataset Statistics

- **Total Documents:** 626,780
- **Total Queries:** 5,217
- **Features per Document:** 136

