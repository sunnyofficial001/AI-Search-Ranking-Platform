# Machine Learning System Design Guide

> **AI Search & Ranking Platform** — Detailed design documentation covering Learning-to-Rank (LTR) algorithms, feature engineering, model objectives, SHAP explainability, statistical drift detection, and online A/B experimentation.

---

## 1. Learning-to-Rank (LTR) Paradigms & Algorithms

This platform implements all three foundational Learning-to-Rank paradigms to provide a comprehensive comparison of ranking quality versus computational complexity.

```
+-----------------------------------------------------------------------------------+
|                            Learning-to-Rank Paradigms                              |
+--------------------------+--------------------------+-----------------------------+
|    Pointwise (XGBoost)   |    Pairwise (RankNet)    |    Listwise (LambdaMART)   |
+--------------------------+--------------------------+-----------------------------+
| Loss: Mean Squared Error | Loss: Pair Cross-Entropy | Loss: Lambda-adjusted NDCG  |
| Input: Single Query-Doc  | Input: Document Pair     | Input: Full Query List      |
| Ignores item context     | Optimizes preference     | Directly optimizes NDCG@10  |
+--------------------------+--------------------------+-----------------------------+
```

### 1.1 Pointwise Ranking (XGBoost Regressor)
* **Objective**: Predicts absolute numerical relevance label $y_i \in [0, 4]$ for each query-document pair independently.
* **Loss Function**: Mean Squared Error (MSE)
  $$\mathcal{L}_{\text{pointwise}} = \frac{1}{N} \sum_{i=1}^N \left( \hat{y}_i - y_i \right)^2$$
* **Pros**: Simple, fast to train, compatible with standard regression algorithms.
* **Cons**: Fails to capture relative ordering between documents for the same query. A predicted score difference between ranks 1 and 2 is penalized identically to a difference between ranks 99 and 100.

### 1.2 Pairwise Ranking (PyTorch RankNet)
* **Objective**: Models the probability that document $i$ is more relevant than document $j$ for a given query $q$.
* **Target Probability**: $P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$
* **Loss Function**: Binary Cross-Entropy over document pairs:
  $$\mathcal{L}_{\text{pairwise}} = -\sum_{(i,j) \in C} \left( \bar{P}_{ij} \log P_{ij} + (1 - \bar{P}_{ij}) \log (1 - P_{ij}) \right)$$
* **Pros**: Focuses on relative order rather than arbitrary absolute scores.
* **Cons**: Quadratic pair complexity $\mathcal{O}(K^2)$ per query; treats all rank swaps equally regardless of position in the result list.

### 1.3 Listwise Ranking (LambdaMART / LightGBM)
* **Objective**: Directly optimizes list-level Information Retrieval metrics (**NDCG@10**) by weighting gradient updates by the $\Delta \text{NDCG}$ caused by swapping items.
* **Lambda Gradient Formulation**:
  $$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta \text{NDCG}_{ij}|$$
* **Pros**: Directly optimizes the business objective (NDCG@10); prioritizes top-of-list precision.
* **Cons**: Requires full query group context during tree building.

---

## 2. Feature Store & Feature Engineering Schema

The platform ingests the 136 dense features from the **Microsoft MSLR-WEB10K** benchmark, categorized into 5 functional feature domains:

| Feature Category | Description | Primary Features | Example Signals |
| :--- | :--- | :--- | :--- |
| **Text Matching** | Classical IR lexical similarity | 1 - 30 | BM25, TF-IDF, Term Frequency, Document Frequency |
| **Language Models** | Probabilistic & smoothing models | 31 - 60 | LMIR.ABS, LMIR.DIR, LMIR.JM across body/title/anchor |
| **Web Graph / PageRank** | Authority & link structure | 61 - 90 | PageRank, InLink count, OutLink count, SiteRank |
| **Document Quality** | Structural content signals | 91 - 110 | Title length, Body length, URL depth, Header count |
| **Engagement & Recency** | User click & freshness feedback | 111 - 136 | Historical CTR, Popularity score, Freshness decay |

---

## 3. Empirical Evaluation Metrics (MSLR-WEB10K Test Set)

Evaluated across **2,000 test set queries** (Fold 1 test set):

$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{r_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}, \quad \text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$

### Reproducible Benchmark Results Table

| Model Architecture | Strategy | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 (Primary) | MAP | MRR | P@10 | R@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Pointwise | 0.4238 | 0.4161 | 0.4230 | **0.4441** | 0.6099 | 0.8269 | 0.6628 | 0.1863 |
| **RankNet** | Pairwise | 0.1595 | 0.1743 | 0.1880 | **0.2107** | 0.4682 | 0.5996 | 0.4445 | 0.1135 |
| **LambdaMART** | Listwise | **0.4369** | **0.4257** | **0.4327** | **0.4512** | **0.6061** | **0.8302** | 0.6578 | 0.1848 |

---

## 4. Explainable AI (SHAP Integration)

To provide transparency for ranking decisions, the platform integrates **TreeSHAP** (`shap.TreeExplainer`).

* **Additive Feature Attribution**:
  $$f(x) = \phi_0 + \sum_{j=1}^M \phi_j$$
  where $\phi_0$ is the expected baseline ranking score and $\phi_j$ is the attribution value of feature $j$.
* **Production Explainability Output**: The `/api/explain` endpoint computes per-document SHAP force values, identifying exactly which text match or engagement signals boosted or lowered a product's rank.

---

## 5. Statistical Drift Detection Pipeline

To detect feature distribution decay and concept drift in query traffic, four complementary statistical detectors are implemented in `backend/monitoring/`:

1. **Population Stability Index (PSI)**:
   $$\text{PSI} = \sum \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$
   *Threshold*: $\text{PSI} > 0.2$ indicates significant feature drift requiring model retraining.
2. **Kolmogorov-Smirnov (KS) Test**: Non-parametric test comparing cumulative distributions of baseline vs. live query features.
3. **Jensen-Shannon Divergence (JSD)**: Symmetric, bounded measure ($[0, 1]$) of probability distribution distance.
4. **Page-Hinkley Test**: Sequential analysis detector designed for real-time concept drift detection in streaming query volumes.

---

## 6. Online A/B Testing & Bandit Framework

The platform includes a dedicated experimentation subsystem (`backend/ab_testing/`):

* **Deterministic Variant Assignment**: Murmur3 hash mapping of `user_id` to experiment variants ensures consistent user experience across sessions.
* **Thompson Sampling Multi-Armed Bandit**: Dynamically routes live traffic towards higher-performing rankers while balancing exploration:
  $$\theta_k \sim \text{Beta}(\alpha_k + 1, \beta_k + 1)$$
* **Statistical Significance Engine**: Performs Welch's t-test and computes $p$-values to evaluate online click-through rate (CTR) and conversion differences between ranking models.
