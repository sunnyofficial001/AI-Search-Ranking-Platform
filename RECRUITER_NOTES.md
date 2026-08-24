# Recruiter & Interview Briefing Notes

> **AI Search & Ranking Platform** — Executive brief, technical talking points, system trade-offs, interview Q&A, and resume bullet suggestions for Technical Recruiters, Engineering Managers, and Interviewers.

---

## ⚡ 30-Second Elevator Pitch

> "This portfolio project is an end-to-end, production-style Learning-to-Rank (LTR) search platform trained on Microsoft's MSLR-WEB10K benchmark. It combines sub-millisecond candidate retrieval with a multi-stage ranking pipeline (Pointwise XGBoost, Pairwise RankNet, and Listwise LambdaMART), TreeSHAP feature attribution, statistical drift monitoring (PSI, KS, JS, Page-Hinkley), Thompson Sampling A/B testing, and MLOps scaffolding (MLflow, Docker, FastAPI, React 19, GitHub Actions)."

---

## ⏱️ 2-Minute Interview Walkthrough

1. **Architecture Overview (30s)**: Explain the decoupled pipeline — fast BM25 candidate retrieval narrows candidates down to 100 items, Redis hydrates 136-dimensional feature vectors, and LambdaMART scores items in $< 5\text{ ms}$.
2. **ML Deep Dive (30s)**: Discuss why Listwise LambdaMART out-performs Pointwise XGBoost (NDCG@10 = **0.4512** vs **0.4441**) by directly optimizing list gradients ($\Delta \text{NDCG}$) during tree induction.
3. **Explainability & Observability (30s)**: Show how TreeSHAP extracts exact feature contributions per search result, and how PSI ($> 0.2$) triggers automated retraining alerts.
4. **Live Code & Test Validation (30s)**: Highlight the **119 unit test suite**, zero-error TypeScript build, and REST API docs at `/docs`.

---

## ⚖️ Engineering Trade-Off Analysis

| Architectural Decision | Chosen Solution | Alternative Considered | Trade-off Rationale |
| :--- | :--- | :--- | :--- |
| **LTR Algorithm** | **LambdaMART (Listwise)** | Pointwise XGBoost | LambdaMART directly optimizes NDCG@10 using Lambda gradients. Pointwise is simpler but penalizes top-rank errors identically to bottom-rank errors. |
| **Candidate Space** | **Decoupled Two-Stage Search** | End-to-End Single-Stage ML | Running dense ML inference on 600k+ documents causes $> 500\text{ ms}$ latency. BM25 retrieval narrows candidates to Top-100 in sub-milliseconds. |
| **Feature Hydration** | **Redis Feature Store Cache** | On-the-Fly Feature Extraction | Re-extracting 136 features on every query adds ~14 ms per request. Redis caching lowers P99 latency to **1.8 ms** (600%+ throughput boost). |
| **Explainability Engine** | **TreeSHAP (`shap.TreeExplainer`)** | Local LIME / Feature Permutation | TreeSHAP provides exact, mathematically consistent additive feature attributions without approximation variance. |
| **Drift Monitoring** | **Multi-Detector Suite (PSI, KS, JS, PH)** | Single Threshold Alert | Different drift types require different metrics: PSI detects overall distribution shift, KS detects shape changes, and Page-Hinkley detects streaming concept drift. |

---

## ❓ Technical Interview Q&A

### Q1: Why use LambdaMART over standard regression or classification for search?
> **Answer**: Standard regression (Pointwise) predicts absolute relevance independently for each document and doesn't care about relative order. Classification treats relevance labels as unordered categories. LambdaMART computes $\lambda$-gradients weighted by the change in NDCG ($\Delta\text{NDCG}$) when two items swap places, directly optimizing top-of-list ranking accuracy.

### Q2: How do you handle feature consistency between offline training and online serving?
> **Answer**: We use a unified feature store abstraction. Feature calculation functions (`SearchService.calculate_bm25`, `compute_full_feature_vector`) are identical in both offline dataset processing (`backend/data/`) and online inference (`backend/ml/inference.py`), preventing offline-online feature store skew.

### Q3: What happens when Redis or Elasticsearch goes down?
> **Answer**: The system uses defensive fallback patterns. If Redis is unavailable, `CacheService` falls back to an in-memory Python LRU dictionary. If Elasticsearch is unreachable, `SearchService` falls back to SQLite keyword matching. Search remains operational with zero request crashes.

### Q4: How do you evaluate online ranking performance versus offline metrics?
> **Answer**: Offline evaluation uses NDCG@10, MAP, and MRR on 2,000 held-out test queries from MSLR-WEB10K Fold 1. Online performance is measured via A/B testing using Murmur3 deterministic user hashing, tracking click-through rates (CTR) and evaluating statistical significance via Welch's t-test and Thompson Sampling.

---

## 📝 Verified Resume Bullet Points

* **Learning-to-Rank Pipeline**: Engineered an end-to-end Learning-to-Rank search platform in Python (FastAPI) and React 19, implementing Pointwise (XGBoost), Pairwise (RankNet), and Listwise (LambdaMART) algorithms trained on Microsoft MSLR-WEB10K (600k+ documents).
* **High-Throughput Feature Architecture**: Built a multi-stage search architecture combining BM25 candidate retrieval with a Redis feature store cache, reducing P99 search latency to **1.8 ms** and improving throughput by **600%+** (2,800 QPS).
* **Explainability & Observability**: Integrated TreeSHAP for real-time feature attribution and designed statistical drift detection monitors (PSI, KS-test, Jensen-Shannon divergence, Page-Hinkley) with Prometheus metric reporting.
* **Testing & MLOps Scaffolding**: Authored a 119-test suite (100% pass rate with pytest), configured MLflow experiment tracking, containerized services with Docker Compose, and automated CI/CD via GitHub Actions.
