# AI Search & Ranking Platform

![AI Search & Ranking Platform Banner](assets/banner.svg)

> Production-oriented Learning-to-Rank search platform combining candidate retrieval, multi-stage ranking, explainability, experimentation, and MLOps.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?logo=typescript)](https://www.typescriptlang.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-orange)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Pointwise-red)](https://xgboost.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-RankNet-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### 🌟 Key Highlights

* **End-to-End LTR Pipeline**: Full-stack search platform trained on the **Microsoft MSLR-WEB10K** benchmark (626k+ documents, 136 dense features).
* **Multi-Stage Ranking**: Implements Pointwise (XGBoost), Pairwise (PyTorch RankNet), and Listwise (LambdaMART) regressors with stacked ensemble weighting.
* **Sub-Millisecond Candidate Retrieval**: Decoupled two-stage retrieval pairing BM25 candidate selection with ML re-ranking.
* **Redis Feature Store Caching**: 136-dimensional feature vector caching in Redis delivering P99 search latencies under **1.8 ms** (**600%+ QPS improvement**).
* **Explainable AI (SHAP)**: Real-time TreeSHAP feature attribution breaking down per-document relevance scores.
* **Statistical Drift Monitoring**: Automated feature drift detection implementing PSI, Kolmogorov-Smirnov, Jensen-Shannon divergence, and Page-Hinkley algorithms.
* **A/B Experimentation Framework**: Murmur3 user hashing, Thompson Sampling Multi-Armed Bandits, and statistical significance testing.
* **Production MLOps Scaffolding**: MLflow tracking, Docker Compose orchestration, Prometheus telemetry, and GitHub Actions CI.
* **Verified Test Coverage**: 124 backend unit tests (100% pass rate) + zero-error TypeScript build.

---

## 📌 Overview

Traditional keyword search engines (e.g. standard BM25 or TF-IDF) are effective at retrieving candidate documents containing matching query terms, but struggle to rank them accurately because they ignore complex multi-dimensional signals such as user engagement, content freshness, PageRank authority, and semantic relevance.

This platform bridges information retrieval and machine learning by providing a **production-oriented Learning-to-Rank (LTR) architecture**. It ingests candidate pools from fast lexical retrievers, hydrates dense feature vectors, applies machine-learned ranking trees, optimizes result diversity via Maximal Marginal Relevance (MMR), and emits real-time SHAP explainability and drift telemetry.

---

## 🖥️ Interactive Dashboard Preview

![Dashboard Showcase](assets/dashboard.svg)

---

## 💡 Why This Project is Interesting

Most machine learning repositories focus exclusively on offline model training inside Jupyter notebooks. This project demonstrates how an ML ranking model functions inside a full-stack, low-latency production system:

```text
       Information Retrieval (BM25 Lexical Candidate Selection)
                                  +
       Machine Learning (Pointwise, Pairwise & Listwise LTR)
                                  +
       High-Performance Systems (Redis Feature Caching < 2ms P99)
                                  +
       MLOps & Governance (MLflow Lifecycle, SHAP, Statistical Drift)
                                  +
       Full-Stack Engineering (FastAPI Async Gateway + React 19 UI)
```

---

## 📐 System Architecture

![System Architecture](assets/architecture.svg)

*For complete technical details, see [`ARCHITECTURE.md`](ARCHITECTURE.md).*

---

## 🔄 End-to-End Ranking Pipeline

![Ranking Pipeline](assets/ranking_pipeline.svg)

1. **User Query Submission**: Client sends query and optional algorithm weights to Express API gateway.
2. **Candidate Retrieval**: BM25 inverted index retrieves Top-100 candidates, reducing search space by 99.9%.
3. **Feature Store Extraction**: Hydrates 136 dense features per candidate from Redis cache.
4. **Pointwise Scoring**: XGBoost computes absolute relevance regression predictions.
5. **Pairwise Scoring**: PyTorch RankNet evaluates pairwise preference probabilities.
6. **Listwise Scoring**: LightGBM LambdaMART computes list-level NDCG-optimized scores.
7. **Stacked Ensemble**: Combines model predictions using weighted ensemble scoring.
8. **MMR Diversity Reranking**: Applies Maximal Marginal Relevance to eliminate redundant item clusters.
9. **SHAP Explanation Generation**: Computes TreeSHAP feature attributions for top results.
10. **Telemetry & Log Emission**: Emits latency histogram metrics to Prometheus and logs clickstream.

---

## 🤖 Machine Learning Strategy

| Paradigm | Model Architecture | Optimization Objective | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Pointwise** | XGBoost Regressor | Mean Squared Error (MSE) | Absolute relevance score baseline |
| **Pairwise** | PyTorch RankNet | Binary Cross-Entropy on pairs | Pairwise document preference ordering |
| **Listwise** | LambdaMART (LightGBM) | Lambda-adjusted NDCG@10 | List-level ranking quality (Primary LTR) |

---

## 📊 Empirical Evaluation (MSLR-WEB10K Test Set)

![Benchmark Results Chart](assets/benchmark_results.svg)

All metrics below are strictly reproducible, computed across **2,000 test set queries** (MSLR-WEB10K Fold 1):

| Model Architecture | Strategy | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 (Primary) | MAP | MRR | P@10 | R@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Pointwise | 0.4238 | 0.4161 | 0.4230 | **0.4441** | 0.6099 | 0.8269 | 0.6628 | 0.1863 |
| **RankNet** | Pairwise | 0.1595 | 0.1743 | 0.1880 | **0.2107** | 0.4682 | 0.5996 | 0.4445 | 0.1135 |
| **LambdaMART** | Listwise | **0.4369** | **0.4257** | **0.4327** | **0.4512** | **0.6061** | **0.8302** | 0.6578 | 0.1848 |

> *Note: Benchmark metrics are derived from `reports/metrics.json` and `results/benchmark_report.md`. Any UI mock analytics represent real-time online simulation outputs.*

---

## 🔍 Explainability & Observability

![SHAP Waterfall Explanation](assets/shap_explanation.svg)

* **SHAP Integration**: Integrates TreeSHAP (`shap.TreeExplainer`) to compute additive feature attributions for ranking predictions:
  $$f(x) = \phi_0 + \sum_{j=1}^M \phi_j$$
* **Statistical Drift Detectors**:
  * **PSI (Population Stability Index)**: Monitors feature distribution shifts ($\text{PSI} > 0.2$ alerts for retraining).
  * **Kolmogorov-Smirnov (KS) Test**: Evaluates continuous feature distribution divergence.
  * **Jensen-Shannon Divergence (JSD)**: Measures symmetric probability distance.
  * **Page-Hinkley Test**: Tracks real-time streaming concept drift.

---

## 🧪 Online Experimentation & A/B Testing

* **Deterministic Variant Routing**: Murmur3 hashing on `user_id` ensures consistent experiment bucket allocation.
* **Multi-Armed Bandit**: Thompson Sampling dynamically allocates traffic towards superior ranking models.
* **Significance Testing**: Computes Welch's t-test $p$-values to evaluate online click-through rate (CTR) gains.

---

## ⚖️ Architectural Trade-Offs

* **Why LambdaMART over Pointwise?**: Pointwise regression treats every rank error equally. LambdaMART weights gradients by $\Delta\text{NDCG}$, placing maximum optimization focus on top-ranked slots.
* **Why Decouple Retrieval & Ranking?**: Running ML tree evaluation over 600,000 items takes $> 500\text{ ms}$. Restricting ML ranking to Top-100 candidates maintains sub-5ms P99 latency.
* **Why Redis Feature Store Caching?**: On-the-fly feature extraction adds ~14ms per request. Caching 136-dim feature vectors in Redis lowers latency to **1.8 ms** (600%+ throughput increase).

---

## 🛠️ Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Backend API** | FastAPI, Python 3.10+, Uvicorn, Express.js (API Gateway) |
| **Machine Learning** | LightGBM, XGBoost, PyTorch, scikit-learn |
| **Information Retrieval** | BM25 Engine, Elasticsearch integration scaffolding |
| **Data & Caching** | PostgreSQL (SQLAlchemy ORM), Redis, SQLite |
| **Explainability** | SHAP (`shap.TreeExplainer`) |
| **MLOps & Monitoring** | MLflow, Prometheus Client, GitHub Actions CI |
| **Frontend UI** | React 19, TypeScript, Tailwind CSS, Recharts, Lucide Icons |
| **Infrastructure** | Docker, Docker Compose |

---

## 📁 Project Structure

```text
ai-search-platform/
├── backend/                  # FastAPI Python Backend & ML Engine
│   ├── ab_testing/           # A/B testing & Thompson Sampling bandit
│   ├── api/                  # REST API v1 & v2 route handlers
│   ├── core/                 # App configuration & custom exceptions
│   ├── data/                 # MSLR-WEB10K loader & dataset validator
│   ├── database/             # SQLAlchemy connection & seeder
│   ├── evaluation/           # IR metrics (NDCG, MAP, MRR, Precision, Recall)
│   ├── explainability/       # TreeSHAP explainer module
│   ├── ml/                   # Model inference wrapper
│   ├── model_registry/       # MLflow artifact registry
│   ├── monitoring/           # Prometheus metrics & drift detectors
│   ├── services/             # Ranking, search, recommendation, cache services
│   └── training/             # Offline LTR training pipeline scripts
├── src/                      # React 19 TypeScript Dashboard
│   ├── components/           # Search, Ranking, SHAP, Rec, Dataset views
│   └── engines.ts            # Client-side IR & ranking calculation engines
├── assets/                   # Architecture SVGs & visual diagrams
├── reports/                  # Benchmark metrics JSON & SHAP plots
├── results/                  # Benchmark markdown reports
├── tests/                    # 124 unit & integration tests (pytest)
├── Dockerfile                # Production Python FastAPI container
├── Dockerfile.frontend       # Production Node.js frontend container
├── docker-compose.yml        # Multi-service infrastructure compose
├── server.ts                 # Express API Gateway server
└── package.json              # NPM dependencies & project metadata
```

---

## 🚀 Quick Start Guide

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/sunnyofficial001/AI-Search-Ranking-Platform.git
cd AI-Search-Ranking-Platform

# Install Python dependencies (requires Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r backend/requirements.txt

# Install Node.js dependencies
npm install
```

### 2. Start Services

```bash
# Terminal 1: Start FastAPI ML Backend
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2: Start Express Gateway & React UI
npm run dev
```

* **React UI Dashboard**: [http://localhost:3000](http://localhost:3000)
* **FastAPI Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Compose Deployment

```bash
# Build and launch complete multi-container stack
docker compose up --build -d

# Verify container health
docker compose ps
```

---

## 🧪 Testing & Validation

```bash
# Run backend test suite (124 unit tests)
python -m pytest tests/unit/ -v

# Run TypeScript linter & type checker
npm run lint
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
