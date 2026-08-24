# Recruiter 3-Minute Interactive Demo Guide

> **AI Search & Ranking Platform** — Step-by-step walkthrough to run, evaluate, and inspect the platform locally in under 3 minutes.

---

## 🚀 Fast Track (Quickest 1-Command Startup)

```bash
# 1. Clone repository
git clone https://github.com/sunnyofficial001/AI-Search-Ranking-Platform.git
cd AI-Search-Ranking-Platform

# 2. Run initial check & start frontend + backend gateway
npm run dev
```

* **Frontend Dashboard**: Open [http://localhost:3000](http://localhost:3000)
* **Backend FastAPI Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎯 3-Minute Recruiter Demo Flow

Follow these 5 simple steps to explore the entire end-to-end Learning-to-Rank system:

```text
[Step 1] Search Dashboard  ──► [Step 2] LTR Model Hub ──► [Step 3] SHAP Explainability
                                                                   │
[Step 5] A/B Experiments  ◄── [Step 4] Dataset Features ──────────┘
```

### Step 1: Execute Hybrid Search & Compare Rankers
1. Navigate to **Search Retrieval** ([http://localhost:3000](http://localhost:3000)).
2. Type a query like `"headphones"` or `"running shoes"`.
3. Toggle between algorithm options:
   * **BM25 Baseline**: Keyword frequency ranking.
   * **Pointwise (XGBoost)**: Score regressor.
   * **Listwise (LambdaMART)**: NDCG-optimized tree ranker.
4. Observe how the final document ordering, relevance scores, and latency metrics adapt instantly.

### Step 2: Inspect Machine Learning Training Control
1. Click **Learning To Rank** on the sidebar.
2. Select an algorithm (LambdaMART, XGBoost, or RankNet).
3. Adjust hyper-parameters (e.g. Learning Rate: `0.05`, Trees: `50`).
4. Click **Trigger Model Retraining Job** to simulate MLflow experiment logging.

### Step 3: Inspect SHAP Feature Attribution
1. Click **Explainable AI** on the sidebar.
2. Select any product from the catalog.
3. Review the TreeSHAP feature waterfall chart showing how text matching (BM25), popularity, CTR, and freshness signals contribute positively or negatively to the final ranking.

### Step 4: Explore the Feature Store & ETL Pipeline
1. Click **MSLR Ingestion & Feature Store**.
2. Click **Re-run Backend ETL**.
3. Watch the live execution progress log as 136-dimensional feature vectors are processed and hydrated into the feature store.

### Step 5: Check MLOps & Experiment Tracking
1. Click **MLflow & Registry**.
2. Review active model versions, parameters, and historical training runs.

---

## 🐳 Full Production Stack Demo (Docker Compose)

If you prefer to run the complete infrastructure with PostgreSQL, Redis, and FastAPI:

```bash
# Launch multi-container production stack
docker compose up --build -d

# Check status of container services
docker compose ps
```

* **Frontend**: `http://localhost:3000`
* **FastAPI Backend**: `http://localhost:8000`
* **API Documentation**: `http://localhost:8000/docs`

---

## 🧪 Verification & Health Check Commands

To verify that all services are operational:

```bash
# Express + FastAPI Health Check
curl http://localhost:3000/api/health

# Run complete pytest test suite (119 unit tests)
python -m pytest tests/unit/ -v
```
