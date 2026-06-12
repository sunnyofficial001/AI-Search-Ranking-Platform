# Recruiter & Interview Notes

## Top Resume Bullets
- Implemented a full-stack Learning-to-Rank platform (React frontend + FastAPI backend) demonstrating retrieval, feature engineering, and multi-stage ranking pipelines.
- Built and evaluated multiple ranking models (Pointwise XGBoost, Pairwise RankNet, Listwise LambdaMART) with reproducible training pipelines and evaluation scripts.
- Integrated explainability using SHAP to produce per-query and global feature importance reports for ranked results.
- Added MLOps scaffolding: Dockerfiles, `docker-compose.yml`, MLflow hooks, Prometheus metrics endpoints, and GitHub Actions CI for automated testing and builds.
- Wrote a 70+ unit and integration test suite covering ranking metrics (NDCG, MAP, MRR), drift detection algorithms, and recommendation components.

## Top Interview Talking Points
- Explain the ranking pipeline: candidate retrieval → feature extraction → model scoring → reranking.
- Describe NDCG, MAP, and MRR: what they measure, when to prefer each, and how you compute them in `backend/evaluation`.
- Discuss LambdaMART: how lambda gradients approximate the change in NDCG and why listwise loss is effective for ranking.
- Talk about drift detection: PSI, KS-test, Jensen-Shannon divergence, and Page-Hinkley for sequential detection, and how thresholds trigger retraining.
- Explain SHAP for tree models and how per-query SHAP explanations help debug model decisions and surface feature bias.

## Likely Interview Questions & Suggested Short Answers
- Q: How do you handle cold-start for recommendations?
  A: Use hybrid approaches (content-based features + collaborative signals) and fall back to popularity-based ranking; store item metadata and compute content-similarity features for unseen users/items.
- Q: How do you evaluate ranking models offline vs online?
  A: Offline via NDCG/MAP/MRR on held-out queries; online via A/B experiments measuring click-through, dwell time, and downstream metrics; use consistent hashing for assignment and statistical tests for significance.
- Q: How to ensure reproducibility?
  A: Pin dependencies (`requirements.lock`), store model artifacts and parameters in MLflow, log random seeds and environment metadata, and provide demo scripts and small synthetic datasets for reviewers.

## LinkedIn Project Description (short)
Built a production-oriented Learning-to-Rank platform implementing retrieval, feature engineering, and multi-stage ranking with explainability and MLOps scaffolding. Includes reproducible training pipelines, evaluation metrics, and a React dashboard for visualization.

## Prepped Demo Walkthrough Script (2-3 minutes)
1. Start backend (`uvicorn backend.main:app --reload --port 8000`) and frontend (`npm run dev`).
2. Open dashboard at `http://localhost:3000` and show a sample search and the resulting rankings.
3. Trigger a small training run or show saved evaluation metrics in `reports/` and open a SHAP explanation for a sample query.
