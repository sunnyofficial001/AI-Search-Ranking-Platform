# ML Architecture — AI Search & Recommendation Platform v2.0

## Overview

The platform implements a **multi-stage Learning-to-Rank (LTR) pipeline** built on three ranking paradigms:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│          Stage 1: Candidate Retrieval       │
│  BM25 (lexical) + TF-IDF Cosine (semantic)  │
│  → Top-100 candidate documents              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│          Stage 2: Multi-Model Ranking       │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐│
│  │XGBoost      │  │RankNet   │  │LambdaMART││
│  │Pointwise    │  │Pairwise  │  │Listwise  ││
│  │(MSE on 0-4) │  │(CE pairs)│  │(NDCG@10) ││
│  └─────────────┘  └──────────┘  └──────────┘│
│         └────────────┬────────────┘          │
│                 Stacked Ensemble              │
│     0.25×XGB + 0.30×RankNet + 0.45×LambdaMART│
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│          Stage 3: Re-Ranking                │
│  Optional personalization (user vec dot-product)│
│  + MMR diversity optimization               │
└─────────────────────────────────────────────┘
    │
    ▼
Final Ranked List → API Response
```

---

## Component Architecture

### Data Layer
```
backend/data/
├── mslr_loader.py        # SVMLight parser, query group builder, split loader
└── dataset_validator.py  # NaN/Inf detection, label range, leakage checks
```

### ML Training Layer
```
backend/training/
├── train_pipeline.py     # TrainingPipeline: XGBoost, RankNet, LambdaMART trainers
├── hpo.py                # Optuna HPO for XGBoost and LambdaMART
└── run_training.py       # Orchestration: validate → load → HPO → train → eval → register
```

### Evaluation Layer
```
backend/evaluation/
├── ranking_metrics.py    # NDCG@k, MAP, MRR, P@k, R@k (query-averaged)
└── evaluate.py           # Evaluator class used by API endpoints
```

### Inference Layer
```
backend/ml/
└── inference.py          # InferencePipeline singleton: loads all 3 models at startup
```

### Explainability Layer
```
backend/explainability/
└── explain.py            # ShapExplainer: SHAP TreeExplainer + analytical attribution API
```

### Model Registry
```
backend/model_registry/
└── artifact_registry.py  # SHA-256 integrity, metadata persistence, champion tracking
models/
├── pointwise_xgboost.json
├── pairwise_ranknet.pt
├── listwise_lambdamart.txt
├── best_params.json
└── registry.json         # Version metadata, metrics, checksums
```

---

## Feature Pipeline

MSLR-WEB10K provides 136 pre-computed features per query-document pair:

| Category | Features | Count |
|----------|----------|-------|
| TF signals (body, anchor, title, url, whole_doc) | 1–45 | 45 |
| Covered query term ratio/count | 46–55 | 10 |
| Stream length | 56–60 | 5 |
| IDF signals (sum, min, max, mean, variance) | 61–85 | 25 |
| TF-IDF composites | 86–105 | 20 |
| Boolean query model | 106–110 | 5 |
| Language Model (JM, Linear, Dirichlet) | 111–125 | 15 |
| BM25 per field | 126–130 | 5 |
| LMIR-ABS per field | 131–135 | 5 |
| SiteMap quality | 136 | 1 |

**Total: 136 features** — no normalization applied (tree models are scale-invariant).

---

## Recommendation Engine

The recommendation system operates independently of the LTR pipeline:

```
backend/services/
└── advanced_recommend.py
    ├── EmbeddingEngine       # Item embeddings (cosine similarity)
    ├── SessionRecommender    # Session-based (weighted recent history)
    ├── TwoTowerRecommender   # Dual-encoder (user × item embedding)
    ├── Item2VecRecommender   # Skip-gram style item co-occurrence
    ├── CollaborativeFilter   # SGD Matrix Factorization
    └── DiversityOptimizer    # MMR re-ranking for ILD
```

---

## Observability Stack

| Component | Implementation |
|-----------|---------------|
| Metrics | Prometheus-compatible `MetricsRegistry` (in-process) |
| Structured Logs | JSON structured logging via `StructuredLogger` |
| Rate Limiting | Token Bucket algorithm per client IP |
| SLA Tracking | P99 latency breach detection vs. configured targets |
| Drift Detection | PSI, KS-test, JSD, Page-Hinkley per feature stream |
| A/B Testing | Welch's t-test, Z-test, Cohen's d, min sample size calculator |

---

## Reproducibility

All training is deterministic given fixed seeds:
- `random_state=42` (XGBoost, scikit-learn)
- `seed=42` (LightGBM)
- `torch.manual_seed(42)` (RankNet)
- `numpy.random.seed(42)` (data subsetting)

Re-running `python -m backend.training.run_training` with the same data produces metrics within ±0.001 NDCG@10.
