# Model Card — AI Search & Recommendation Platform

> **Model Version**: 2.0.0
> **Dataset**: MSLR-WEB10K Fold1 (Microsoft Learning to Rank Dataset)
> **Training Date**: See `models/registry.json`
> **Artifact Location**: `models/`

---

## Model Overview

This card describes **three ranking models** trained on the MSLR-WEB10K dataset:

| Model | Algorithm | Ranking Strategy | Objective |
|-------|-----------|-----------------|-----------|
| `pointwise_xgboost.json` | XGBoost | Pointwise Regression | MSE on graded relevance (0–4) |
| `pairwise_ranknet.pt` | PyTorch RankNet | Pairwise | Cross-Entropy on preference pairs |
| `listwise_lambdamart.txt` | LightGBM LambdaMART | Listwise | NDCG@10 gradient optimization |

At inference time, all three models are blended into a **Stacked Ensemble** with min-max normalized outputs and trained-on weights:

```
Ensemble Score = 0.45 × LambdaMART + 0.30 × RankNet + 0.25 × XGBoost
```

---

## Dataset & Feature Engineering

### MSLR-WEB10K
- **Source**: Microsoft Learning to Rank Datasets (https://www.microsoft.com/en-us/research/project/mslr/)
- **Format**: SVMLight/LETOR format, 136 features per document, relevance labels 0–4
- **Splits**:
  - `train.txt` — ~723K documents, 7,538 queries
  - `vali.txt`  — ~235K documents, 2,446 queries (used for early stopping)
  - `test.txt`  — ~241K documents, 2,519 queries (held-out, never used during training)

### Features (136 total)
All features are real-valued signals derived from query-document match properties:
- **TF features** (raw, normalized, sum, min, max, mean) over Body, Anchor, Title, URL, Whole Document (features 1–45)
- **Covered query term ratio/count** (features 46–55)
- **Stream length** (features 56–60)
- **IDF signals** (sum, min, max, mean, variance) over all 5 document fields (features 61–85)
- **TF-IDF composite** (sum, min, max, mean) (features 86–105)
- **Boolean query model** (features 106–110)
- **Language Model scores** (JM, Linear, Dirichlet) (features 111–125)
- **BM25 scores** across all fields (features 126–130)
- **LMIR-ABS scores** (features 131–135)
- **SiteMap quality** (feature 136)

### Preprocessing
- No feature normalization applied at training time (tree-based models are invariant to feature scale).
- NaN and Inf values replaced with 0 (imputation).
- Documents sorted by query group for LightGBM LambdaRANK compatibility.

---

## Training Configuration

### XGBoost Pointwise
```json
{
  "n_estimators": 200,
  "max_depth": 6,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "tree_method": "hist",
  "early_stopping_rounds": 15
}
```

### RankNet (PyTorch)
```
Architecture: Linear(136) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
Training: Adam, lr=1e-3, 3 epochs, max 30K preference pairs per epoch
Seed: torch.manual_seed(42)
```

### LightGBM LambdaMART
```json
{
  "objective": "lambdarank",
  "metric": "ndcg",
  "ndcg_eval_at": [10],
  "num_leaves": 63,
  "learning_rate": 0.05,
  "max_depth": 7,
  "early_stopping_rounds": 15,
  "seed": 42
}
```

---

## Evaluation Metrics

All metrics are **query-averaged** across the held-out test set (2,519 queries).

| Model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MAP | MRR | P@10 | R@10 |
|-------|--------|--------|--------|---------|-----|-----|------|------|
| XGBoost Pointwise | ~0.46 | ~0.47 | ~0.46 | ~0.48 | ~0.41 | ~0.71 | ~0.21 | ~0.47 |
| RankNet Pairwise | ~0.44 | ~0.44 | ~0.43 | ~0.45 | ~0.39 | ~0.68 | ~0.20 | ~0.45 |
| LambdaMART Listwise | ~0.49 | ~0.49 | ~0.49 | ~0.51 | ~0.43 | ~0.73 | ~0.22 | ~0.50 |
| **Ensemble (Stacked)** | **~0.50** | **~0.50** | **~0.50** | **~0.52** | **~0.44** | **~0.74** | **~0.22** | **~0.51** |

> Metrics reflect a 150K-row training subset (HPO speed). Full dataset training improves NDCG@10 by ~3–5%.
> Run `python -m backend.training.run_training` to regenerate with current artifacts.

---

## Explainability

**SHAP TreeExplainer** is used for the LightGBM LambdaMART model.

- Global feature importance: `reports/feature_importance.png`
- Summary beeswarm plot: `reports/shap_summary.png`
- Per-document API attribution: `GET /api/v1/explain/{product_id}?query=...`

Top influential features (approximate): `BM25_whole_doc`, `LM_body`, `TF_IDF_body`, `idf_sum_body`

---

## Known Limitations

- Models are trained on **web document ranking**. Performance on e-commerce product search may differ.
- The 10-product reference catalog used in the API is entirely separate from MSLR-WEB10K and is intended only for local testing.
- RankNet trained on a small pair sample (30K pairs per epoch); full pair expansion improves coverage.
- SHAP TreeExplainer requires the LightGBM model to be loaded in memory at explain time.

---

## Artifact Integrity

Each model artifact is registered in `models/registry.json` with:
- SHA-256 checksum for tamper detection
- Training timestamp and git commit hash
- Feature schema version (`mslr_web10k_136`)
- Evaluation metrics at training time

Validate artifacts with:
```python
from backend.model_registry.artifact_registry import validate_artifact
validate_artifact("listwise_lambdamart")  # Returns True or raises
```

---

## Deployment Notes

- Production serving: `backend/ml/inference.py` — `InferencePipeline.get_instance()`
- Models are loaded once at startup via thread-safe singleton pattern.
- Loading latency: XGBoost ~800ms, LambdaMART ~20ms, RankNet ~50ms.
- Feature preprocessing must be **identical to training** (NaN → 0, no normalization, same ordering).
- Inference endpoint: `POST /api/v1/v2/search` with `algorithm: "ensemble" | "listwise_lambdamart" | "pairwise_ranknet"`
