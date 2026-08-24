"""
Orchestration Script for MSLR-WEB10K Production Training
=========================================================
Stages:
  1. Dataset validation (integrity checks, leakage detection)
  2. Data loading (train / vali / test splits)
  3. Hyperparameter optimization via Optuna
  4. Model training (XGBoost, RankNet, LambdaMART)
  5. Evaluation on held-out test set (NDCG, MAP, MRR, P@K, R@K)
  6. SHAP explainability reports
  7. MLflow run tracking
  8. Artifact registry metadata persistence
  9. Benchmark report generation
"""

import os
import json
import logging
from datetime import datetime

# Logger must be defined before any conditional imports that use it
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import xgboost as xgb
import lightgbm as lgb
import torch

from backend.data.mslr_loader import load_all_splits, get_query_groups
from backend.data.dataset_validator import validate_dataset
from backend.training.train_pipeline import TrainingPipeline, PyTorchRankNet
from backend.training.hpo import run_hpo
from backend.evaluation.ranking_metrics import compute_all_metrics, format_metrics_table
from backend.model_registry.artifact_registry import register_model
from backend.services.mlflow_service import MLflowService

try:
    from backend.explainability.explain import generate_shap_reports
except ImportError as e:
    logger.warning(f"Failed to import SHAP/sklearn: {e}")
    generate_shap_reports = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_benchmark_report(metrics_dict: dict, dataset_stats: dict):
    """Write results/benchmark_report.md"""
    report_path = os.path.join(RESULTS_DIR, "benchmark_report.md")
    
    table_str = format_metrics_table(metrics_dict)
    
    content = f"""# Performance Benchmark & MSLR-WEB10K Evaluation Report

*(Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")})*

This report presents performance metrics for the Pointwise, Pairwise, and Listwise rankers trained on the real **MSLR-WEB10K** dataset.

## 1. Machine Learning Ranking Quality Evaluation

| Model | Strategy | Target Function |
|-------|----------|-----------------|
| **XGBoost** | Pointwise | Regression MSE |
| **RankNet** | Pairwise | Cross-Entropy |
| **LambdaMART** | Listwise | NDCG@10 (Lambda gradients) |

### Results on Test Set

{table_str}

## 2. Dataset Statistics

- **Total Documents:** {dataset_stats['train']['n_docs'] + dataset_stats['vali']['n_docs'] + dataset_stats['test']['n_docs']:,}
- **Total Queries:** {dataset_stats['train']['n_queries'] + dataset_stats['vali']['n_queries'] + dataset_stats['test']['n_queries']:,}
- **Features per Document:** {dataset_stats['num_features']}

"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved benchmark report to {report_path}")


def main():
    logger.info("=== Starting MSLR-WEB10K Production Training Pipeline ===")

    # 1. Dataset Validation
    import os as _os
    fold_dir_full = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "MSLR-WEB10K", "Fold1")
    )
    fold_dir_dev = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "Fold1")
    )
    fold_dir = fold_dir_full if _os.path.exists(_os.path.join(fold_dir_full, "train.txt")) else fold_dir_dev
    logger.info(f"Dataset path: {fold_dir}")
    validate_dataset(fold_dir)

    # 2. Load Data (subset 150k rows for HPO speed; full vali/test always)
    splits = load_all_splits(hpo_subset_rows=150_000)

    X_train, y_train, qids_train = splits["train"]
    X_vali,  y_vali,  qids_vali  = splits["vali"]
    X_test,  y_test,  qids_test  = splits["test"]

    group_train = get_query_groups(qids_train)
    group_vali = get_query_groups(qids_vali)

    with open(os.path.join(REPORTS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(splits["stats"], f, indent=4)

    # 3. HPO
    best_params = run_hpo(
        X_train, y_train, qids_train, group_train,
        X_vali, y_vali, qids_vali, group_vali
    )
    
    # 3. Train Models
    logger.info("\n=== Training Final Models ===")
    xgb_path = TrainingPipeline.train_pointwise_xgb(
        X_train, y_train, X_vali, y_vali, params=best_params.get("xgboost")
    )
    
    ranknet_path = TrainingPipeline.train_pairwise_ranknet(
        X_train, y_train, qids_train, epochs=3, max_pairs=30_000
    )
    
    lgb_path = TrainingPipeline.train_listwise_lambdamart(
        X_train, y_train, group_train,
        X_vali, y_vali, group_vali, params=best_params.get("lambdamart")
    )
    
    # 4. Evaluation
    logger.info("\n=== Evaluating on Test Set ===")
    metrics_all = {}
    
    # Eval XGBoost
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    xgb_preds = xgb_model.predict(X_test)
    metrics_all["Pointwise (XGBoost)"] = compute_all_metrics("Pointwise (XGBoost)", y_test, xgb_preds, qids_test)
    
    # Eval RankNet
    net = PyTorchRankNet(input_dim=136)
    net.load_state_dict(torch.load(ranknet_path, weights_only=True))
    net.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        rn_preds = net(X_test_t).squeeze().numpy()
    metrics_all["Pairwise (RankNet)"] = compute_all_metrics("Pairwise (RankNet)", y_test, rn_preds, qids_test)
    
    # Eval LambdaMART
    gbm = lgb.Booster(model_file=lgb_path)
    lgb_preds = gbm.predict(X_test)
    metrics_all["Listwise (LambdaMART)"] = compute_all_metrics("Listwise (LambdaMART)", y_test, lgb_preds, qids_test)
    
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=4)
        
    # 5. Explainability
    logger.info("\n=== Running SHAP Explainability ===")
    if generate_shap_reports:
        try:
            generate_shap_reports(gbm, X_test, max_samples=5000)
        except Exception as e:
            logger.error(f"Failed to run SHAP explainability: {e}")
    else:
        logger.warning("SHAP is disabled due to import errors.")
    
    # 6. Artifact Registry Metadata Persistence
    logger.info("\n=== Registering model artifacts ===")
    dataset_info = {
        "dataset": "MSLR-WEB10K",
        "fold": "Fold1",
        "train_docs": int(len(y_train)),
        "vali_docs": int(len(y_vali)),
        "test_docs": int(len(y_test)),
    }
    _artifact_map = [
        ("pointwise_xgb",       "XGBoost Pointwise Regressor",  metrics_all.get("Pointwise (XGBoost)", {})),
        ("pairwise_ranknet",     "PyTorch RankNet Pairwise",      metrics_all.get("Pairwise (RankNet)", {})),
        ("listwise_lambdamart",  "LightGBM LambdaMART Listwise", metrics_all.get("Listwise (LambdaMART)", {})),
    ]
    for model_key, algorithm, metrics in _artifact_map:
        try:
            register_model(
                model_key=model_key,
                algorithm=algorithm,
                hyperparameters={},
                evaluation_metrics=metrics,
                dataset_info=dataset_info,
            )
        except Exception as e:
            logger.warning(f"Registry update failed for {model_key}: {e}")

    # 7. MLflow Tracking
    logger.info("\n=== Logging to MLflow ===")
    for model_name, metrics in metrics_all.items():
        run_name = model_name.split()[0]
        MLflowService.log_run(
            run_id=f"run-{run_name.lower()}-{int(datetime.now().timestamp())}",
            name=model_name,
            algorithm=run_name.lower(),
            parameters={"dataset": "MSLR-WEB10K Fold1 (subset)"},
            metrics=metrics
        )

    # 8. Generate Reports
    generate_benchmark_report(metrics_all, splits["stats"])

    logger.info("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
