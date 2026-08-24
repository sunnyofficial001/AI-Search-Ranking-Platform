"""
Hyperparameter Optimization (HPO) Module
=========================================
Uses Optuna to find the best hyperparameters for XGBoost and LambdaMART models.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb

from backend.evaluation.ranking_metrics import evaluate_per_query

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def optimize_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, qids_train: np.ndarray,
    X_vali: np.ndarray, y_vali: np.ndarray, qids_vali: np.ndarray,
    n_trials: int = 10
) -> Dict[str, Any]:
    """Optimize XGBoost Pointwise Regressor."""
    logger.info("Starting Optuna HPO for XGBoost Pointwise...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "tree_method": "hist",
        }
        
        regressor = xgb.XGBRegressor(**params, early_stopping_rounds=15)
        regressor.fit(
            X_train, y_train,
            eval_set=[(X_vali, y_vali)],
            verbose=False
        )
        
        preds = regressor.predict(X_vali)
        metrics = evaluate_per_query(y_vali, preds, qids_vali, k_values=(10,))
        return metrics["NDCG@10"]

    study = optuna.create_study(direction="maximize")
    # Silence optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"XGBoost Best NDCG@10: {study.best_value:.4f}")
    logger.info(f"XGBoost Best Params: {study.best_params}")
    return study.best_params


def optimize_lambdamart(
    X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
    X_vali: np.ndarray, y_vali: np.ndarray, group_vali: np.ndarray,
    qids_vali: np.ndarray,
    n_trials: int = 15
) -> Dict[str, Any]:
    """Optimize LightGBM LambdaMART Listwise."""
    logger.info("Starting Optuna HPO for LambdaMART Listwise...")
    
    train_dataset = lgb.Dataset(X_train, label=y_train, group=group_train)
    vali_dataset = lgb.Dataset(X_vali, label=y_vali, group=group_vali, reference=train_dataset)

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "feature_pre_filter": False,
            "verbose": -1,
            "seed": 42
        }
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        
        gbm = lgb.train(
            params,
            train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[vali_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        preds = gbm.predict(X_vali)
        metrics = evaluate_per_query(y_vali, preds, qids_vali, k_values=(10,))
        return metrics["NDCG@10"]

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"LambdaMART Best NDCG@10: {study.best_value:.4f}")
    logger.info(f"LambdaMART Best Params: {study.best_params}")
    
    best_params = study.best_params
    best_params["n_estimators"] = best_params.pop("n_estimators", 200)
    return best_params


def run_hpo(
    X_train, y_train, qids_train, group_train,
    X_vali, y_vali, qids_vali, group_vali
) -> Dict[str, Dict[str, Any]]:
    """Run HPO for all models and save to models/best_params.json"""
    best_params = {}
    
    best_params["xgboost"] = optimize_xgboost(
        X_train, y_train, qids_train, X_vali, y_vali, qids_vali, n_trials=5
    )
    
    best_params["lambdamart"] = optimize_lambdamart(
        X_train, y_train, group_train, X_vali, y_vali, group_vali, qids_vali, n_trials=8
    )
    
    # RankNet uses fixed default parameters for stability in this script
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
        
    return best_params
