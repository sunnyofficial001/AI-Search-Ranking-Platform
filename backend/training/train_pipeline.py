"""
Production Training Pipeline for MSLR-WEB10K
===========================================
Trains Pointwise (XGBoost), Pairwise (RankNet), and Listwise (LambdaMART)
models on real MSLR-WEB10K SVMLight data.
"""

import logging
import os
import random
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class PyTorchRankNet(nn.Module):
    """
    136-Dimensional Neural Ranker optimized via Pairwise Cross Entropy loss.
    """

    def __init__(self, input_dim: int = 136):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class TrainingPipeline:
    @staticmethod
    def train_pointwise_xgb(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
        params: Dict[str, Any] = None,
        save_dir: str = MODELS_DIR,
    ) -> str:
        """Train XGBoost Pointwise Regressor."""
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, "pointwise_xgboost.json")

        default_params = {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
        }
        if params:
            default_params.update(params)

        logger.info(f"Training XGBoost Pointwise with params: {default_params}")

        regressor = xgb.XGBRegressor(**default_params, early_stopping_rounds=20)
        regressor.fit(X_train, y_train, eval_set=[(X_vali, y_vali)], verbose=False)

        regressor.save_model(model_file)
        logger.info(f"Saved Pointwise XGBoost model to {model_file}")
        return model_file

    @staticmethod
    def train_pairwise_ranknet(
        X_train: np.ndarray,
        y_train: np.ndarray,
        qids_train: np.ndarray,
        epochs: int = 20,
        lr: float = 0.001,
        max_pairs: int = 100_000,
        save_dir: str = MODELS_DIR,
    ) -> str:
        """Train PyTorch RankNet using sampled preference pairs from query groups."""
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, "pairwise_ranknet.pt")

        # Sample pairs
        logger.info("Sampling pairs for RankNet...")
        pairs = []
        unique_qids = np.unique(qids_train)

        for qid in unique_qids:
            if len(pairs) >= max_pairs:
                break

            mask = qids_train == qid
            X_q = X_train[mask]
            y_q = y_train[mask]

            n_docs = len(y_q)
            # Find all valid pairs where doc i is more relevant than doc j
            for i in range(n_docs):
                for j in range(n_docs):
                    if y_q[i] > y_q[j]:
                        pairs.append((X_q[i], X_q[j]))
                        if len(pairs) >= max_pairs:
                            break
                if len(pairs) >= max_pairs:
                    break

        random.shuffle(pairs)
        actual_pairs = len(pairs)
        logger.info(f"Sampled {actual_pairs:,} preference pairs.")

        net = PyTorchRankNet(input_dim=X_train.shape[1])
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        net.train()
        batch_size = 256

        for epoch in range(epochs):
            running_loss = 0.0
            np.random.shuffle(pairs)

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                if not batch_pairs:
                    break

                X1 = torch.FloatTensor(np.array([p[0] for p in batch_pairs]))
                X2 = torch.FloatTensor(np.array([p[1] for p in batch_pairs]))

                optimizer.zero_grad()

                s1 = net(X1)
                s2 = net(X2)

                # Target is 1.0 since we constructed pairs such that doc1 > doc2
                target = torch.ones_like(s1)

                loss = criterion(s1 - s2, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_pairs)

            epoch_loss = running_loss / actual_pairs
            logger.info(f"RankNet Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

        torch.save(net.state_dict(), model_file)
        logger.info(f"Saved Pairwise RankNet model to {model_file}")
        return model_file

    @staticmethod
    def train_listwise_lambdamart(
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
        group_vali: np.ndarray,
        params: Dict[str, Any] = None,
        save_dir: str = MODELS_DIR,
    ) -> str:
        """Train LightGBM LambdaMART Listwise Model."""
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, "listwise_lambdamart.txt")

        train_dataset = lgb.Dataset(X_train, label=y_train, group=group_train)
        vali_dataset = lgb.Dataset(X_vali, label=y_vali, group=group_vali, reference=train_dataset)

        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "max_depth": 6,
            "verbose": -1,
            "seed": 42,
        }
        if params:
            # Pop n_estimators if present since we pass it to lgb.train
            n_estimators = params.pop("n_estimators", 200)
            default_params.update(params)
        else:
            n_estimators = 200

        logger.info(f"Training LightGBM LambdaMART with params: {default_params}")

        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        gbm = lgb.train(
            default_params,
            train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[vali_dataset],
            callbacks=callbacks,
        )

        gbm.save_model(model_file)
        logger.info(f"Saved Listwise LambdaMART model to {model_file}")
        return model_file


class FeatureExtractor:
    @staticmethod
    def extract_136_ranking_features(
        query: str,
        doc_title: str,
        doc_description: str,
        popularity: float,
        ctr: float,
        freshness: float,
        engagement: float,
    ) -> list:
        # Mock feature vector for the API, since real feature extraction requires a complex pipeline
        return [0.0] * 136
