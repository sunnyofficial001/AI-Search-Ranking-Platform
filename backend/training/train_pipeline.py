"""
Production Training Pipeline for MSLR-WEB10K
===========================================
Trains Pointwise (XGBoost), Pairwise (RankNet), and Listwise (LambdaMART)
models on real MSLR-WEB10K SVMLight data.
"""

import os
import json
import logging
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

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
            nn.Linear(32, 1)
        )
            
    def forward(self, x):
        return self.net(x)


class TrainingPipeline:
    @staticmethod
    def train_pointwise_xgb(
        X_train: np.ndarray, y_train: np.ndarray,
        X_vali: np.ndarray, y_vali: np.ndarray,
        params: Dict[str, Any] = None,
        save_dir: str = MODELS_DIR
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
        regressor.fit(
            X_train, y_train,
            eval_set=[(X_vali, y_vali)],
            verbose=False
        )
        
        regressor.save_model(model_file)
        logger.info(f"Saved Pointwise XGBoost model to {model_file}")
        return model_file


    @staticmethod
    def train_pairwise_ranknet(
        X_train: np.ndarray, y_train: np.ndarray, qids_train: np.ndarray,
        epochs: int = 20, lr: float = 0.001, max_pairs: int = 100_000,
        save_dir: str = MODELS_DIR
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
                batch_pairs = pairs[i:i+batch_size]
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
            logger.info(f"RankNet Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
            
        torch.save(net.state_dict(), model_file)
        logger.info(f"Saved Pairwise RankNet model to {model_file}")
        return model_file


    @staticmethod
    def train_listwise_lambdamart(
        X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
        X_vali: np.ndarray, y_vali: np.ndarray, group_vali: np.ndarray,
        params: Dict[str, Any] = None,
        save_dir: str = MODELS_DIR
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
            "seed": 42
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
            lgb.log_evaluation(period=50)
        ]
        
        gbm = lgb.train(
            default_params,
            train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[vali_dataset],
            callbacks=callbacks
        )
        
        gbm.save_model(model_file)
        logger.info(f"Saved Listwise LambdaMART model to {model_file}")
        return model_file

class FeatureExtractor:
    @staticmethod
    def extract_136_ranking_features(query: str, doc_title: str, doc_description: str,
                                     popularity: float, ctr: float, freshness: float, engagement: float) -> list:
        import re
        import math
        import numpy as np

        def clean_and_tokenize(text: str) -> list:
            return re.sub(r'[^\w\s-]', '', text.lower()).split()

        q_tokens = clean_and_tokenize(query)
        body_tokens = clean_and_tokenize(doc_description)
        title_tokens = clean_and_tokenize(doc_title)
        anchor_tokens = []
        url_tokens = []
        whole_doc_tokens = body_tokens + title_tokens

        # Load products for IDF calculation (dynamic corpus search)
        products = []
        try:
            from backend.database.connection import SessionLocal
            from backend.database.models import ProductModel
            db = SessionLocal()
            try:
                rows = db.query(ProductModel).all()
                products = [
                    {
                        "title": r.title,
                        "description": r.description or "",
                    }
                    for r in rows
                ]
            finally:
                db.close()
        except Exception:
            pass

        if not products:
            try:
                from backend.services.search_service import MOCK_PRODUCTS
                products = MOCK_PRODUCTS
            except Exception:
                products = []

        streams = [body_tokens, anchor_tokens, title_tokens, url_tokens, whole_doc_tokens]

        metric_names = [
            "TF", "IDF", "TF_IDF", "doc_len", "TF_normalized", "sum_TF", "min_TF", "max_TF", "mean_TF",
            "covered_query_term_ratio", "covered_query_term_num", "stream_length", "idf_sum", "idf_min",
            "idf_max", "idf_mean", "idf_variance", "tf_idf_sum", "tf_idf_min", "tf_idf_max", "tf_idf_mean",
            "bool_query", "LM", "LM_linear", "LM_dir", "BM25", "LMIR_ABS"
        ]

        stream_metrics = []
        for stream in streams:
            n_q = len(q_tokens)
            doc_len = len(stream)

            if n_q == 0:
                metrics = {k: 0.0 for k in metric_names}
                metrics["doc_len"] = float(doc_len)
                metrics["stream_length"] = float(doc_len)
                stream_metrics.append(metrics)
                continue

            tfs = [stream.count(t) for t in q_tokens]
            idfs = []
            N = len(products) if len(products) > 0 else 10
            for t in q_tokens:
                df = 0
                for p in products:
                    title = (p.get("title") or "").lower()
                    desc = (p.get("description") or "").lower()
                    if t in title or t in desc:
                        df += 1
                idfs.append(max(0.0001, math.log((N - df + 0.5) / (df + 0.5) + 1.0)))

            tf_idfs = [tf * idf for tf, idf in zip(tfs, idfs)]

            sum_TF = float(sum(tfs))
            min_TF = float(min(tfs)) if tfs else 0.0
            max_TF = float(max(tfs)) if tfs else 0.0
            mean_TF = float(sum_TF / n_q)

            covered = sum(1 for tf in tfs if tf > 0)
            covered_ratio = float(covered / n_q)
            covered_num = float(covered)

            idf_sum = float(sum(idfs))
            idf_min = float(min(idfs)) if idfs else 0.0
            idf_max = float(max(idfs)) if idfs else 0.0
            idf_mean = float(idf_sum / n_q)
            idf_var = float(np.var(idfs)) if len(idfs) > 1 else 0.0

            tf_idf_sum = float(sum(tf_idfs))
            tf_idf_min = float(min(tf_idfs)) if tf_idfs else 0.0
            tf_idf_max = float(max(tf_idfs)) if tf_idfs else 0.0
            tf_idf_mean = float(tf_idf_sum / n_q)

            bool_query = 1.0 if any(tf > 0 for tf in tfs) else 0.0

            # Language Model and Retrieval Heuristics
            lm_linear = 0.0
            lm_dir = 0.0
            lmir_abs = 0.0
            bm25 = 0.0

            k1 = 1.2
            b = 0.75
            avg_dl = 45.0

            for tf, idf in zip(tfs, idfs):
                if tf > 0:
                    bm25 += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))

            for tf in tfs:
                p_d = (tf / doc_len) if doc_len > 0 else 0.0
                p_c = 0.02

                lambda_jm = 0.1
                lm_linear += math.log(lambda_jm * p_d + (1 - lambda_jm) * p_c) if (lambda_jm * p_d + (1 - lambda_jm) * p_c) > 0 else -10.0

                mu = 2000
                p_dir = (tf + mu * p_c) / (doc_len + mu)
                lm_dir += math.log(p_dir) if p_dir > 0 else -10.0

                delta = 0.7
                p_abs = (max(tf - delta, 0) / doc_len + delta * covered * p_c / doc_len) if doc_len > 0 else p_c
                lmir_abs += math.log(p_abs) if p_abs > 0 else -10.0

            stream_metrics.append({
                "TF": sum_TF,
                "IDF": idf_sum,
                "TF_IDF": tf_idf_sum,
                "doc_len": float(doc_len),
                "TF_normalized": sum_TF / max(doc_len, 1),
                "sum_TF": sum_TF,
                "min_TF": min_TF,
                "max_TF": max_TF,
                "mean_TF": mean_TF,
                "covered_query_term_ratio": covered_ratio,
                "covered_query_term_num": covered_num,
                "stream_length": float(doc_len),
                "idf_sum": idf_sum,
                "idf_min": idf_min,
                "idf_max": idf_max,
                "idf_mean": idf_mean,
                "idf_variance": idf_var,
                "tf_idf_sum": tf_idf_sum,
                "tf_idf_min": tf_idf_min,
                "tf_idf_max": tf_idf_max,
                "tf_idf_mean": tf_idf_mean,
                "bool_query": bool_query,
                "LM": lm_linear * 0.5,
                "LM_linear": lm_linear,
                "LM_dir": lm_dir,
                "BM25": bm25,
                "LMIR_ABS": lmir_abs
            })

        fvec = [0.0] * 136
        for m_idx, metric in enumerate(metric_names):
            for s_idx in range(5):
                pos = m_idx * 5 + s_idx
                fvec[pos] = stream_metrics[s_idx][metric]

        fvec[135] = float(ctr)
        return fvec
