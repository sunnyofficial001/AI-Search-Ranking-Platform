"""
Production-Grade Feature Store
================================
Implements offline feature generation, online feature serving,
feature validation, versioning, and data lineage tracking.
Pattern: Google Feast / Tecton / Hopsworks style architecture.
"""

import datetime
import hashlib
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("feature_store")

# ---------------------------------------------------------------------------
# Feature Metadata & Schema
# ---------------------------------------------------------------------------


class FeatureType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    EMBEDDING = "embedding"


@dataclass
class FeatureSpec:
    """Declarative schema for a single feature."""

    name: str
    feature_type: FeatureType
    description: str
    source: str  # e.g. "query_doc_interaction", "user_behavior"
    tags: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = False
    version: str = "1.0"
    owner: str = "ml-team"
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class FeatureVector:
    """A computed feature vector with full lineage."""

    entity_key: str  # e.g. "query:headphones|doc:prod-3"
    features: Dict[str, Any]
    computed_at: str
    feature_version: str
    pipeline_run_id: str
    ttl_seconds: int = 300

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature Registry (declarative catalog of all features)
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: Dict[str, FeatureSpec] = {
    # --- Lexical Match Features ---
    "bm25_score": FeatureSpec(
        name="bm25_score",
        feature_type=FeatureType.FLOAT,
        description="BM25 relevance score between query and document",
        source="query_doc_interaction",
        tags=["lexical", "retrieval"],
        min_value=0.0,
        max_value=50.0,
    ),
    "tfidf_cosine": FeatureSpec(
        name="tfidf_cosine",
        feature_type=FeatureType.FLOAT,
        description="TF-IDF weighted cosine similarity",
        source="query_doc_interaction",
        tags=["lexical", "similarity"],
        min_value=0.0,
        max_value=1.0,
    ),
    "exact_match_title": FeatureSpec(
        name="exact_match_title",
        feature_type=FeatureType.BOOL,
        description="Whether query appears verbatim in title",
        source="query_doc_interaction",
        tags=["lexical", "exact_match"],
    ),
    "query_term_coverage": FeatureSpec(
        name="query_term_coverage",
        feature_type=FeatureType.FLOAT,
        description="Fraction of query terms present in document",
        source="query_doc_interaction",
        tags=["lexical"],
        min_value=0.0,
        max_value=1.0,
    ),
    "title_term_density": FeatureSpec(
        name="title_term_density",
        feature_type=FeatureType.FLOAT,
        description="Query term density in title field",
        source="query_doc_interaction",
        tags=["lexical", "field_specific"],
        min_value=0.0,
        max_value=1.0,
    ),
    # --- User Behavior Features ---
    "ctr_7d": FeatureSpec(
        name="ctr_7d",
        feature_type=FeatureType.FLOAT,
        description="Click-through rate over the last 7 days",
        source="user_behavior_logs",
        tags=["behavioral", "ctr"],
        min_value=0.0,
        max_value=1.0,
    ),
    "ctr_30d": FeatureSpec(
        name="ctr_30d",
        feature_type=FeatureType.FLOAT,
        description="Click-through rate over the last 30 days",
        source="user_behavior_logs",
        tags=["behavioral", "ctr"],
        min_value=0.0,
        max_value=1.0,
    ),
    "dwell_time_median_seconds": FeatureSpec(
        name="dwell_time_median_seconds",
        feature_type=FeatureType.FLOAT,
        description="Median post-click dwell time in seconds",
        source="user_behavior_logs",
        tags=["behavioral", "engagement"],
        min_value=0.0,
        max_value=3600.0,
    ),
    "add_to_cart_rate": FeatureSpec(
        name="add_to_cart_rate",
        feature_type=FeatureType.FLOAT,
        description="Fraction of views that resulted in cart add",
        source="user_behavior_logs",
        tags=["behavioral", "conversion"],
        min_value=0.0,
        max_value=1.0,
    ),
    "purchase_rate": FeatureSpec(
        name="purchase_rate",
        feature_type=FeatureType.FLOAT,
        description="Fraction of views that resulted in purchase",
        source="user_behavior_logs",
        tags=["behavioral", "conversion"],
        min_value=0.0,
        max_value=1.0,
    ),
    # --- Document Quality Features ---
    "freshness_score": FeatureSpec(
        name="freshness_score",
        feature_type=FeatureType.FLOAT,
        description="Time-decayed document freshness (1.0=new, 0.0=stale)",
        source="document_metadata",
        tags=["quality", "freshness"],
        min_value=0.0,
        max_value=1.0,
    ),
    "popularity_score": FeatureSpec(
        name="popularity_score",
        feature_type=FeatureType.FLOAT,
        description="Normalized document popularity (0-100)",
        source="document_metadata",
        tags=["quality", "popularity"],
        min_value=0.0,
        max_value=100.0,
    ),
    "review_count": FeatureSpec(
        name="review_count",
        feature_type=FeatureType.INT,
        description="Number of user reviews",
        source="document_metadata",
        tags=["quality", "social_proof"],
        min_value=0,
    ),
    "avg_rating": FeatureSpec(
        name="avg_rating",
        feature_type=FeatureType.FLOAT,
        description="Average star rating (1.0-5.0)",
        source="document_metadata",
        tags=["quality", "social_proof"],
        min_value=1.0,
        max_value=5.0,
    ),
    # --- Query Context Features ---
    "query_length_tokens": FeatureSpec(
        name="query_length_tokens",
        feature_type=FeatureType.INT,
        description="Number of tokens in the search query",
        source="query_analysis",
        tags=["query"],
        min_value=1,
        max_value=50,
    ),
    "query_is_navigational": FeatureSpec(
        name="query_is_navigational",
        feature_type=FeatureType.BOOL,
        description="Whether query shows navigational intent",
        source="query_analysis",
        tags=["query", "intent"],
    ),
    "query_category_match": FeatureSpec(
        name="query_category_match",
        feature_type=FeatureType.FLOAT,
        description="Soft category alignment score between query and document",
        source="query_doc_interaction",
        tags=["semantic"],
        min_value=0.0,
        max_value=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Feature Validator
# ---------------------------------------------------------------------------


class FeatureValidator:
    """Validates computed features against their registered specs."""

    @staticmethod
    def validate(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        for fname, value in features.items():
            spec = FEATURE_REGISTRY.get(fname)
            if spec is None:
                continue  # unknown features pass through (warn only)

            # Null check
            if value is None:
                if not spec.nullable:
                    errors.append(f"[{fname}] is null but marked non-nullable")
                continue

            # Range checks
            if spec.min_value is not None and isinstance(value, (int, float)):
                if value < spec.min_value:
                    errors.append(f"[{fname}] value {value} < min {spec.min_value}")
            if spec.max_value is not None and isinstance(value, (int, float)):
                if value > spec.max_value:
                    errors.append(f"[{fname}] value {value} > max {spec.max_value}")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def validate_and_clamp(features: Dict[str, Any]) -> Dict[str, Any]:
        """Returns features with out-of-range values clamped to spec bounds."""
        clean = {}
        for fname, value in features.items():
            spec = FEATURE_REGISTRY.get(fname)
            if spec is None or value is None:
                clean[fname] = value
                continue
            if isinstance(value, (int, float)):
                if spec.min_value is not None:
                    value = max(spec.min_value, value)
                if spec.max_value is not None:
                    value = min(spec.max_value, value)
            clean[fname] = value
        return clean


# ---------------------------------------------------------------------------
# Offline Feature Pipeline
# ---------------------------------------------------------------------------


class OfflineFeaturePipeline:
    """
    Batch feature computation pipeline.
    In production: Spark / Flink / dbt job writing to a feature table.
    Here: computes a full feature vector for a (query, document) pair.
    """

    @staticmethod
    def compute_query_doc_features(
        query: str,
        doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute all query-document interaction features."""
        q_tokens = [t.lower() for t in query.split() if len(t) > 1]
        title_tokens = [t.lower() for t in doc.get("title", "").split()]
        desc_tokens = [t.lower() for t in doc.get("description", "").split()]
        all_doc_tokens = title_tokens + desc_tokens

        # --- Lexical features ---
        title_matches = sum(1 for t in q_tokens if t in set(title_tokens))
        doc_matches = sum(1 for t in q_tokens if t in set(all_doc_tokens))
        query_coverage = doc_matches / max(len(q_tokens), 1)
        title_density = title_matches / max(len(title_tokens), 1)

        # BM25
        k1, b, avg_dl = 1.2, 0.75, 45.0
        freq_map: Dict[str, int] = {}
        for t in all_doc_tokens:
            freq_map[t] = freq_map.get(t, 0) + 1
        bm25 = 0.0
        for t in q_tokens:
            tf = freq_map.get(t, 0)
            if tf > 0:
                idf = max(0.01, math.log(10 / 1.5 + 1))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * len(all_doc_tokens) / avg_dl)
                bm25 += idf * (numerator / denominator)

        # Cosine TF-IDF
        vocab = list(set(q_tokens + all_doc_tokens))
        dot, qnorm, dnorm = 0.0, 0.0, 0.0
        for term in vocab:
            q_tf = q_tokens.count(term)
            d_tf = freq_map.get(term, 0)
            idf = max(0.01, math.log(10 / (1 + d_tf) + 1))
            qv, dv = q_tf * idf, d_tf * idf
            dot += qv * dv
            qnorm += qv * qv
            dnorm += dv * dv
        cosine = dot / (math.sqrt(qnorm) * math.sqrt(dnorm)) if qnorm > 0 and dnorm > 0 else 0.0

        # Exact match
        exact_match = any(t in doc.get("title", "").lower() for t in q_tokens)

        # Query category match (heuristic)
        cat = doc.get("category", "").lower()
        category_signals = {
            "electronics": [
                "electronics",
                "tech",
                "gadget",
                "device",
                "speaker",
                "headphone",
                "laptop",
                "charger",
            ],
            "apparel": ["apparel", "clothing", "fashion", "shirt", "jeans", "jacket"],
            "footwear": ["footwear", "shoes", "sneakers", "boots", "running"],
            "books": ["books", "novel", "fiction", "reading", "literature"],
        }
        cat_match = 0.0
        for cat_key, keywords in category_signals.items():
            if cat_key in cat:
                cat_match = sum(1 for t in q_tokens if t in keywords) / max(len(q_tokens), 1)
                break

        return {
            "bm25_score": round(bm25, 4),
            "tfidf_cosine": round(cosine, 4),
            "exact_match_title": exact_match,
            "query_term_coverage": round(query_coverage, 4),
            "title_term_density": round(title_density, 4),
            "query_category_match": round(cat_match, 4),
            "query_length_tokens": len(q_tokens),
            "query_is_navigational": len(q_tokens) == 1,
        }

    @staticmethod
    def compute_document_features(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Compute document-side features (independent of query)."""
        return {
            "freshness_score": doc.get("freshness", 0.5),
            "popularity_score": doc.get("popularity", 50.0),
            "ctr_7d": doc.get("ctr", 0.05),
            "ctr_30d": doc.get("ctr", 0.05) * 0.9,  # Simulate 30d smoothing
            "dwell_time_median_seconds": doc.get("engagement", 3.0) * 45,
            "add_to_cart_rate": doc.get("ctr", 0.05) * 0.4,
            "purchase_rate": doc.get("ctr", 0.05) * 0.15,
            "review_count": int(doc.get("popularity", 50) * 12),
            "avg_rating": min(5.0, 3.0 + doc.get("engagement", 3.0) * 0.4),
        }

    @classmethod
    def compute_full_feature_vector(
        cls,
        query: str,
        doc: Dict[str, Any],
        pipeline_run_id: str = "offline-batch-001",
    ) -> FeatureVector:
        """
        Compute and validate full feature vector for ranking.
        Returns a FeatureVector with lineage metadata attached.
        """
        qd_features = cls.compute_query_doc_features(query, doc)
        doc_features = cls.compute_document_features(doc)
        all_features = {**qd_features, **doc_features}

        # Validate
        is_valid, errors = FeatureValidator.validate(all_features)
        if not is_valid:
            logger.warning(f"Feature validation warnings: {errors}")
            all_features = FeatureValidator.validate_and_clamp(all_features)

        entity_key = f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}|doc:{doc.get('id', 'unknown')}"
        return FeatureVector(
            entity_key=entity_key,
            features=all_features,
            computed_at=datetime.datetime.utcnow().isoformat(),
            feature_version="1.0",
            pipeline_run_id=pipeline_run_id,
            ttl_seconds=300,
        )


# ---------------------------------------------------------------------------
# Online Feature Store (with in-memory cache + optional Redis)
# ---------------------------------------------------------------------------


class OnlineFeatureStore:
    """
    Online serving layer for pre-computed features.
    In production: connects to Redis / DynamoDB / Bigtable.
    Falls back to in-process LRU cache.
    """

    _cache: Dict[str, Dict] = {}
    _expiry: Dict[str, float] = {}
    _hit_count: int = 0
    _miss_count: int = 0

    @classmethod
    def put(cls, vector: FeatureVector) -> None:
        key = vector.entity_key
        cls._cache[key] = vector.to_dict()
        cls._expiry[key] = time.time() + vector.ttl_seconds

    @classmethod
    def get(cls, entity_key: str) -> Optional[Dict]:
        if entity_key in cls._cache:
            if time.time() < cls._expiry.get(entity_key, 0):
                cls._hit_count += 1
                return cls._cache[entity_key]
            else:
                del cls._cache[entity_key]
                del cls._expiry[entity_key]
        cls._miss_count += 1
        return None

    @classmethod
    def get_or_compute(
        cls,
        query: str,
        doc: Dict[str, Any],
        pipeline_run_id: str = "online-001",
    ) -> Dict[str, Any]:
        """Try cache first, then compute on miss."""
        entity_key = f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}|doc:{doc.get('id', 'unknown')}"
        cached = cls.get(entity_key)
        if cached:
            return cached["features"]

        vector = OfflineFeaturePipeline.compute_full_feature_vector(query, doc, pipeline_run_id)
        cls.put(vector)
        return vector.features

    @classmethod
    def cache_stats(cls) -> Dict[str, Any]:
        total = cls._hit_count + cls._miss_count
        return {
            "hits": cls._hit_count,
            "misses": cls._miss_count,
            "hit_rate": round(cls._hit_count / total, 4) if total > 0 else 0.0,
            "cached_keys": len(cls._cache),
        }

    @classmethod
    def get_feature_catalog(cls) -> List[Dict]:
        """Return full feature catalog for documentation / UI."""
        return [
            {
                "name": spec.name,
                "type": spec.feature_type.value,
                "description": spec.description,
                "source": spec.source,
                "tags": spec.tags,
                "version": spec.version,
                "owner": spec.owner,
            }
            for spec in FEATURE_REGISTRY.values()
        ]


# ---------------------------------------------------------------------------
# Data Lineage Tracker
# ---------------------------------------------------------------------------


class DataLineageTracker:
    """
    Records the lineage of feature computation runs.
    In production: writes to OpenLineage / Marquez / DataHub.
    """

    _lineage_log: List[Dict] = []

    @classmethod
    def record(
        cls,
        pipeline_name: str,
        run_id: str,
        input_datasets: List[str],
        output_features: List[str],
        row_count: int,
        duration_ms: float,
        status: str = "SUCCESS",
    ) -> None:
        entry = {
            "pipeline": pipeline_name,
            "run_id": run_id,
            "started_at": datetime.datetime.utcnow().isoformat(),
            "inputs": input_datasets,
            "outputs": output_features,
            "row_count": row_count,
            "duration_ms": round(duration_ms, 2),
            "status": status,
        }
        cls._lineage_log.append(entry)
        logger.info(f"[Lineage] {pipeline_name} run {run_id}: {row_count} rows, {duration_ms:.1f}ms, {status}")

    @classmethod
    def get_lineage(cls, pipeline_name: Optional[str] = None) -> List[Dict]:
        if pipeline_name:
            return [e for e in cls._lineage_log if e["pipeline"] == pipeline_name]
        return list(reversed(cls._lineage_log))
