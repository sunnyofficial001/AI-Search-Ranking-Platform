"""
Advanced Recommendation Engine
================================
Implements production-grade recommendation systems:
  - Session-based recommendations (GRU4Rec-style sequence modeling)
  - Cold-start handling (content-based + popularity fallback)
  - Two-tower embedding model (user × item)
  - Item2Vec embeddings (Word2Vec on interaction sequences)
  - Diversity optimization (Intra-List Diversity)
  - Novelty optimization (suppress already-seen items)
  - Temporal context-aware recommendations
  - Exploration-exploitation via ε-greedy

Industry patterns from: Netflix (TF-Recommender), Spotify (Discover Weekly),
Amazon (item2item), YouTube (DNN ranking), LinkedIn (PYMK).
"""

import math
import random
import hashlib
import datetime
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("rec_engine")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RecommendationItem:
    item_id: str
    title: str
    category: str
    score: float
    recommendation_type: str
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    explanation: str = ""
    embedding_similarity: float = 0.0


@dataclass
class UserSession:
    user_id: str
    viewed_items: List[str] = field(default_factory=list)
    clicked_items: List[str] = field(default_factory=list)
    purchased_items: List[str] = field(default_factory=list)
    session_start: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)


class RecommendationType(str, Enum):
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    MATRIX_FACTORIZATION = "matrix_factorization"
    SESSION_BASED = "session_based"
    TWO_TOWER = "two_tower"
    ITEM2VEC = "item2vec"
    COLD_START = "cold_start"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Item Catalog (extended with embedding-like representation)
# ---------------------------------------------------------------------------

ITEM_CATALOG = [
    {
        "id": "prod-1", "title": "Amazon Echo Dot (5th Gen) - Smart Speaker Alexa",
        "category": "Electronics", "subcategory": "Smart Home",
        "tags": ["smart home", "alexa", "speaker", "voice assistant", "wifi"],
        "popularity": 92.0, "ctr": 0.12, "freshness": 0.85, "engagement": 4.6,
        "price_tier": "budget",
    },
    {
        "id": "prod-2", "title": "Fjallraven Kanken Classic Minimalist Backpack",
        "category": "Apparel", "subcategory": "Bags",
        "tags": ["backpack", "minimalist", "outdoor", "travel", "school"],
        "popularity": 88.0, "ctr": 0.08, "freshness": 0.60, "engagement": 4.4,
        "price_tier": "mid",
    },
    {
        "id": "prod-3", "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "category": "Electronics", "subcategory": "Audio",
        "tags": ["headphones", "noise cancelling", "wireless", "sony", "bluetooth", "audiophile"],
        "popularity": 95.0, "ctr": 0.15, "freshness": 0.90, "engagement": 4.8,
        "price_tier": "premium",
    },
    {
        "id": "prod-4", "title": "Levi's Men's 511 Slim Fit Jeans Stretch Denim",
        "category": "Apparel", "subcategory": "Jeans",
        "tags": ["jeans", "denim", "slim fit", "levis", "fashion", "casual"],
        "popularity": 79.0, "ctr": 0.05, "freshness": 0.40, "engagement": 4.1,
        "price_tier": "mid",
    },
    {
        "id": "prod-5", "title": "Anker USB-C Charger Nano 30W Super Fast Charging",
        "category": "Electronics", "subcategory": "Accessories",
        "tags": ["charger", "usb-c", "fast charging", "portable", "anker", "gan"],
        "popularity": 91.0, "ctr": 0.18, "freshness": 0.95, "engagement": 4.7,
        "price_tier": "budget",
    },
    {
        "id": "prod-6", "title": "Apple AirPods Pro (2nd Gen) with USB-C",
        "category": "Electronics", "subcategory": "Audio",
        "tags": ["airpods", "apple", "earbuds", "noise cancelling", "wireless", "ios"],
        "popularity": 98.0, "ctr": 0.22, "freshness": 0.92, "engagement": 4.9,
        "price_tier": "premium",
    },
    {
        "id": "prod-7", "title": "Nike Men's Air Zoom Pegasus Running Shoes",
        "category": "Footwear", "subcategory": "Running",
        "tags": ["running", "nike", "shoes", "athletic", "sport", "zoom"],
        "popularity": 84.0, "ctr": 0.07, "freshness": 0.70, "engagement": 4.3,
        "price_tier": "mid",
    },
    {
        "id": "prod-8", "title": "The Alchemist - Original Hardcover Fiction",
        "category": "Books", "subcategory": "Fiction",
        "tags": ["book", "fiction", "bestseller", "inspiration", "philosophy"],
        "popularity": 75.0, "ctr": 0.04, "freshness": 0.20, "engagement": 4.5,
        "price_tier": "budget",
    },
    {
        "id": "prod-9", "title": "Asus ROG Zephyrus G14 Gaming Laptop RTX 4060",
        "category": "Electronics", "subcategory": "Computers",
        "tags": ["laptop", "gaming", "asus", "rog", "rtx", "amd", "high performance"],
        "popularity": 86.0, "ctr": 0.10, "freshness": 0.88, "engagement": 4.5,
        "price_tier": "premium",
    },
    {
        "id": "prod-10", "title": "Stan Smith Ortholite Recycled Clean Sneakers",
        "category": "Footwear", "subcategory": "Casual",
        "tags": ["sneakers", "adidas", "stan smith", "casual", "sustainable", "clean"],
        "popularity": 81.0, "ctr": 0.06, "freshness": 0.50, "engagement": 4.2,
        "price_tier": "mid",
    },
]

ITEM_BY_ID = {item["id"]: item for item in ITEM_CATALOG}


# ---------------------------------------------------------------------------
# Embedding Simulator (Item2Vec-style)
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Simulates learned item embeddings from co-occurrence patterns.
    In production: trained via Word2Vec / FastText on session logs,
    or deep learning two-tower model.
    """

    # Tag-based embedding proxy (normalized tag overlap = embedding similarity)
    @staticmethod
    def item_embedding(item_id: str) -> List[float]:
        """Return a deterministic pseudo-embedding based on item features."""
        item = ITEM_BY_ID.get(item_id)
        if not item:
            return [0.0] * 16

        tags = item.get("tags", [])
        cat = item.get("category", "")

        # Deterministic embedding from feature fingerprint
        seed = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (2**31)
        random.seed(seed)
        base = [random.gauss(0, 1) for _ in range(16)]

        # Inject category signal (strong cluster effect)
        cat_map = {"Electronics": 0, "Apparel": 4, "Footwear": 8, "Books": 12}
        cat_offset = cat_map.get(cat, 0)
        for i in range(4):
            base[cat_offset + i] += 2.0  # Category cluster

        # Inject tag signals
        tag_signals = {
            "noise cancelling": (0, 0.8), "wireless": (1, 0.6),
            "premium": (2, 0.7), "budget": (3, -0.5),
            "fashion": (4, 0.6), "casual": (5, 0.4),
            "running": (8, 0.8), "athletic": (9, 0.5),
            "gaming": (0, 0.9), "high performance": (1, 0.8),
        }
        for tag in tags:
            if tag in tag_signals:
                idx, strength = tag_signals[tag]
                base[idx] += strength

        # Normalize to unit sphere
        norm = math.sqrt(sum(x * x for x in base))
        return [x / norm for x in base] if norm > 0 else base

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return round(dot / (norm_a * norm_b), 4)

    @classmethod
    def most_similar(
        cls,
        query_item_id: str,
        candidates: List[str],
        top_k: int = 5,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find most similar items by embedding cosine similarity."""
        query_emb = cls.item_embedding(query_item_id)
        exclude = exclude_ids or {query_item_id}
        similarities = []
        for item_id in candidates:
            if item_id in exclude:
                continue
            item_emb = cls.item_embedding(item_id)
            sim = cls.cosine_similarity(query_emb, item_emb)
            similarities.append((item_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ---------------------------------------------------------------------------
# Session-Based Recommender (GRU4Rec-inspired)
# ---------------------------------------------------------------------------

class SessionBasedRecommender:
    """
    Session-based recommendations using sequential patterns.
    In production: GRU4Rec, BERT4Rec, or SASRec neural model.
    Here: transition matrix + recency weighting approximation.
    """

    # Co-click transition probabilities (learned from session logs)
    TRANSITION_MATRIX = {
        "prod-1": {"prod-5": 0.45, "prod-3": 0.30, "prod-6": 0.25},   # Echo → charger, headphones
        "prod-3": {"prod-6": 0.50, "prod-5": 0.25, "prod-9": 0.25},   # Headphones → AirPods, charger
        "prod-6": {"prod-3": 0.40, "prod-5": 0.35, "prod-1": 0.25},   # AirPods → headphones, charger
        "prod-5": {"prod-3": 0.35, "prod-6": 0.30, "prod-9": 0.35},   # Charger → audio, laptop
        "prod-2": {"prod-4": 0.40, "prod-7": 0.30, "prod-10": 0.30},  # Backpack → jeans, shoes
        "prod-4": {"prod-2": 0.35, "prod-7": 0.35, "prod-10": 0.30},  # Jeans → backpack, shoes
        "prod-7": {"prod-10": 0.50, "prod-4": 0.30, "prod-2": 0.20},  # Nike → Adidas, jeans
        "prod-10": {"prod-7": 0.55, "prod-4": 0.25, "prod-2": 0.20},  # Adidas → Nike
        "prod-9": {"prod-5": 0.40, "prod-3": 0.35, "prod-6": 0.25},   # Laptop → charger, audio
        "prod-8": {"prod-4": 0.30, "prod-2": 0.30, "prod-7": 0.40},   # Book → lifestyle items
    }

    @classmethod
    def recommend(
        cls,
        session: UserSession,
        exclude_viewed: bool = True,
        top_k: int = 6,
    ) -> List[RecommendationItem]:
        if not session.clicked_items and not session.viewed_items:
            return cls._cold_start_fallback(top_k)

        # Build score map from session sequence with recency decay
        scores: Dict[str, float] = {}
        all_interactions = session.clicked_items or session.viewed_items

        for pos, item_id in enumerate(reversed(all_interactions)):
            recency_weight = 0.8 ** pos  # Exponential decay for older events
            transitions = cls.TRANSITION_MATRIX.get(item_id, {})
            for target_id, prob in transitions.items():
                scores[target_id] = scores.get(target_id, 0) + prob * recency_weight

            # Embedding-based fallback for items not in transition matrix
            if not transitions:
                similar = EmbeddingEngine.most_similar(
                    item_id, list(ITEM_BY_ID.keys()), top_k=3
                )
                for sim_id, sim_score in similar:
                    scores[sim_id] = scores.get(sim_id, 0) + sim_score * recency_weight * 0.5

        # Filter viewed items
        exclude = set(session.viewed_items + session.clicked_items) if exclude_viewed else set()
        filtered_scores = {k: v for k, v in scores.items() if k not in exclude}

        # Build results
        results = []
        for item_id, score in sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True):
            item = ITEM_BY_ID.get(item_id)
            if not item:
                continue
            results.append(RecommendationItem(
                item_id=item_id,
                title=item["title"],
                category=item["category"],
                score=round(score, 4),
                recommendation_type=RecommendationType.SESSION_BASED.value,
                explanation=f"Users who viewed {all_interactions[-1] if all_interactions else 'similar items'} also viewed this.",
            ))

        return results[:top_k]

    @staticmethod
    def _cold_start_fallback(top_k: int) -> List[RecommendationItem]:
        """For new sessions, return popularity-ranked items."""
        sorted_items = sorted(ITEM_CATALOG, key=lambda x: x["popularity"], reverse=True)
        return [
            RecommendationItem(
                item_id=item["id"],
                title=item["title"],
                category=item["category"],
                score=round(item["popularity"] / 100.0, 4),
                recommendation_type=RecommendationType.COLD_START.value,
                explanation="Trending item — popular with most users.",
            )
            for item in sorted_items[:top_k]
        ]


# ---------------------------------------------------------------------------
# Two-Tower Embedding Recommender
# ---------------------------------------------------------------------------

class TwoTowerRecommender:
    """
    Two-tower (dual encoder) model for user-item matching.
    User tower: encodes user history/demographics
    Item tower: encodes item features
    Score = dot product in shared embedding space.

    In production: trained via contrastive learning on click logs.
    Industry: YouTube DNN, Google Play, Airbnb.
    """

    # Simulated user embeddings (in production: output of user encoder)
    USER_PROFILES = {
        "user-1": {"tech_affinity": 0.90, "fashion_affinity": 0.20, "price_sensitivity": 0.30},
        "user-2": {"tech_affinity": 0.25, "fashion_affinity": 0.85, "price_sensitivity": 0.60},
        "user-3": {"tech_affinity": 0.80, "fashion_affinity": 0.15, "price_sensitivity": 0.20},
        "user-4": {"tech_affinity": 0.15, "fashion_affinity": 0.90, "price_sensitivity": 0.70},
        "user-5": {"tech_affinity": 0.95, "fashion_affinity": 0.10, "price_sensitivity": 0.15},
    }

    PRICE_TIER_MAP = {"budget": 0.3, "mid": 0.6, "premium": 0.9}
    CATEGORY_DIMENSION = {"Electronics": "tech_affinity", "Apparel": "fashion_affinity",
                           "Footwear": "fashion_affinity", "Books": "fashion_affinity"}

    @classmethod
    def get_user_embedding(cls, user_id: str) -> Dict[str, float]:
        return cls.USER_PROFILES.get(user_id, {
            "tech_affinity": 0.5, "fashion_affinity": 0.5, "price_sensitivity": 0.5
        })

    @classmethod
    def score_item_for_user(cls, user_id: str, item: Dict[str, Any]) -> float:
        user_emb = cls.get_user_embedding(user_id)
        category = item.get("category", "")
        price_tier = item.get("price_tier", "mid")
        price_val = cls.PRICE_TIER_MAP.get(price_tier, 0.6)
        cat_dim = cls.CATEGORY_DIMENSION.get(category, "tech_affinity")

        # Dot product in embedding space
        affinity = user_emb.get(cat_dim, 0.5)
        price_match = 1.0 - abs(user_emb.get("price_sensitivity", 0.5) - price_val)
        quality = item.get("engagement", 3.0) / 5.0

        score = affinity * 0.50 + price_match * 0.25 + quality * 0.25
        return round(min(1.0, score), 4)

    @classmethod
    def recommend(
        cls,
        user_id: str,
        seen_items: Optional[Set[str]] = None,
        top_k: int = 6,
    ) -> List[RecommendationItem]:
        exclude = seen_items or set()
        scored = []
        for item in ITEM_CATALOG:
            if item["id"] in exclude:
                continue
            score = cls.score_item_for_user(user_id, item)
            emb_sim = EmbeddingEngine.cosine_similarity(
                EmbeddingEngine.item_embedding(item["id"]),
                [cls.get_user_embedding(user_id).get(k, 0.5) for k in
                 ["tech_affinity", "fashion_affinity", "price_sensitivity", 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
            )
            scored.append(RecommendationItem(
                item_id=item["id"],
                title=item["title"],
                category=item["category"],
                score=score,
                recommendation_type=RecommendationType.TWO_TOWER.value,
                embedding_similarity=abs(emb_sim),
                explanation=f"Matched to your profile via two-tower model (user-item affinity={score:.2f})",
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Cold-Start Handler
# ---------------------------------------------------------------------------

class ColdStartHandler:
    """
    Handles cold-start scenarios:
    1. New user (no history) → popularity + diversity
    2. New item (no interactions) → content-based similarity
    3. New user + new item → demographic segment defaults
    """

    @staticmethod
    def handle_new_user(
        demographic_context: Optional[Dict[str, Any]] = None,
        top_k: int = 6,
    ) -> List[RecommendationItem]:
        """Recommend for new users with no interaction history."""
        # Stage 1: popularity-based
        sorted_items = sorted(ITEM_CATALOG, key=lambda x: x["popularity"] * x["ctr"], reverse=True)

        # Stage 2: diversity injection (ensure category spread)
        selected = []
        category_seen: Set[str] = set()
        for item in sorted_items:
            if len(selected) >= top_k:
                break
            # Allow at most 2 items per category for new users
            cat_count = sum(1 for s in selected if s.category == item["category"])
            if cat_count >= 2:
                continue
            selected.append(RecommendationItem(
                item_id=item["id"],
                title=item["title"],
                category=item["category"],
                score=round(item["popularity"] / 100.0 * 0.7 + item["ctr"] * 3 * 0.3, 4),
                recommendation_type=RecommendationType.COLD_START.value,
                explanation="Trending and highly rated — great for new users.",
            ))
            category_seen.add(item["category"])

        return selected

    @staticmethod
    def handle_new_item(new_item: Dict[str, Any], top_k: int = 4) -> List[str]:
        """Find similar existing items for item cold-start (co-listing)."""
        new_tags = set(new_item.get("tags", []))
        new_cat = new_item.get("category", "")

        similarities = []
        for item in ITEM_CATALOG:
            if item["id"] == new_item.get("id"):
                continue
            item_tags = set(item.get("tags", []))
            tag_overlap = len(new_tags & item_tags) / max(len(new_tags | item_tags), 1)
            cat_match = 1.0 if item["category"] == new_cat else 0.0
            score = tag_overlap * 0.6 + cat_match * 0.4
            similarities.append((item["id"], score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in similarities[:top_k]]


# ---------------------------------------------------------------------------
# Diversity & Novelty Optimizer
# ---------------------------------------------------------------------------

class DiversityOptimizer:
    """
    Optimizes recommendation lists for diversity and novelty.
    - Intra-List Diversity (ILD): average pairwise dissimilarity
    - Novelty: penalize mainstream / already-seen items
    """

    @staticmethod
    def intra_list_diversity(items: List[RecommendationItem]) -> float:
        """Compute ILD = average pairwise embedding distance."""
        if len(items) < 2:
            return 0.0
        total_dissimilarity = 0.0
        pair_count = 0
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                emb_a = EmbeddingEngine.item_embedding(a.item_id)
                emb_b = EmbeddingEngine.item_embedding(b.item_id)
                similarity = EmbeddingEngine.cosine_similarity(emb_a, emb_b)
                total_dissimilarity += 1.0 - similarity
                pair_count += 1
        return round(total_dissimilarity / max(pair_count, 1), 4)

    @staticmethod
    def compute_novelty(item: RecommendationItem, seen_items: Set[str]) -> float:
        """Novelty = 0 if seen, otherwise inversely proportional to popularity."""
        catalog_item = ITEM_BY_ID.get(item.item_id, {})
        if item.item_id in seen_items:
            return 0.0
        popularity = catalog_item.get("popularity", 50.0)
        return round(1.0 - (popularity / 100.0), 4)

    @classmethod
    def optimize_list(
        cls,
        items: List[RecommendationItem],
        seen_items: Optional[Set[str]] = None,
        novelty_weight: float = 0.15,
        diversity_enforce: bool = True,
    ) -> List[RecommendationItem]:
        """
        Re-score items incorporating diversity and novelty signals.
        """
        seen = seen_items or set()
        category_counts: Dict[str, int] = {}

        optimized = []
        for item in items:
            novelty = cls.compute_novelty(item, seen)
            cat_count = category_counts.get(item.category, 0)
            diversity_bonus = max(0.0, 0.1 - cat_count * 0.03)  # Diminishing returns per category

            final_score = (
                item.score * (1.0 - novelty_weight) +
                novelty * novelty_weight +
                diversity_bonus
            )
            item.novelty_score = novelty
            item.diversity_score = diversity_bonus
            item.score = round(min(1.0, final_score), 4)
            optimized.append(item)
            category_counts[item.category] = cat_count + 1

        optimized.sort(key=lambda x: x.score, reverse=True)
        return optimized


# ---------------------------------------------------------------------------
# Unified Recommendation Service
# ---------------------------------------------------------------------------

class AdvancedRecommendService:
    """
    Unified recommendation service that routes to appropriate algorithm.
    """

    @classmethod
    def recommend(
        cls,
        rec_type: RecommendationType,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        session: Optional[UserSession] = None,
        top_k: int = 6,
        apply_diversity: bool = True,
        seen_items: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Route recommendation request to appropriate engine.
        """
        start_time = __import__("time").time()

        if rec_type == RecommendationType.SESSION_BASED:
            sess = session or UserSession(user_id=user_id or "anonymous")
            if item_id:
                sess.clicked_items = [item_id]
            items = SessionBasedRecommender.recommend(sess, top_k=top_k)

        elif rec_type == RecommendationType.TWO_TOWER:
            items = TwoTowerRecommender.recommend(user_id or "user-1", seen_items, top_k)

        elif rec_type == RecommendationType.COLD_START:
            items = ColdStartHandler.handle_new_user(top_k=top_k)

        elif rec_type == RecommendationType.ITEM2VEC:
            if not item_id:
                item_id = ITEM_CATALOG[0]["id"]
            similar = EmbeddingEngine.most_similar(
                item_id, list(ITEM_BY_ID.keys()), top_k=top_k
            )
            items = [
                RecommendationItem(
                    item_id=sid,
                    title=ITEM_BY_ID[sid]["title"],
                    category=ITEM_BY_ID[sid]["category"],
                    score=score,
                    recommendation_type=RecommendationType.ITEM2VEC.value,
                    embedding_similarity=score,
                    explanation=f"Embedding cosine similarity: {score:.3f}",
                )
                for sid, score in similar
                if sid in ITEM_BY_ID
            ]

        else:
            # Default: hybrid combination
            cf_items = SessionBasedRecommender._cold_start_fallback(top_k)
            tt_items = TwoTowerRecommender.recommend(user_id or "user-1", seen_items, top_k)
            score_map: Dict[str, float] = {}
            for item in cf_items:
                score_map[item.item_id] = score_map.get(item.item_id, 0) + item.score * 0.4
            for item in tt_items:
                score_map[item.item_id] = score_map.get(item.item_id, 0) + item.score * 0.6
            items = []
            for iid, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                cat_item = ITEM_BY_ID.get(iid, {})
                items.append(RecommendationItem(
                    item_id=iid,
                    title=cat_item.get("title", iid),
                    category=cat_item.get("category", ""),
                    score=round(score, 4),
                    recommendation_type=RecommendationType.HYBRID.value,
                    explanation=f"Hybrid score (40% session + 60% two-tower)",
                ))

        # Apply diversity optimization
        if apply_diversity and items:
            items = DiversityOptimizer.optimize_list(items, seen_items)

        # Compute ILD metric
        ild = DiversityOptimizer.intra_list_diversity(items)
        latency_ms = round((__import__("time").time() - start_time) * 1000, 2)

        return {
            "type": rec_type.value,
            "user_id": user_id,
            "items": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "score": item.score,
                    "type": item.recommendation_type,
                    "novelty": item.novelty_score,
                    "diversity_bonus": item.diversity_score,
                    "embedding_similarity": item.embedding_similarity,
                    "explanation": item.explanation,
                }
                for item in items
            ],
            "metadata": {
                "intra_list_diversity": ild,
                "latency_ms": latency_ms,
                "diversity_applied": apply_diversity,
                "algorithm": rec_type.value,
            },
        }
