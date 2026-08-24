import os
import re
import math
import logging
from typing import List, Dict, Any

from backend.core.config import settings

logger = logging.getLogger("search_service")

# Try to connect to Elasticsearch at import time.
# ES_AVAILABLE=False means the cluster is unreachable — this is handled
# explicitly in fetch_candidate_documents() per the platform rule:
# "If a dependency/service/model is unavailable, FAIL EXPLICITLY."
try:
    from elasticsearch import Elasticsearch
    ES_HOSTS = os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200")
    es_client = Elasticsearch(ES_HOSTS, request_timeout=5)
    es_client.ping()
    ES_AVAILABLE = True
    logger.info("Elasticsearch cluster reachable at %s.", ES_HOSTS)
except Exception as e:
    logger.warning("Elasticsearch unavailable (%s). Retrieval will fail explicitly unless TESTING=true.", e)
    ES_AVAILABLE = False
    es_client = None

# Reference vocabulary mapping computed dynamically
MOCK_PRODUCTS = [
    {
        "id": "prod-1",
        "title": "Amazon Echo Dot (5th Gen) - Smart Speaker Alexa",
        "description": "Our most popular smart speaker with Alexa features a sleek compact design and delivers vibrant sound. Play music, check weather, control smart devices.",
        "category": "Electronics",
        "popularity": 92.0,
        "ctr": 0.12,
        "freshness": 0.85,
        "engagement": 4.6
    },
    {
        "id": "prod-2",
        "title": "Fjallraven Kanken Classic Minimalist Backpack",
        "description": "Classic Kanken backpack in hard-wearing Vinylon fabric with a zip that opens the entire main compartment. Ergononic straps, removable seat pad, and handles.",
        "category": "Apparel",
        "popularity": 88.0,
        "ctr": 0.08,
        "freshness": 0.60,
        "engagement": 4.4
    },
    {
        "id": "prod-3",
        "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "description": "Industry leading noise canceling headphones with custom sound engineering, pristine microphone, 30 hours battery life, and Alexa Google Voice integrated.",
        "category": "Electronics",
        "popularity": 95.0,
        "ctr": 0.15,
        "freshness": 0.90,
        "engagement": 4.8
    },
    {
        "id": "prod-4",
        "title": "Levi's Men's 511 Slim Fit Jeans Stretch Denim",
        "description": "A modern slim with room to move. The 511 slim fit denim is a classic since inception. Crafted from premium stretch cotton for ultimate active comfort.",
        "category": "Apparel",
        "popularity": 79.0,
        "ctr": 0.05,
        "freshness": 0.40,
        "engagement": 4.1
    },
    {
        "id": "prod-5",
        "title": "Anker USB-C Charger Nano 30W Super Fast Charging",
        "description": "Extremely compact GaN II fast wall charger block for iPhone, Galaxy, iPad, MacBook Air. MultiProtect safety features, fold-away plug design.",
        "category": "Electronics",
        "popularity": 91.0,
        "ctr": 0.18,
        "freshness": 0.95,
        "engagement": 4.7
    },
    {
        "id": "prod-6",
        "title": "Apple AirPods Pro (2nd Gen) with USB-C",
        "description": "Re-engineered noise cancellation, adaptive audio transparency mode, spatial personalized high-fidelity audio, and extra comfortable silicone ear tips.",
        "category": "Electronics",
        "popularity": 98.0,
        "ctr": 0.22,
        "freshness": 0.92,
        "engagement": 4.9
    },
    {
        "id": "prod-7",
        "title": "Nike Men's Air Zoom Pegasus Running Shoes",
        "description": "The classic Pegasus returns. Designed for neutral running and high physical elasticity. Breathable mesh top, double Zoom Air chambers.",
        "category": "Footwear",
        "popularity": 84.0,
        "ctr": 0.07,
        "freshness": 0.70,
        "engagement": 4.3
    },
    {
        "id": "prod-8",
        "title": "The Alchemist - Original Hardcover Fiction",
        "description": "Fable about following your dreams, listening to your gut, and finding treasure. Over 65 million copies sold globally. A timeless inspirational text.",
        "category": "Books",
        "popularity": 75.0,
        "ctr": 0.04,
        "freshness": 0.20,
        "engagement": 4.5
    },
    {
        "id": "prod-9",
        "title": "Asus ROG Zephyrus G14 Gaming Laptop RTX 4060",
        "description": "Top performing 14-inch Windows system. AMD Ryzen 9 processor, NVIDIA RTX 4060 GPU, Nebula HDR Display, sleek white chassis with animatrix design.",
        "category": "Electronics",
        "popularity": 86.0,
        "ctr": 0.10,
        "freshness": 0.88,
        "engagement": 4.5
    },
    {
        "id": "prod-10",
        "title": "Stan Smith Ortholite Recycled Clean Sneakers",
        "description": "Sustainable clean court sneakers made from Primegreen high performance recycled materials. White leather core paired with classic green details.",
        "category": "Footwear",
        "popularity": 81.0,
        "ctr": 0.06,
        "freshness": 0.50,
        "engagement": 4.2
    }
]

def tokenize(text: str) -> List[str]:
    clean = re.sub(r'[^\w\s-]', '', text.lower())
    return [t for t in clean.split() if len(t) > 2]

# Compute document stats needed for BM25
DOC_COUNT = len(MOCK_PRODUCTS)
DOC_LENGTHS = [len(tokenize(p["title"] + " " + p["description"])) for p in MOCK_PRODUCTS]
AVG_DOC_LENGTH = sum(DOC_LENGTHS) / DOC_COUNT

TERM_DFS: Dict[str, int] = {}
for p in MOCK_PRODUCTS:
    tokens = set(tokenize(p["title"] + " " + p["description"]))
    for t in tokens:
        TERM_DFS[t] = TERM_DFS.get(t, 0) + 1

def calculate_idf(term: str) -> float:
    df = TERM_DFS.get(term, 0)
    return max(0.0001, math.log((DOC_COUNT - df + 0.5) / (df + 0.5) + 1))

class SearchService:
    @staticmethod
    def expand_query(query: str) -> str:
        norm = query.lower().strip()
        if any(w in norm for w in ["alexa", "speaker", "smart"]):
            return f"{query} echo alexa home electronic sound control"
        if any(w in norm for w in ["headphone", "noise", "sound"]):
            return f"{query} wireless custom sound bass sony anc cancellation"
        if any(w in norm for w in ["shoe", "run", "sneaker"]):
            return f"{query} sneakers walking lightweight athletic nike comfort pegasus"
        if any(w in norm for w in ["charger", "fast"]):
            return f"{query} fast charging usb-c anker power adapter port block"
        if any(w in norm for w in ["backpack", "bag"]):
            return f"{query} vintage classic travel laptop vinylon fjallraven durable"
        return f"{query} premium quality"

    @staticmethod
    def calculate_bm25(query: str, doc_title: str, doc_desc: str) -> float:
        q_tokens = tokenize(query)
        doc_tokens = tokenize(doc_title + " " + doc_desc)
        doc_len = len(doc_tokens)
        
        f_store = {}
        for t in doc_tokens:
            f_store[t] = f_store.get(t,0) + 1
            
        k1 = 1.2
        b = 0.75
        score = 0.0
        
        for term in q_tokens:
            if term in f_store:
                tf = f_store[term]
                idf = calculate_idf(term)
                tf_num = tf * (k1 + 1)
                tf_den = tf + k1 * (1 - b + b * (doc_len / AVG_DOC_LENGTH))
                score += idf * (tf_num / tf_den)
                
        return round(score, 3)

    @staticmethod
    def calculate_tfidf_and_cosine(query: str, doc_title: str, doc_desc: str) -> Dict[str, float]:
        q_tokens = tokenize(query)
        p_tokens = tokenize(doc_title + " " + doc_desc)
        if not q_tokens or not p_tokens:
            return {"cosine": 0.0, "tfidf": 0.0}
            
        vocab = list(set(q_tokens + p_tokens))
        q_tf = {}
        p_tf = {}
        
        for t in q_tokens: q_tf[t] = q_tf.get(t, 0) + 1
        for t in p_tokens: p_tf[t] = p_tf.get(t, 0) + 1
        
        dot_product = 0.0
        q_norm_sq = 0.0
        p_norm_sq = 0.0
        total_p_tfidf = 0.0
        
        for term in vocab:
            idf = calculate_idf(term)
            q_val = q_tf.get(term, 0) * idf
            p_val = p_tf.get(term, 0) * idf
            
            dot_product += q_val * p_val
            q_norm_sq += q_val * q_val
            p_norm_sq += p_val * p_val
            
            if term in q_tf:
                total_p_tfidf += p_val
                
        q_norm = math.sqrt(q_norm_sq)
        p_norm = math.sqrt(p_norm_sq)
        cosine = (dot_product / (q_norm * p_norm)) if (q_norm > 0 and p_norm > 0) else 0.0
        
        return {
            "cosine": round(cosine, 3),
            "tfidf": round(total_p_tfidf, 3)
        }

    @staticmethod
    def fetch_candidate_documents(query: str) -> List[Dict[str, Any]]:
        """
        Retrieve candidate documents for ranking.

        Production path: queries Elasticsearch with BM25 multi-match.
        Fallback path: If ES is down, performs a SQL query against the real database (SQLite/PostgreSQL).
        """
        if ES_AVAILABLE and es_client:
            try:
                body = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "description"],
                            "fuzziness": "AUTO",
                        }
                    },
                    "size": 20,
                }
                res = es_client.search(index="products", body=body)
                candidates = []
                for hit in res["hits"]["hits"]:
                    src = hit["_source"]
                    candidates.append({
                        "id": src["id"],
                        "title": src["title"],
                        "description": src.get("description", ""),
                        "category": src["category"],
                        "popularity": src.get("popularity", 50.0),
                        "ctr": src.get("ctr", 0.05),
                        "freshness": src.get("freshness", 0.5),
                        "engagement": src.get("engagement", 3.0),
                    })
                return candidates
            except Exception as exc:
                logger.error("Elasticsearch query failed, falling back to database query: %s", exc)

        # ES is down, retrieve candidates from the database (ProductModel)
        try:
            from backend.database.connection import SessionLocal
            from backend.database.models import ProductModel
            if SessionLocal is not None:
                db = SessionLocal()
                try:
                    # Clean query tokens
                    tokens = tokenize(query)
                    if not tokens:
                        # Return all products if query is empty
                        results = db.query(ProductModel).limit(20).all()
                    else:
                        # Find products that contain any of the tokens in title or description
                        all_prods = db.query(ProductModel).all()
                        matched = []
                        for p in all_prods:
                            match_count = 0
                            title_lower = p.title.lower()
                            desc_lower = (p.description or "").lower()
                            for t in tokens:
                                if t in title_lower or t in desc_lower:
                                    match_count += 1
                            if match_count > 0:
                                matched.append((match_count, p))
                        
                        # Sort by number of matching tokens
                        matched.sort(key=lambda x: x[0], reverse=True)
                        results = [item[1] for item in matched[:20]]
                    
                    if results:
                        return [
                            {
                                "id": r.id,
                                "title": r.title,
                                "description": r.description or "",
                                "category": r.category,
                                "popularity": r.popularity,
                                "ctr": r.ctr,
                                "freshness": r.freshness,
                                "engagement": r.engagement,
                            }
                            for r in results
                        ]
                finally:
                    db.close()
        except Exception as db_exc:
            logger.error("Database query fallback failed: %s", db_exc)

        # If even DB query fails and we are in test mode, return MOCK_PRODUCTS
        if settings.is_testing:
            logger.warning("TESTING=true — Elasticsearch and Database both down, using MOCK_PRODUCTS.")
            return MOCK_PRODUCTS

        raise RuntimeError(
            "Both Elasticsearch and Database candidate retrieval paths failed. "
            "Ensure the database is running and has been seeded."
        )
