"""
Database Seeder
===============
Idempotent startup seeder. Creates tables (via SQLAlchemy metadata) and populates
canonical seed rows if the tables are empty.

Design rules:
- Idempotent: safe to call on every startup — checks before inserting.
- Deterministic: all seeds are constants drawn from ITEM_CATALOG / ITEM_BY_ID.
- No random data: every field is a real, fixed value.
- Test-safe: skips gracefully if engine is None (in-memory SQLite test sessions
  manage their own schema via the same call path, triggered by conftest fixtures).
"""

import logging
from typing import Optional

logger = logging.getLogger("database.seeder")

# Canonical product catalog — single source of truth shared with advanced_recommend.py
_SEED_PRODUCTS = [
    {
        "id": "prod-1",
        "title": "Amazon Echo Dot (5th Gen) - Smart Speaker Alexa",
        "description": (
            "Our most popular smart speaker with Alexa features a sleek compact design "
            "and delivers vibrant sound. Play music, check weather, control smart devices."
        ),
        "category": "Electronics",
        "popularity": 92.0,
        "ctr": 0.12,
        "freshness": 0.85,
        "engagement": 4.6,
    },
    {
        "id": "prod-2",
        "title": "Fjallraven Kanken Classic Minimalist Backpack",
        "description": (
            "Classic Kanken backpack in hard-wearing Vinylon fabric with a zip that opens "
            "the entire main compartment. Ergonomic straps, removable seat pad, and handles."
        ),
        "category": "Apparel",
        "popularity": 88.0,
        "ctr": 0.08,
        "freshness": 0.60,
        "engagement": 4.4,
    },
    {
        "id": "prod-3",
        "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "description": (
            "Industry leading noise canceling headphones with custom sound engineering, "
            "pristine microphone, 30 hours battery life, and Alexa Google Voice integrated."
        ),
        "category": "Electronics",
        "popularity": 95.0,
        "ctr": 0.15,
        "freshness": 0.90,
        "engagement": 4.8,
    },
    {
        "id": "prod-4",
        "title": "Levi's Men's 511 Slim Fit Jeans Stretch Denim",
        "description": (
            "A modern slim with room to move. The 511 slim fit denim is a classic since "
            "inception. Crafted from premium stretch cotton for ultimate active comfort."
        ),
        "category": "Apparel",
        "popularity": 79.0,
        "ctr": 0.05,
        "freshness": 0.40,
        "engagement": 4.1,
    },
    {
        "id": "prod-5",
        "title": "Anker USB-C Charger Nano 30W Super Fast Charging",
        "description": (
            "Extremely compact GaN II fast wall charger block for iPhone, Galaxy, iPad, "
            "MacBook Air. MultiProtect safety features, fold-away plug design."
        ),
        "category": "Electronics",
        "popularity": 91.0,
        "ctr": 0.18,
        "freshness": 0.95,
        "engagement": 4.7,
    },
    {
        "id": "prod-6",
        "title": "Apple AirPods Pro (2nd Gen) with USB-C",
        "description": (
            "Re-engineered noise cancellation, adaptive audio transparency mode, spatial "
            "personalized high-fidelity audio, and extra comfortable silicone ear tips."
        ),
        "category": "Electronics",
        "popularity": 98.0,
        "ctr": 0.22,
        "freshness": 0.92,
        "engagement": 4.9,
    },
    {
        "id": "prod-7",
        "title": "Nike Men's Air Zoom Pegasus Running Shoes",
        "description": (
            "The classic Pegasus returns. Designed for neutral running and high physical "
            "elasticity. Breathable mesh top, double Zoom Air chambers."
        ),
        "category": "Footwear",
        "popularity": 84.0,
        "ctr": 0.07,
        "freshness": 0.70,
        "engagement": 4.3,
    },
    {
        "id": "prod-8",
        "title": "The Alchemist - Original Hardcover Fiction",
        "description": (
            "Fable about following your dreams, listening to your gut, and finding "
            "treasure. Over 65 million copies sold globally. A timeless inspirational text."
        ),
        "category": "Books",
        "popularity": 75.0,
        "ctr": 0.04,
        "freshness": 0.20,
        "engagement": 4.5,
    },
    {
        "id": "prod-9",
        "title": "Asus ROG Zephyrus G14 Gaming Laptop RTX 4060",
        "description": (
            "Top performing 14-inch Windows system. AMD Ryzen 9 processor, NVIDIA RTX 4060 "
            "GPU, Nebula HDR Display, sleek white chassis with animatrix design."
        ),
        "category": "Electronics",
        "popularity": 86.0,
        "ctr": 0.10,
        "freshness": 0.88,
        "engagement": 4.5,
    },
    {
        "id": "prod-10",
        "title": "Stan Smith Ortholite Recycled Clean Sneakers",
        "description": (
            "Sustainable clean court sneakers made from Primegreen high performance recycled "
            "materials. White leather core paired with classic green details."
        ),
        "category": "Footwear",
        "popularity": 81.0,
        "ctr": 0.06,
        "freshness": 0.50,
        "engagement": 4.2,
    },
]

_SEED_USERS = [
    {"id": "user-1", "username": "alice_m",   "email": "alice@example.com"},
    {"id": "user-2", "username": "bob_k",     "email": "bob@example.com"},
    {"id": "user-3", "username": "carol_r",   "email": "carol@example.com"},
    {"id": "user-4", "username": "david_s",   "email": "david@example.com"},
    {"id": "user-5", "username": "eve_t",     "email": "eve@example.com"},
]


def seed_database(engine=None) -> None:
    """
    Create all SQLAlchemy tables and seed canonical rows if empty.

    Args:
        engine: Optional SQLAlchemy engine. If None, imports from connection module.
    """
    if engine is None:
        try:
            from backend.database.connection import engine as _engine
            engine = _engine
        except Exception as exc:
            logger.warning(f"Database engine unavailable — skipping seeder: {exc}")
            return

    if engine is None:
        logger.warning("Database engine is None — skipping seeder.")
        return

    # Import models so their metadata is registered with Base
    from backend.database.models import (  # noqa: F401
        Base, UserModel, ProductModel, DocumentModel,
        QueryModel, ClickLogModel, RecommendationModel, ExperimentModel,
    )

    # 1. Create all tables (idempotent — skips if already exist)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema ensured (tables created or verified).")
    except Exception as exc:
        logger.error(f"Failed to create database schema: {exc}")
        raise

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        # 2. Seed users
        existing_users = session.query(UserModel).count()
        if existing_users == 0:
            for u in _SEED_USERS:
                session.add(UserModel(id=u["id"], username=u["username"], email=u["email"]))
            session.commit()
            logger.info(f"Seeded {len(_SEED_USERS)} canonical users.")
        else:
            logger.debug(f"Users table already has {existing_users} rows — skipping user seed.")

        # 3. Seed products
        existing_products = session.query(ProductModel).count()
        if existing_products == 0:
            for p in _SEED_PRODUCTS:
                session.add(ProductModel(
                    id=p["id"],
                    title=p["title"],
                    description=p["description"],
                    category=p["category"],
                    popularity=p["popularity"],
                    ctr=p["ctr"],
                    freshness=p["freshness"],
                    engagement=p["engagement"],
                ))
            session.commit()
            logger.info(f"Seeded {len(_SEED_PRODUCTS)} canonical products.")
        else:
            logger.debug(f"Products table already has {existing_products} rows — skipping product seed.")

    logger.info("Database seeder complete.")
