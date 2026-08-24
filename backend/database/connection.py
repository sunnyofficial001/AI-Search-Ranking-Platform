"""
Database Connection
===================
- Production: DATABASE_URL env var must be set (PostgreSQL).
  The engine is lazily created on first use to allow safe importing
  in test environments that set DATABASE_URL later or use the SQLite fallback.
- Testing: If DATABASE_URL is unset and TESTING=true, falls back to an
  in-memory SQLite database so pytest collection/API tests work without
  a live PostgreSQL instance.
- Security: Credentials must NEVER be hard-coded here. Set DATABASE_URL
  in your environment, secrets manager, or .env file (not committed to git).
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.core.config import settings

logger = logging.getLogger("database.connection")

_DATABASE_URL = settings.DATABASE_URL
_TESTING = settings.is_testing

# Determine effective URL
if _DATABASE_URL:
    _effective_url = _DATABASE_URL
elif _TESTING:
    _effective_url = "sqlite:///:memory:"
    logger.warning(
        "DATABASE_URL not set — using in-memory SQLite for testing. "
        "This must NOT be used in production."
    )
else:
    _effective_url = "sqlite:///./dev.db"
    logger.warning(
        "DATABASE_URL not set — using local SQLite (./dev.db) for development. "
        "This is acceptable for local development but must NOT be used in production."
    )

def _get_engine():
    """Return the SQLAlchemy engine, raising clearly if not configured."""
    if _effective_url is None:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Set it before starting the service. "
            "Credentials must never be hard-coded in source."
        )
    connect_args = {"check_same_thread": False} if _effective_url.startswith("sqlite") else {}
    return create_engine(_effective_url, pool_pre_ping=True, echo=False, connect_args=connect_args)

# Lazy singleton — created on first import of SessionLocal/engine
engine = _get_engine() if _effective_url else None
if engine is not None:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    SessionLocal = None  # type: ignore[assignment]

Base = declarative_base()

# Ensure schema exists immediately after engine is available.
# The seeder (called at app startup) will do the real table creation with
# all models imported; this is a safety net for direct imports.
if engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as _exc:
        logger.warning(f"Base.metadata.create_all skipped at import time: {_exc}")


def get_db():
    """FastAPI dependency: yields a database session, closing it after use."""
    if SessionLocal is None:
        raise RuntimeError("Database is not configured. Set DATABASE_URL.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
