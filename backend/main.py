"""
FastAPI Application Entry Point
================================
Production-grade startup sequence:
  1. CORS configured with explicit origins (not wildcard in production).
  2. MonitoringMiddleware attached for Prometheus metrics + correlation IDs.
  3. Startup event seeds the database (idempotent — safe on every restart).
  4. Custom PlatformException handler for structured error responses.
"""

from contextlib import asynccontextmanager
import logging
import uvicorn

# IMPORTANT: import lightgbm FIRST to prevent Windows OpenMP DLL conflict
# (lightgbm must load its OpenMP runtime before xgboost/torch do)
try:
    import lightgbm as _lgb  # noqa: F401
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_v2 import router as api_router
from backend.core import settings, PlatformException

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run database schema creation and canonical seeding on every startup."""
    logger.info("=== AI Search Platform startup ===")
    try:
        from backend.database.seeder import seed_database
        seed_database()
        logger.info("Database ready.")
    except Exception as exc:
        # Log but do not abort startup — the DB may not be needed for all routes.
        logger.error(f"Database seeding failed: {exc}")
    yield


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Search & Ranking Platform",
    description=(
        "Production-grade Learning-to-Rank (LTR) retrieval and hybrid recommendation "
        "engine. Implements LambdaMART, RankNet, XGBoost ensemble ranking with "
        "SHAP explainability, drift detection, and A/B testing."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — restrict origins in production; allow all only in dev/test
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = (
    ["*"]
    if not settings.is_production
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Monitoring middleware (Prometheus metrics + correlation IDs)
# ---------------------------------------------------------------------------

try:
    from backend.monitoring.monitoring import MonitoringMiddleware
    app.add_middleware(MonitoringMiddleware)
    logger.info("MonitoringMiddleware registered.")
except Exception as _exc:
    logger.warning(f"MonitoringMiddleware could not be loaded: {_exc}")

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(PlatformException)
async def platform_exception_handler(request: Request, exc: PlatformException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Convert explicit RuntimeErrors (e.g. ES unavailable) into 503 responses."""
    logger.error(f"RuntimeError on {request.url}: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": str(exc), "status_code": 503, "details": {}},
    )





# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"])
def root_endpoint():
    return {
        "service": "AI Search & Ranking Platform",
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT,
        "documentation": "/docs",
        "health": "/api/v1/health",
    }


# ---------------------------------------------------------------------------
# Register API router
# ---------------------------------------------------------------------------

app.include_router(api_router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=not settings.is_production,
        log_level=settings.LOG_LEVEL.lower(),
    )
