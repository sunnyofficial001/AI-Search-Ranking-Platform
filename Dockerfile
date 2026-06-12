# ─────────────────────────────────────────────
# Stage 1: Builder — install all dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer cache optimization)
COPY backend/requirements.txt .

# Install Python dependencies into a prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime — minimal production image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ml-platform-team"
LABEL org.opencontainers.image.title="AI Search & Recommendation Platform"
LABEL org.opencontainers.image.description="Production ML search ranking and recommender system"
LABEL org.opencontainers.image.version="2.0.0"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user (security best practice)
RUN groupadd --gid 1001 mlapp && \
    useradd --uid 1001 --gid mlapp --shell /bin/bash --create-home mlapp

WORKDIR /app

# Copy application code (ordered from least to most frequently changed)
COPY --chown=mlapp:mlapp backend/ ./backend/
COPY --chown=mlapp:mlapp tests/ ./tests/
COPY --chown=mlapp:mlapp alembic.ini ./
COPY --chown=mlapp:mlapp .env.example ./.env.example

# Create directories for runtime artifacts
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R mlapp:mlapp /app/models /app/logs /app/data

# Switch to non-root user
USER mlapp

# Environment defaults (override at runtime via -e or Kubernetes secrets)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    WORKERS=4 \
    ENVIRONMENT=production \
    LOG_LEVEL=info \
    DATABASE_URL=sqlite:///./production.db \
    REDIS_URL=redis://redis:6379

EXPOSE 8000

# Health check — uses /api/v1/health (fast, no DB dependency)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/api/v1/health || exit 1

# Start with Gunicorn + Uvicorn workers for production
CMD ["sh", "-c", \
    "gunicorn backend.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WORKERS} \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL} \
    --forwarded-allow-ips='*'"]
