# Dockerfile — Ember Energy Pipeline
# Multi-stage build: builder installs deps, runner is lean

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix (isolated from system)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runner ───────────────────────────────────────────────────────────
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/           ./src/
COPY pipeline.py    ./pipeline.py
COPY pipeline_cache.py ./pipeline_cache.py
COPY params.yaml    ./params.yaml

# Create output directories
RUN mkdir -p \
    outputs/eda/figures \
    outputs/preprocessing/figures \
    outputs/modeling/figures \
    outputs/forecasting/figures \
    outputs/deeplearning/figures \
    data/raw \
    .cache/pipeline

# Non-root user for security
RUN useradd -m -u 1000 ember
RUN chown -R ember:ember /app
USER ember

# Environment defaults (overridden by docker run -e or .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OUTPUTS_DIR=outputs \
    TRAIN_UNTIL=2019 \
    FORECAST_UNTIL=2025

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
