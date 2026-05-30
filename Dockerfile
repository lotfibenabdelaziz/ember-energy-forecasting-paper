# Dockerfile — Ember Energy Pipeline
# Multi-stage build: builder installs deps, runner is lean

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y curl

# Install torch CPU separately (heaviest package — cache independently)
RUN pip install --prefix=/install --no-cache-dir \
    --retries 5 --timeout 120 \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else
RUN pip install --prefix=/install --no-cache-dir \
    --retries 5 --timeout 120 \
    -r requirements-docker.txt

# ── Stage 2: runner ───────────────────────────────────────────────────────────
FROM python:3.10-slim AS runner

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/              ./src/
COPY api/              ./api/
COPY pipeline.py       ./pipeline.py
COPY pipeline_cache.py ./pipeline_cache.py
COPY params.yaml       ./params.yaml

RUN mkdir -p \
    outputs/eda/figures \
    outputs/preprocessing/figures \
    outputs/modeling/figures \
    outputs/forecasting/figures \
    outputs/deeplearning/figures \
    data/raw \
    .cache/pipeline

RUN useradd -m -u 1000 ember \
 && chown -R ember:ember /app
USER ember

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OUTPUTS_DIR=outputs \
    OUTPUT_ROOT=outputs \
    TRAIN_UNTIL=2016 \
    FORECAST_UNTIL=2030

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]
