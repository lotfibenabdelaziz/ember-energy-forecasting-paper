# =============================================================================
# Makefile — Ember Energy Forecasting Pipeline
# IEEE Paper | CI/CD + MLflow
# Cross-platform: Windows (PowerShell) + Linux/macOS (bash)
# =============================================================================

# ── Detect OS and configure shell ────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    SHELL        := powershell.exe
    .SHELLFLAGS  := -NoProfile -NonInteractive -Command
    RM           := Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    MKDIR        := New-Item -ItemType Directory -Force -Path
    SEP          := ;
    DEVNULL      := 2>$$null
    IS_WIN       := 1
else
    SHELL        := /bin/bash
    .SHELLFLAGS  := -euo pipefail -c
    RM           := rm -rf
    MKDIR        := mkdir -p
    SEP          := :
    DEVNULL      := 2>/dev/null
    IS_WIN       :=
endif

export PATH := $(PATH)$(SEP)C:/Program Files/Docker/Docker/resources/bin
PYTEST_OPTS = -p no:cacheprovider
# ── Image config ──────────────────────────────────────────────────────────────
IMAGE_NAME  ?= ember-energy-pipeline
IMAGE_TAG   ?= latest
REGISTRY    ?= ghcr.io/lotfibenabdelaziz
FULL_IMAGE   = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Pipeline config ───────────────────────────────────────────────────────────
CSV_PATH       ?= ./data/raw/yearly_full_release_long_format.csv
TRAIN_UNTIL    ?= 2016
FORECAST_UNTIL ?= 2030
NAMESPACE      ?= ember-pipeline
MLFLOW_PORT    ?= 5000
PORT           ?= 8000

PYTHON ?= python
PIP    ?= pip

# ── Environment helpers ───────────────────────────────────────────────────────
ifdef IS_WIN
    SET_MLFLOW  = $$env:MLFLOW_TRACKING_URI='http://localhost:$(MLFLOW_PORT)';
    SET_DEV_ENV = $$env:ENV='development'; $$env:OUTPUT_ROOT='outputs';
else
    SET_MLFLOW  = MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT)
    SET_DEV_ENV = ENV=development OUTPUT_ROOT=outputs
endif

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: help install api-install lint lint-fix format typecheck check \
	    run run-eda run-preprocessing run-modeling run-forecasting run-deeplearning \
	    rerun-eda rerun-preprocessing rerun-modeling rerun-forecasting rerun-deeplearning \
	    test test-fast test-unit test-dl test-api test-pipeline test-parity \
	    test-parity-full test-cov \
	    api-run \
	    mlflow-ui mlflow-list mlflow-clean \
	    cache-status cache-clear \
	    cache-invalidate-eda cache-invalidate-preprocessing \
	    cache-invalidate-modeling cache-invalidate-forecasting \
	    cache-invalidate-deeplearning run-force \
	    docker-build docker-push docker-run \
	    compose-up compose-pipeline compose-jupyter compose-logs compose-down \
	    k8s-apply k8s-run-pipeline k8s-status k8s-logs k8s-delete k8s-secret-gen \
	    dvc-init dvc-remote-add dvc-add-data dvc-pull dvc-push \
	    dvc-repro dvc-repro-eda dvc-repro-preprocessing dvc-repro-modeling \
	    dvc-repro-forecasting dvc-repro-deeplearning \
	    dvc-dag dvc-status dvc-params dvc-metrics dvc-plots dvc-gc dvc-cache-info \
	    clean clean-all

.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════╗"
	@echo "  ║      Ember Energy Forecasting Pipeline               ║"
	@echo "  ║      IEEE Paper  |  CI/CD + MLflow                   ║"
	@echo "  ╚══════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ── LOCAL PIPELINE ───────────────────────────────────────"
	@echo "    make install                Install all dependencies"
	@echo "    make run                    Full pipeline (smart cache)"
	@echo "    make run-force              Full pipeline (ignore cache)"
	@echo "    make run-eda                EDA step only"
	@echo "    make run-preprocessing      Preprocessing step only"
	@echo "    make run-modeling           Modeling step only"
	@echo "    make run-forecasting        Forecasting step only"
	@echo "    make run-deeplearning       Deep learning step only"
	@echo ""
	@echo "  ── CACHE ────────────────────────────────────────────────"
	@echo "    make cache-status                    Show cached steps"
	@echo "    make cache-clear                     Wipe entire cache"
	@echo "    make cache-invalidate-eda            Invalidate EDA cache"
	@echo "    make cache-invalidate-preprocessing  Invalidate preprocessing"
	@echo "    make cache-invalidate-modeling       Invalidate modeling"
	@echo "    make cache-invalidate-forecasting    Invalidate forecasting"
	@echo "    make cache-invalidate-deeplearning   Invalidate deep learning"
	@echo ""
	@echo "  ── TESTS ────────────────────────────────────────────────"
	@echo "    make test                   Full suite + coverage"
	@echo "    make test-fast              Unit tests, stop on first fail"
	@echo "    make test-unit              Unit tests only"
	@echo "    make test-dl                Deep learning tests"
	@echo "    make test-api               FastAPI endpoint tests"
	@echo "    make test-pipeline          Integration tests"
	@echo "    make test-parity            Notebook vs script parity"
	@echo "    make test-cov               Coverage report (no API/pipeline)"
	@echo ""
	@echo "  ── CODE QUALITY ─────────────────────────────────────────"
	@echo "    make lint                   flake8 checks"
	@echo "    make format                 black + isort auto-format"
	@echo "    make typecheck              mypy static analysis"
	@echo "    make check                  lint + typecheck + test-fast"
	@echo ""
	@echo "  ── API ──────────────────────────────────────────────────"
	@echo "    make api-run                FastAPI dev server (port $(PORT))"
	@echo ""
	@echo "  ── MLFLOW ───────────────────────────────────────────────"
	@echo "    make mlflow-ui              MLflow UI (port $(MLFLOW_PORT))"
	@echo "    make mlflow-list            List all experiment runs"
	@echo "    make mlflow-clean           Delete local mlruns/"
	@echo ""
	@echo "  ── DVC ──────────────────────────────────────────────────"
	@echo "    make dvc-init               Initialise DVC in repo"
	@echo "    make dvc-add-data           Track raw CSV with DVC"
	@echo "    make dvc-pull               Pull data from remote"
	@echo "    make dvc-push               Push data to remote"
	@echo "    make dvc-repro              Reproduce full DVC pipeline"
	@echo "    make dvc-status             Show pipeline status"
	@echo "    make dvc-metrics            Show + diff metrics"
	@echo "    make dvc-dag                Print pipeline DAG"
	@echo ""
	@echo "  ── DOCKER ───────────────────────────────────────────────"
	@echo "    make docker-build           Build pipeline + API images"
	@echo "    make docker-push            Push images to ghcr.io"
	@echo "    make docker-run             Run pipeline in container"
	@echo ""
	@echo "  ── COMPOSE ──────────────────────────────────────────────"
	@echo "    make compose-up             Start API + MLflow"
	@echo "    make compose-pipeline       Run pipeline via compose"
	@echo "    make compose-jupyter        Jupyter Lab (dev profile)"
	@echo "    make compose-logs           Stream logs"
	@echo "    make compose-down           Stop all services"
	@echo ""
	@echo "  ── KUBERNETES ───────────────────────────────────────────"
	@echo "    make k8s-apply              Apply all manifests"
	@echo "    make k8s-run-pipeline       Submit pipeline Job"
	@echo "    make k8s-status             Jobs + Pods + PVCs"
	@echo "    make k8s-logs               Stream pod logs"
	@echo "    make k8s-delete             Tear down namespace"
	@echo ""
	@echo "  ── CLEANUP ──────────────────────────────────────────────"
	@echo "    make clean                  Remove outputs/"
	@echo "    make clean-all              Remove outputs/ + mlruns/ + cache"
	@echo ""

# =============================================================================
# INSTALL
# =============================================================================
install:
	@echo "── [install] Upgrading pip…"
	$(PIP) install --upgrade pip
	@echo "── [install] Installing requirements.txt…"
	$(PIP) install -r requirements.txt
	@echo "✓  Dependencies installed."

api-install:
	@echo "── [api-install] Installing API + dev extras…"
	$(PIP) install -e ".[api,langchain,dev]"
	@echo "✓  API dependencies installed."

# =============================================================================
# PIPELINE — local runs
# =============================================================================
all: run

# ── Full pipeline (smart cache) ───────────────────────────────────────────────
run: $(CSV_PATH)
	@echo "── [run] Starting full pipeline (smart cache)…"
	$(SET_MLFLOW) $(PYTHON) pipeline.py --csv $(CSV_PATH) --train_until $(TRAIN_UNTIL) --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Pipeline complete."

# ── Full pipeline (cache disabled) ───────────────────────────────────────────
run-force: $(CSV_PATH)
	@echo "── [run-force] Starting full pipeline (cache disabled)…"
	$(PYTHON) pipeline.py --csv $(CSV_PATH) --train_until $(TRAIN_UNTIL) --forecast_until $(FORECAST_UNTIL) --force
	@echo "✓  Pipeline complete (forced)."

# ── Individual steps ──────────────────────────────────────────────────────────
run-eda: $(CSV_PATH)
	@echo "── [run-eda] Running EDA step…"
	$(PYTHON) src/01_eda.py --csv $(CSV_PATH) --output_dir outputs/eda
	@echo "✓  EDA complete → outputs/eda/"

run-preprocessing:
	@echo "── [run-preprocessing] Running preprocessing step…"
	$(PYTHON) src/02_preprocessing.py --input_dir outputs/eda --output_dir outputs/preprocessing --train_until $(TRAIN_UNTIL)
	@echo "✓  Preprocessing complete → outputs/preprocessing/"

run-modeling:
	@echo "── [run-modeling] Running modeling step…"
	$(SET_MLFLOW) $(PYTHON) src/03_modeling.py --input_dir outputs/preprocessing --output_dir outputs/modeling
	@echo "✓  Modeling complete → outputs/modeling/"

run-forecasting:
	@echo "── [run-forecasting] Running forecasting step…"
	$(SET_MLFLOW) $(PYTHON) src/04_forecasting.py --model_dir outputs/modeling --pre_dir outputs/preprocessing --output_dir outputs/forecasting --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Forecasting complete → outputs/forecasting/"

run-deeplearning:
	@echo "── [run-deeplearning] Running deep learning step (quick mode)…"
	$(PYTHON) src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning --quick
	@echo "✓  Deep learning complete → outputs/deeplearning/"

# ── Re-run steps (always executes, no cache check) ───────────────────────────
rerun-eda:
	@echo "── [rerun-eda] Re-running EDA (cache bypassed)…"
	$(PYTHON) src/01_eda.py --csv $(CSV_PATH) --output_dir outputs/eda
	@echo "✓  EDA complete → outputs/eda/"

rerun-preprocessing:
	@echo "── [rerun-preprocessing] Re-running preprocessing (cache bypassed)…"
	$(PYTHON) src/02_preprocessing.py --input_dir outputs/eda --output_dir outputs/preprocessing --train_until $(TRAIN_UNTIL)
	@echo "✓  Preprocessing complete → outputs/preprocessing/"

rerun-modeling:
	@echo "── [rerun-modeling] Re-running modeling…"
	$(PYTHON) src/03_modeling.py --input_dir outputs/preprocessing --output_dir outputs/modeling
	@echo "✓  Modeling complete → outputs/modeling/"

rerun-forecasting:
	@echo "── [rerun-forecasting] Re-running forecasting…"
	$(PYTHON) src/04_forecasting.py --pre_dir outputs/preprocessing --model_dir outputs/modeling --output_dir outputs/forecasting --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Forecasting complete → outputs/forecasting/"

rerun-deeplearning:
	@echo "── [rerun-deeplearning] Re-running deep learning (full epochs)…"
	$(PYTHON) src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning
	@echo "✓  Deep learning complete → outputs/deeplearning/"

# =============================================================================
# CACHE
# =============================================================================
cache-status:
	@echo "── [cache] Querying pipeline cache status…"
	$(PYTHON) pipeline.py --cache-status
	@echo "✓  Cache status displayed."

cache-clear:
	@echo "── [cache] Clearing entire pipeline cache…"
	$(PYTHON) pipeline.py --invalidate ALL
	@echo "✓  All cache entries cleared."

cache-invalidate-eda:
	@echo "── [cache] Invalidating EDA cache entry…"
	$(PYTHON) pipeline.py --invalidate eda
	@echo "✓  EDA cache invalidated — step will re-run next 'make run'."

cache-invalidate-preprocessing:
	@echo "── [cache] Invalidating preprocessing cache entry…"
	$(PYTHON) pipeline.py --invalidate preprocessing
	@echo "✓  Preprocessing cache invalidated."

cache-invalidate-modeling:
	@echo "── [cache] Invalidating modeling cache entry…"
	$(PYTHON) pipeline.py --invalidate modeling
	@echo "✓  Modeling cache invalidated."

cache-invalidate-forecasting:
	@echo "── [cache] Invalidating forecasting cache entry…"
	$(PYTHON) pipeline.py --invalidate forecasting
	@echo "✓  Forecasting cache invalidated."

cache-invalidate-deeplearning:
	@echo "── [cache] Invalidating deep learning cache entry…"
	$(PYTHON) pipeline.py --invalidate deeplearning
	@echo "✓  Deep learning cache invalidated."

# =============================================================================
# TESTS
# =============================================================================
test:
	@echo "── [test] Running full test suite with coverage…"
	pytest tests/ $(PYTEST_OPTS) -v --tb=short \
	    --cov=src --cov=api --cov=pipeline --cov=mlflow_config \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-report=xml:coverage.xml \
	    --cov-config=.coveragerc
	@echo "✓  Tests complete. HTML report → htmlcov/index.html"

test-fast:
	@echo "── [test-fast] Running fast unit tests (stop on first failure)…"
	pytest tests/ $(PYTEST_OPTS) -v --tb=short -x -q \
	    --ignore=tests/test_api.py \
	    --ignore=tests/test_pipeline.py \
	    --ignore=tests/test_deeplearning.py \
	    --ignore=tests/test_parity.py
	@echo "✓  Fast tests complete."

test-unit:
	@echo "── [test-unit] Running unit tests…"
	pytest tests/test_eda.py \
	       tests/test_preprocessing.py \
	       tests/test_modeling.py \
	       tests/test_forecasting.py \
	       tests/test_mlflow_config.py \
	       tests/test_deeplearning.py \
	    -v --tb=short
	@echo "✓  Unit tests complete."

test-dl:
	@echo "── [test-dl] Running deep learning tests…"
	pytest tests/test_deeplearning.py -v --tb=short
	@echo "✓  Deep learning tests complete."

test-api:
	@echo "── [test-api] Running FastAPI endpoint tests…"
	pytest tests/test_api.py -v --tb=short
	@echo "✓  API tests complete."

test-pipeline:
	@echo "── [test-pipeline] Running integration tests…"
	pytest tests/test_pipeline.py -v --tb=short
	@echo "✓  Integration tests complete."

test-parity:
	@echo "── [test-parity] Running notebook vs script parity tests…"
	pytest tests/test_parity.py -v --tb=short
	@echo "✓  Parity tests complete."

test-parity-full: run
	@echo "── [test-parity-full] Full parity check (pipeline re-run + parity tests)…"
	pytest tests/test_parity.py -v --tb=short
	@echo "✓  Full parity check complete."

test-cov:
	@echo "── [test-cov] Running coverage report (excluding API + pipeline tests)…"
	pytest tests/ -q \
	    --ignore=tests/test_api.py \
	    --ignore=tests/test_pipeline.py \
	    --cov=src --cov=pipeline --cov=mlflow_config \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-config=.coveragerc
	@echo "✓  Coverage report → htmlcov/index.html"

# =============================================================================
# CODE QUALITY — powered by RUFF
# =============================================================================
RUFF_TARGETS = src/ api/ pipeline.py pipeline_cache.py mlflow_config.py

lint:
	@echo "── [lint] Running ruff check…"
	ruff check $(RUFF_TARGETS)
	@echo "✓  Lint passed."

lint-fix:
	@echo "── [lint-fix] Auto-fixing with ruff…"
	ruff check --fix $(RUFF_TARGETS)
	@echo "── [lint-fix] Formatting with ruff format…"
	ruff format $(RUFF_TARGETS)
	@echo "✓  Code fixed and formatted."

format: lint-fix

format-check:
	@echo "── [format-check] Checking format without changes…"
	ruff format --check $(RUFF_TARGETS)
	ruff check $(RUFF_TARGETS)
	@echo "✓  Format check passed."

typecheck:
	@echo "── [typecheck] Running mypy…"
	mypy src/ api/ mlflow_config.py --config-file mypy.ini
	@echo "✓  Type check passed."

check: lint typecheck test-fast
	@echo "✓  All quality gates passed."

# =============================================================================
# API
# =============================================================================
api-run:
	@echo "── [api-run] Starting FastAPI development server…"
	@echo "   Docs → http://localhost:$(PORT)/docs"
	$(SET_DEV_ENV) uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

# =============================================================================
# MLFLOW
# =============================================================================
mlflow-ui:
	@echo "── [mlflow-ui] Starting MLflow tracking server…"
	@echo "   UI → http://127.0.0.1:$(MLFLOW_PORT)"
	mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) --backend-store-uri ./mlruns --default-artifact-root ./mlruns/artifacts

mlflow-list:
	@echo "── [mlflow-list] Listing runs in experiment 'ember-demand-forecasting'…"
	mlflow runs list --experiment-name ember-demand-forecasting
	@echo "✓  Run list complete."

mlflow-clean:
	@echo "── [mlflow-clean] Deleting local mlruns/ directory…"
	python -c "import shutil; shutil.rmtree('mlruns', ignore_errors=True)"
	@echo "✓  mlruns/ removed."

# =============================================================================
# DVC — Data Version Control
# =============================================================================

# ── Setup ─────────────────────────────────────────────────────────────────────
dvc-init:
	@echo "── [dvc-init] Initialising DVC in repository…"
	dvc init
	dvc config core.autostage true
	dvc config core.analytics false
	@echo "✓  DVC initialised. Next: make dvc-remote-add"

dvc-remote-add:
	@echo "── [dvc-remote-add] Edit .dvc/config and uncomment your preferred remote, then run:"
	@echo "     dvc remote default <name>"

# ── Data ──────────────────────────────────────────────────────────────────────
dvc-add-data:
	@echo "── [dvc-add-data] Tracking raw CSV with DVC…"
	dvc add data/raw/yearly_full_release_long_format.csv
	@echo "✓  Data file tracked. Commit the generated .dvc pointer file."

dvc-pull:
	@echo "── [dvc-pull] Pulling data and outputs from DVC remote…"
	dvc pull
	@echo "✓  Pull complete."

dvc-push:
	@echo "── [dvc-push] Pushing data and outputs to DVC remote…"
	dvc push
	@echo "✓  Push complete."

# ── Pipeline ──────────────────────────────────────────────────────────────────
dvc-repro:
	@echo "── [dvc-repro] Reproducing full DVC pipeline…"
	dvc repro
	@echo "✓  Pipeline reproduced. Run 'make dvc-push' to cache outputs."

dvc-repro-eda:
	@echo "── [dvc-repro-eda] Reproducing EDA stage…"
	dvc repro eda
	@echo "✓  EDA stage reproduced."

dvc-repro-preprocessing:
	@echo "── [dvc-repro-preprocessing] Reproducing preprocessing stage…"
	dvc repro preprocessing
	@echo "✓  Preprocessing stage reproduced."

dvc-repro-modeling:
	@echo "── [dvc-repro-modeling] Reproducing modeling stage…"
	dvc repro modeling
	@echo "✓  Modeling stage reproduced."

dvc-repro-forecasting:
	@echo "── [dvc-repro-forecasting] Reproducing forecasting stage…"
	dvc repro forecasting
	@echo "✓  Forecasting stage reproduced."

dvc-repro-deeplearning:
	@echo "── [dvc-repro-deeplearning] Reproducing deep learning stage…"
	dvc repro deeplearning
	@echo "✓  Deep learning stage reproduced."

# ── Inspection ────────────────────────────────────────────────────────────────
dvc-dag:
	@echo "── [dvc-dag] Printing pipeline DAG…"
	dvc dag

dvc-status:
	@echo "── [dvc-status] DVC pipeline status…"
	dvc status

dvc-params:
	@echo "── [dvc-params] Showing parameter diff…"
	dvc params diff

dvc-metrics:
	@echo "── [dvc-metrics] Showing metrics…"
	dvc metrics show
	dvc metrics diff

dvc-plots:
	@echo "── [dvc-plots] Generating DVC plots…"
	dvc plots show

# ── Cache ─────────────────────────────────────────────────────────────────────
dvc-gc:
	@echo "── [dvc-gc] Removing unused DVC cache entries…"
	dvc gc --workspace --force
	@echo "✓  Garbage collection complete."

dvc-cache-info:
	@echo "── [dvc-cache-info] Local DVC cache size:"
ifdef IS_WIN
	if (Test-Path .dvc/cache) { (Get-ChildItem .dvc/cache -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB } else { Write-Host "No local cache yet." }
else
	du -sh .dvc/cache $(DEVNULL) || echo "No local cache yet."
endif

# =============================================================================
# DOCKER
# =============================================================================
docker-build:
	@echo "── [docker-build] Building pipeline image: $(IMAGE_NAME):$(IMAGE_TAG)…"
	docker build -t $(FULL_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✓  Pipeline image built."
	@echo "── [docker-build] Building API image: ember-energy-api:$(IMAGE_TAG)…"
	docker build -f Dockerfile.api \
	    -t $(REGISTRY)/ember-energy-api:$(IMAGE_TAG) \
	    -t ember-energy-api:$(IMAGE_TAG) .
	@echo "✓  API image built."
	@echo "✓  Both images ready."

docker-push: docker-build
	@echo "── [docker-push] Pushing pipeline image to $(REGISTRY)…"
	docker push $(FULL_IMAGE)
	@echo "✓  Pipeline image pushed."
	@echo "── [docker-push] Pushing API image to $(REGISTRY)…"
	docker push $(REGISTRY)/ember-energy-api:$(IMAGE_TAG)
	@echo "✓  API image pushed."
	@echo "✓  Both images available at $(REGISTRY)"

docker-run: $(CSV_PATH)
	@echo "── [docker-run] Running pipeline inside Docker container…"
	docker run --rm \
	    -e MLFLOW_TRACKING_URI=http://host.docker.internal:$(MLFLOW_PORT) \
	    -v $(PWD)/data:/app/data:ro \
	    -v $(PWD)/outputs:/app/outputs \
	    -v $(PWD)/mlruns:/app/mlruns \
	    $(IMAGE_NAME):$(IMAGE_TAG) \
	    --csv /app/$(CSV_PATH) \
	    --train_until $(TRAIN_UNTIL) \
	    --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Pipeline container finished."

# =============================================================================
# DOCKER COMPOSE
# =============================================================================
compose-up:
	@echo "── [compose-up] Starting API + MLflow services (detached)…"
	docker compose up mlflow api --build -d
	@echo "✓  Services started."
	@echo "   API    → http://localhost:$(PORT)/docs"
	@echo "   MLflow → http://localhost:$(MLFLOW_PORT)"

compose-pipeline: $(CSV_PATH)
	@echo "── [compose-pipeline] Running pipeline via Docker Compose…"
	docker compose run --rm pipeline
	@echo "✓  Pipeline container finished."

compose-jupyter:
	@echo "── [compose-jupyter] Starting Jupyter Lab → http://localhost:8888"
	docker compose --profile dev up jupyter --build

compose-logs:
	@echo "── [compose-logs] Streaming Docker Compose logs (Ctrl+C to stop)…"
	docker compose logs --follow

compose-down:
	@echo "── [compose-down] Stopping all Compose services…"
	docker compose down --remove-orphans
	@echo "✓  All services stopped."

# =============================================================================
# KUBERNETES
# =============================================================================
k8s-apply: k8s-secret-gen
	@echo "── Applying Kubernetes manifests…"
	kubectl apply -f k8s/namespace.yaml
	@echo "  ✓ namespace"
	kubectl apply -f k8s/pvc.yaml
	@echo "  ✓ pvcs"
	kubectl apply -f k8s/configmap.yaml
	@echo "  ✓ configmap"
	kubectl apply -f k8s/secret.yaml
	@echo "  ✓ secret"
	kubectl apply -f k8s/deployment-mlflow.yaml
	@echo "  ✓ mlflow"
	kubectl apply -f k8s/deployment-api.yaml
	@echo "  ✓ api"
	kubectl apply -f k8s/hpa.yaml
	@echo "  ✓ hpa"
	kubectl apply -f k8s/cronjob.yaml
	@echo "  ✓ cronjob"
	@echo "✓  All manifests applied to namespace: $(NAMESPACE)"

k8s-run-pipeline:
	@echo "── [k8s-run-pipeline] Submitting pipeline Job to Kubernetes…"
	kubectl delete job ember-pipeline -n $(NAMESPACE) --ignore-not-found
	kubectl apply -f k8s/job-pipeline.yaml
	@echo "✓  Job submitted. Watch with: make k8s-logs"

k8s-status:
	@echo "── [k8s-status] Kubernetes resource status (namespace: $(NAMESPACE)):"
	kubectl get jobs,pods,pvc -n $(NAMESPACE)

k8s-logs:
	@echo "── [k8s-logs] Streaming pipeline pod logs (Ctrl+C to stop)…"
	kubectl logs -n $(NAMESPACE) -l app=ember-pipeline --follow --tail=100

k8s-delete:
	@echo "── [k8s-delete] Tearing down namespace $(NAMESPACE)…"
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	@echo "✓  Namespace $(NAMESPACE) deleted."

k8s-secret-gen:
	@echo "── Generating k8s/secret.yaml from .env…"
	$(PYTHON) k8s/gen_secret.py

# =============================================================================
# CLEANUP
# =============================================================================
clean:
	@echo "── [clean] Removing outputs/ directory…"
	$(RM) outputs/
	@echo "✓  outputs/ removed."

clean-all: clean mlflow-clean
	@echo "── [clean-all] Removing Python cache files and temp directories…"
ifdef IS_WIN
	Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
	Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
	Get-ChildItem -Recurse -Filter ".ipynb_checkpoints" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force .cache/ -ErrorAction SilentlyContinue
else
	find . -type d -name __pycache__ -exec rm -rf {} + $(DEVNULL) || true
	find . -name "*.pyc" -delete $(DEVNULL) || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + $(DEVNULL) || true
	$(RM) .cache/ $(DEVNULL) || true
endif
	@echo "✓  Full cleanup complete."

# =============================================================================
# DATA GUARD — fail fast if CSV is missing
# =============================================================================
$(CSV_PATH):
	@echo ""
	@echo "  ✗  ERROR: CSV data file not found at: $(CSV_PATH)"
	@echo "     Place the Ember yearly release CSV at that path and retry."
	@echo ""
	@exit 1
