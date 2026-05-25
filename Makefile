# =============================================================================
# Makefile — Ember Energy Forecasting Pipeline
# IEEE Paper | CI/CD + MLflow
# =============================================================================

# ── Windows compatibility ─────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    SHELL        := powershell.exe
    .SHELLFLAGS  := -NoProfile -Command
    RM           := Remove-Item -Recurse -Force
    MKDIR        := New-Item -ItemType Directory -Force
else
    RM           := rm -rf
    MKDIR        := mkdir -p
endif

export PATH := $(PATH):C:/Program Files/Docker/Docker/resources/bin

# ── Image config ──────────────────────────────────────────────────────────────
IMAGE_NAME ?= ember-energy-pipeline
IMAGE_TAG  ?= latest
REGISTRY   ?= ghcr.io/lotfibenabdelaziz
FULL_IMAGE  = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Pipeline config ───────────────────────────────────────────────────────────
CSV_PATH       ?= ./data/raw/yearly_full_release_long_format.csv
TRAIN_UNTIL    ?= 2016
FORECAST_UNTIL ?= 2030
NAMESPACE      ?= ember-pipeline
MLFLOW_PORT    ?= 5000
PORT           ?= 8000

PYTHON ?= python
PIP    ?= pip

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
        k8s-apply k8s-run-pipeline k8s-status k8s-logs k8s-delete \
        clean clean-all

.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================
help:
	@echo ""
	@echo "  Ember Energy Forecasting Pipeline"
	@echo "  ══════════════════════════════════════════════════════"
	@echo ""
	@echo "  LOCAL PIPELINE"
	@echo "    make install                Install all dependencies"
	@echo "    make run                    Full pipeline (smart cache)"
	@echo "    make run-force              Full pipeline (ignore cache)"
	@echo "    make run-eda                EDA step only"
	@echo "    make run-preprocessing      Preprocessing step only"
	@echo "    make run-modeling           Modeling step only"
	@echo "    make run-forecasting        Forecasting step only"
	@echo "    make run-deeplearning       Deep learning step only"
	@echo ""
	@echo "  CACHE"
	@echo "    make cache-status           Show cached steps"
	@echo "    make cache-clear            Wipe entire cache"
	@echo "    make cache-invalidate-eda   Force eda to re-run"
	@echo ""
	@echo "  TESTS"
	@echo "    make test                   Full suite + coverage"
	@echo "    make test-fast              Unit tests, stop on first fail"
	@echo "    make test-unit              Unit tests only"
	@echo "    make test-dl                Deep learning tests"
	@echo "    make test-api               FastAPI endpoint tests"
	@echo "    make test-pipeline          Integration tests"
	@echo "    make test-parity            Notebook vs script parity"
	@echo ""
	@echo "  CODE QUALITY"
	@echo "    make lint                   flake8"
	@echo "    make format                 black + isort"
	@echo "    make typecheck              mypy"
	@echo "    make check                  lint + typecheck + test-fast"
	@echo ""
	@echo "  API"
	@echo "    make api-run                FastAPI dev server (port 8000)"
	@echo ""
	@echo "  MLFLOW"
	@echo "    make mlflow-ui              MLflow UI (port 5000)"
	@echo "    make mlflow-list            List all runs"
	@echo "    make mlflow-clean           Delete local mlruns/"
	@echo ""
	@echo "  DOCKER"
	@echo "    make docker-build           Build pipeline + API images"
	@echo "    make docker-push            Push images to ghcr.io"
	@echo "    make docker-run             Run pipeline in container"
	@echo ""
	@echo "  COMPOSE"
	@echo "    make compose-up             Start API + MLflow"
	@echo "    make compose-pipeline       Run pipeline via compose"
	@echo "    make compose-jupyter        Jupyter Lab (dev)"
	@echo "    make compose-logs           Stream logs"
	@echo "    make compose-down           Stop all services"
	@echo ""
	@echo "  KUBERNETES"
	@echo "    make k8s-apply              Apply all manifests"
	@echo "    make k8s-run-pipeline       Submit pipeline Job"
	@echo "    make k8s-status             Jobs + Pods + PVCs"
	@echo "    make k8s-logs               Stream pod logs"
	@echo "    make k8s-delete             Tear down namespace"
	@echo ""
	@echo "  CLEANUP"
	@echo "    make clean                  Remove outputs/"
	@echo "    make clean-all              Remove outputs/ + mlruns/ + cache"
	@echo ""

# =============================================================================
# INSTALL
# =============================================================================
install:
	@echo "── Installing dependencies…"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed."

api-install:
	@echo "── Installing API + dev dependencies…"
	$(PIP) install -e ".[api,langchain,dev]"
	@echo "✓ API dependencies installed."

# =============================================================================
# PIPELINE — local runs
# =============================================================================
all: run

run: $(CSV_PATH)
	@echo "── Starting full pipeline (smart cache)…"
	$(PYTHON) pipeline.py \
		--csv            $(CSV_PATH) \
		--train_until    $(TRAIN_UNTIL) \
		--forecast_until $(FORECAST_UNTIL)
	@echo "✓ Pipeline complete."

run-force: $(CSV_PATH)
	@echo "── Starting full pipeline (cache disabled)…"
	$(PYTHON) pipeline.py \
		--csv            $(CSV_PATH) \
		--train_until    $(TRAIN_UNTIL) \
		--forecast_until $(FORECAST_UNTIL) \
		--force
	@echo "✓ Pipeline complete (forced)."

run-eda: $(CSV_PATH)
	@echo "── Running EDA step…"
	$(PYTHON) src/01_eda.py \
		--csv        $(CSV_PATH) \
		--output_dir outputs/eda
	@echo "✓ EDA complete → outputs/eda/"

run-preprocessing:
	@echo "── Running preprocessing step…"
	$(PYTHON) src/02_preprocessing.py \
		--input_dir   outputs/eda \
		--output_dir  outputs/preprocessing \
		--train_until $(TRAIN_UNTIL)
	@echo "✓ Preprocessing complete → outputs/preprocessing/"

run-modeling:
	@echo "── Running modeling step…"
	$(PYTHON) src/03_modeling.py \
		--input_dir  outputs/preprocessing \
		--output_dir outputs/modeling
	@echo "✓ Modeling complete → outputs/modeling/"

run-forecasting:
	@echo "── Running forecasting step…"
	$(PYTHON) src/04_forecasting.py \
		--model_dir      outputs/modeling \
		--pre_dir        outputs/preprocessing \
		--output_dir     outputs/forecasting \
		--forecast_until $(FORECAST_UNTIL)
	@echo "✓ Forecasting complete → outputs/forecasting/"

run-deeplearning:
	@echo "── Running deep learning step…"
	$(PYTHON) src/05_deeplearning.py \
		--pre_dir    outputs/preprocessing \
		--output_dir outputs/deeplearning \
		--quick
	@echo "✓ Deep learning complete → outputs/deeplearning/"

# ── Re-run steps (skip cache check) ──────────────────────────────────────────
rerun-eda: run-eda

rerun-preprocessing: run-preprocessing

rerun-modeling: run-modeling

rerun-forecasting: run-forecasting

rerun-deeplearning:
	@echo "── Re-running deep learning step (full epochs)…"
	$(PYTHON) src/05_deeplearning.py \
		--pre_dir    outputs/preprocessing \
		--output_dir outputs/deeplearning
	@echo "✓ Deep learning complete → outputs/deeplearning/"

# =============================================================================
# CACHE
# =============================================================================
cache-status:
	@echo "── Pipeline cache status:"
	$(PYTHON) pipeline.py --cache-status
	@echo "✓ Cache status shown."

cache-clear:
	@echo "── Clearing pipeline cache…"
	$(PYTHON) pipeline.py --invalidate ALL
	@echo "✓ Cache cleared."

cache-invalidate-eda:
	@echo "── Invalidating eda cache…"
	$(PYTHON) pipeline.py --invalidate eda
	@echo "✓ eda cache invalidated."

cache-invalidate-preprocessing:
	@echo "── Invalidating preprocessing cache…"
	$(PYTHON) pipeline.py --invalidate preprocessing
	@echo "✓ preprocessing cache invalidated."

cache-invalidate-modeling:
	@echo "── Invalidating modeling cache…"
	$(PYTHON) pipeline.py --invalidate modeling
	@echo "✓ modeling cache invalidated."

cache-invalidate-forecasting:
	@echo "── Invalidating forecasting cache…"
	$(PYTHON) pipeline.py --invalidate forecasting
	@echo "✓ forecasting cache invalidated."

cache-invalidate-deeplearning:
	@echo "── Invalidating deeplearning cache…"
	$(PYTHON) pipeline.py --invalidate deeplearning
	@echo "✓ deeplearning cache invalidated."

# =============================================================================
# TESTS
# =============================================================================
test:
	@echo "── Running full test suite + coverage…"
	pytest tests/ \
		-v --tb=short \
		--cov=src --cov=api --cov=pipeline --cov=mlflow_config \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-config=.coveragerc
	@echo ""
	@echo "✓ Tests complete. Coverage report → htmlcov/index.html"

test-fast:
	@echo "── Running fast unit tests…"
	pytest tests/ \
		-v --tb=short -x -q \
		--ignore=tests/test_api.py \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_deeplearning.py \
		--ignore=tests/test_parity.py
	@echo "✓ Fast tests complete."

test-unit:
	@echo "── Running unit tests…"
	pytest tests/test_eda.py \
	       tests/test_preprocessing.py \
	       tests/test_modeling.py \
	       tests/test_forecasting.py \
	       tests/test_mlflow_config.py \
	       tests/test_deeplearning.py \
		-v --tb=short
	@echo "✓ Unit tests complete."

test-dl:
	@echo "── Running deep learning tests…"
	pytest tests/test_deeplearning.py -v --tb=short
	@echo "✓ DL tests complete."

test-api:
	@echo "── Running API tests…"
	pytest tests/test_api.py -v --tb=short
	@echo "✓ API tests complete."

test-pipeline:
	@echo "── Running integration tests…"
	pytest tests/test_pipeline.py -v --tb=short
	@echo "✓ Integration tests complete."

test-parity:
	@echo "── Running parity tests (notebook vs script)…"
	pytest tests/test_parity.py -v --tb=short
	@echo "✓ Parity tests complete."

test-parity-full: run
	@echo "── Running full parity check (pipeline + parity tests)…"
	pytest tests/test_parity.py -v --tb=short
	@echo "✓ Full parity check complete."

test-cov:
	@echo "── Running coverage report (no API/pipeline)…"
	pytest tests/ -q \
		--ignore=tests/test_api.py \
		--ignore=tests/test_pipeline.py \
		--cov=src --cov=pipeline --cov=mlflow_config \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-config=.coveragerc
	@echo "✓ Coverage report → htmlcov/index.html"

# =============================================================================
# CODE QUALITY
# =============================================================================
lint:
	@echo "── Running flake8…"
	flake8 src/ api/ pipeline.py pipeline_cache.py mlflow_config.py \
		--max-line-length=100 --ignore=E501,W503
	@echo "✓ Lint passed."

lint-fix:
	@echo "── Formatting with isort + black…"
	isort src/ api/ pipeline.py pipeline_cache.py mlflow_config.py
	black src/ api/ pipeline.py pipeline_cache.py mlflow_config.py \
		--line-length 100
	@echo "✓ Format applied."

format: lint-fix

typecheck:
	@echo "── Running mypy…"
	mypy src/ api/ mlflow_config.py --config-file mypy.ini
	@echo "✓ Type check passed."

check: lint typecheck test-fast
	@echo "✓ All quality checks passed."

# =============================================================================
# API
# =============================================================================
api-run:
	@echo "── Starting FastAPI dev server → http://localhost:$(PORT)/docs"
	ENV=development OUTPUT_ROOT=outputs \
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

# =============================================================================
# MLFLOW
# =============================================================================
mlflow-ui:
	@echo "── Starting MLflow UI → http://127.0.0.1:$(MLFLOW_PORT)"
	mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri ./mlruns \
		--default-artifact-root ./mlruns/artifacts

mlflow-list:
	@echo "── Listing MLflow runs…"
	mlflow runs list --experiment-name ember-demand-forecasting
	@echo "✓ Done."

mlflow-clean:
	@echo "── Deleting mlruns/…"
	$(RM) mlruns/
	@echo "✓ mlruns/ deleted."

# =============================================================================
# DOCKER
# =============================================================================
docker-build:
	@echo "── Building pipeline image…"
	docker build -t $(FULL_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✓ Pipeline image built: $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "── Building API image…"
	docker build -f Dockerfile.api -t $(REGISTRY)/ember-energy-api:$(IMAGE_TAG) -t ember-energy-api:$(IMAGE_TAG) .
	@echo "✓ API image built: ember-energy-api:$(IMAGE_TAG)"
	@echo "✓ Both images ready."

docker-push: docker-build
	@echo "── Pushing pipeline image to $(REGISTRY)…"
	docker push $(FULL_IMAGE)
	@echo "✓ Pipeline image pushed."
	@echo "── Pushing API image to $(REGISTRY)…"
	docker push $(REGISTRY)/ember-energy-api:$(IMAGE_TAG)
	@echo "✓ API image pushed."
	@echo "✓ Both images available at $(REGISTRY)"

docker-run: $(CSV_PATH)
	@echo "── Running pipeline in Docker container…"
	docker run --rm \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:$(MLFLOW_PORT) \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/mlruns:/app/mlruns \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		--csv            /app/$(CSV_PATH) \
		--train_until    $(TRAIN_UNTIL) \
		--forecast_until $(FORECAST_UNTIL)
	@echo "✓ Pipeline container finished."

# =============================================================================
# DOCKER COMPOSE
# =============================================================================
compose-up:
	@echo "── Starting API + MLflow services…"
	docker compose up mlflow api --build -d
	@echo "✓ Services started."
	@echo "   API    → http://localhost:$(PORT)/docs"
	@echo "   MLflow → http://localhost:$(MLFLOW_PORT)"

compose-pipeline: $(CSV_PATH)
	@echo "── Running pipeline via Docker Compose…"
	docker compose run --rm pipeline
	@echo "✓ Pipeline container finished."

compose-jupyter:
	@echo "── Starting Jupyter Lab → http://localhost:8888"
	docker compose --profile dev up jupyter --build

compose-logs:
	@echo "── Streaming Docker Compose logs (Ctrl+C to stop)…"
	docker compose logs --follow

compose-down:
	@echo "── Stopping all Compose services…"
	docker compose down --remove-orphans
	@echo "✓ All services stopped."

# =============================================================================
# KUBERNETES
# =============================================================================
k8s-apply:
	@echo "── Applying Kubernetes manifests…"
	kubectl apply -f k8s/namespace.yaml
	@echo "  ✓ namespace"
	kubectl apply -f k8s/pvc.yaml
	@echo "  ✓ pvcs"
	kubectl apply -f k8s/configmap.yaml
	@echo "  ✓ configmap"
	kubectl apply -f k8s/secret.yaml 2>/dev/null || true
	@echo "  ✓ secret"
	kubectl apply -f k8s/deployment-mlflow.yaml
	@echo "  ✓ mlflow deployment"
	kubectl apply -f k8s/deployment-api.yaml
	@echo "  ✓ api deployment"
	kubectl apply -f k8s/hpa.yaml
	@echo "  ✓ hpa"
	kubectl apply -f k8s/cronjob.yaml
	@echo "  ✓ cronjob"
	@echo "✓ All K8s manifests applied to namespace: $(NAMESPACE)"

k8s-run-pipeline:
	@echo "── Submitting pipeline Job to Kubernetes…"
	kubectl delete job ember-pipeline -n $(NAMESPACE) --ignore-not-found
	kubectl apply -f k8s/job-pipeline.yaml
	@echo "✓ Job submitted. Watch with: make k8s-logs"

k8s-status:
	@echo "── Kubernetes status (namespace: $(NAMESPACE)):"
	kubectl get jobs,pods,pvc -n $(NAMESPACE)

k8s-logs:
	@echo "── Streaming pipeline pod logs (Ctrl+C to stop)…"
	kubectl logs -n $(NAMESPACE) -l app=ember-pipeline --follow --tail=100

k8s-delete:
	@echo "── Deleting namespace $(NAMESPACE) and all resources…"
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	@echo "✓ Namespace $(NAMESPACE) deleted."

# =============================================================================
# CLEANUP
# =============================================================================
clean:
	@echo "── Removing outputs/…"
	$(RM) outputs/
	@echo "✓ outputs/ removed."

clean-all: clean mlflow-clean
	@echo "── Removing cache and temp files…"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	$(RM) .cache/ 2>/dev/null || true
	@echo "✓ Full cleanup complete."

# =============================================================================
# DATA GUARD
# =============================================================================
$(CSV_PATH):
	@echo "ERROR: CSV not found at $(CSV_PATH)"
	@echo "Place the Ember CSV at: $(CSV_PATH)"
	@exit 1
