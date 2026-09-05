# =============================================================================
# Makefile — Ember Energy Forecasting Pipeline
# IEEE Paper | CI/CD + MLflow
# Bash-only (Linux / macOS / WSL / Git Bash)
# Works with either uv or conda — set ENV_MANAGER accordingly.
# =============================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# =============================================================================
# ENVIRONMENT MANAGER SWITCH
# =============================================================================
# ENV_MANAGER = uv     -> uses a uv-managed .venv (default)
# ENV_MANAGER = conda  -> uses a named conda environment (CONDA_ENV_NAME)
#
# Override on the command line, e.g.:
#   make install ENV_MANAGER=conda
#   make test    ENV_MANAGER=uv
# =============================================================================
ENV_MANAGER     ?= uv
CONDA_ENV_NAME  ?= ember-energy-forecasting
PYTHON_VERSION  ?= 3.10

UV ?= uv

ifeq ($(ENV_MANAGER),uv)
  # RUN prefixes any tool invocation so it resolves inside the uv-managed venv,
  # never whatever happens to be first on PATH (conda, system python, etc.)
  RUN     := $(UV) run
  PYTHON  := $(UV) run python
else ifeq ($(ENV_MANAGER),conda)
  # --no-capture-output keeps live output (progress bars, streaming logs) intact
  RUN     := conda run -n $(CONDA_ENV_NAME) --no-capture-output
  PYTHON  := conda run -n $(CONDA_ENV_NAME) --no-capture-output python
else
  $(error Unsupported ENV_MANAGER "$(ENV_MANAGER)" — must be "uv" or "conda")
endif

RM      = $(PYTHON) -c "import sys,shutil,os; [shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p) for p in sys.argv[1:] if os.path.exists(p)]"
MKDIR   = $(PYTHON) -c "import sys,os; os.makedirs(sys.argv[1], exist_ok=True)"

PYTEST_OPTS = -p no:cacheprovider

# ── Image config ──────────────────────────────────────────────────────────────
IMAGE_NAME  ?= ember-energy-pipeline
IMAGE_TAG   ?= latest
REGISTRY    ?= ghcr.io/lotfibenabdelaziz
FULL_IMAGE   = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Pipeline config ───────────────────────────────────────────────────────────
CSV_PATH       ?= ./data/raw/yearly_full_release_long_format.csv
TRAIN_UNTIL    ?= 2024
FORECAST_UNTIL ?= 2030
NAMESPACE      ?= ember-pipeline
MLFLOW_PORT    ?= 5000
PORT           ?= 8000

# Fix: on Windows Git Bash, HOME isn't visible to Windows-native Python,
# so matplotlib's Path.home() lookup fails. Give it a local, always-valid dir.
export MPLCONFIGDIR := $(CURDIR)/.mplconfig

# ── Environment helpers ───────────────────────────────────────────────────────
SET_MLFLOW  = MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT) MLFLOW_ALLOW_FILE_STORE=true
SET_DEV_ENV = ENV=development OUTPUT_ROOT=outputs

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: help env-info install install-uv install-conda api-install staging-install \
        lint lint-fix format format-check typecheck check \
        run run-eda run-preprocessing run-modeling run-forecasting run-deeplearning \
        rerun-eda rerun-preprocessing rerun-modeling rerun-forecasting rerun-deeplearning \
        test test-fast test-unit test-dl test-api test-pipeline test-parity \
        test-parity-full test-cov \
        api-run \
        mlflow-ui mlflow-list mlflow-register mlflow-registry-status \
        mlflow-register-classic mlflow-register-dl mlflow-clean \
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
        staging-build staging-query staging-query-all staging-clean \
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
	@echo "  Active ENV_MANAGER: $(ENV_MANAGER)  (override with ENV_MANAGER=uv|conda)"
	@echo ""
	@echo "  ── SETUP ────────────────────────────────────────────────"
	@echo "    make env-info               Show which interpreter/env each command uses"
	@echo "    make install                Install deps (uv .venv OR conda env, per ENV_MANAGER)"
	@echo "    make staging-install        Add optional DuckDB staging backend"
	@echo ""
	@echo "  ── LOCAL PIPELINE ───────────────────────────────────────"
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
	@echo "    make lint                   ruff check"
	@echo "    make lint-fix / format       ruff --fix + ruff format"
	@echo "    make format-check           format check only, no changes"
	@echo "    make typecheck              mypy static analysis"
	@echo "    make check                  lint + typecheck + test-fast"
	@echo ""
	@echo "  ── API ──────────────────────────────────────────────────"
	@echo "    make api-run                FastAPI dev server (port $(PORT))"
	@echo ""
	@echo "  ── MLFLOW ───────────────────────────────────────────────"
	@echo "    make mlflow-ui              MLflow UI (port $(MLFLOW_PORT))"
	@echo "    make mlflow-list            List all experiment runs"
	@echo "    make mlflow-register        Register best models to registry"
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
	@echo "  ── DATA MODELING (staging) ──────────────────────────────"
	@echo "    make staging-build          Build star-schema warehouse (SQLite)"
	@echo "    make staging-query          Query the leak-safe train-split view"
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

# ── Diagnostics ───────────────────────────────────────────────────────────────
env-info:
	@echo "ENV_MANAGER   = $(ENV_MANAGER)"
	@echo "PYTHON cmd    = $(PYTHON)"
	@echo "RUN cmd       = $(RUN)"
	@echo "Resolved python:"
	$(PYTHON) -c "import sys; print(' ', sys.executable); print(' ', sys.version)"

# =============================================================================
# INSTALL
# =============================================================================
install:
ifeq ($(ENV_MANAGER),uv)
	$(MAKE) install-uv
else
	$(MAKE) install-conda
endif

install-uv:
	@echo "── [install-uv] Creating/syncing environment with uv…"
	$(UV) python install $(PYTHON_VERSION)
	$(UV) venv --python $(PYTHON_VERSION)
	$(UV) pip install -r requirements.txt
	@echo "✓  Dependencies installed via uv (.venv, Python $(PYTHON_VERSION))."

install-conda:
	@echo "── [install-conda] Creating/syncing conda environment '$(CONDA_ENV_NAME)'…"
	@if conda env list | grep -qE "^$(CONDA_ENV_NAME)[[:space:]]"; then \
		echo "   Environment exists — updating…"; \
		conda run -n $(CONDA_ENV_NAME) --no-capture-output pip install -r requirements.txt; \
	else \
		echo "   Creating new environment…"; \
		conda create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION); \
		conda run -n $(CONDA_ENV_NAME) --no-capture-output pip install -r requirements.txt; \
	fi
	@echo "✓  Dependencies installed via conda (env: $(CONDA_ENV_NAME))."

api-install:
	@echo "── [api-install] Installing API + dev extras…"
	$(RUN) pip install -e ".[api,langchain,dev]"
	@echo "✓  API dependencies installed."

staging-install:
	@echo "── [staging-install] Installing optional DuckDB backend for src/staging.py…"
	$(RUN) pip install -e ".[staging]"
	@echo "✓  Staging (DuckDB) dependencies installed. SQLite backend needs nothing extra."

# =============================================================================
# PIPELINE — local runs
# =============================================================================
all: run

run: $(CSV_PATH)
	@echo "── [run] Starting full pipeline (smart cache)…"
	$(SET_MLFLOW) $(PYTHON) pipeline.py --csv $(CSV_PATH) --train_until $(TRAIN_UNTIL) --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Pipeline complete."

run-force: $(CSV_PATH)
	@echo "── [run-force] Starting full pipeline (cache disabled)…"
	$(SET_MLFLOW) $(PYTHON) pipeline.py --csv $(CSV_PATH) --train_until $(TRAIN_UNTIL) --forecast_until $(FORECAST_UNTIL) --force
	@echo "✓  Pipeline complete (forced)."

run-eda: $(CSV_PATH)
	@echo "── [run-eda] Running EDA step…"
	$(PYTHON) src/step01_eda.py --csv $(CSV_PATH) --output_dir outputs/eda
	@echo "✓  EDA complete → outputs/eda/"

run-preprocessing:
	@echo "── [run-preprocessing] Running preprocessing step…"
	$(PYTHON) src/step02_preprocessing.py --input_dir outputs/eda --output_dir outputs/preprocessing --train_until $(TRAIN_UNTIL)
	@echo "✓  Preprocessing complete → outputs/preprocessing/"

run-modeling:
	@echo "── [run-modeling] Running modeling step…"
	$(SET_MLFLOW) $(PYTHON) src/step03_modeling.py --input_dir outputs/preprocessing --output_dir outputs/modeling
	@echo "✓  Modeling complete → outputs/modeling/"

run-forecasting:
	@echo "── [run-forecasting] Running forecasting step…"
	$(SET_MLFLOW) $(PYTHON) src/step04_forecasting.py --model_dir outputs/modeling --pre_dir outputs/preprocessing --output_dir outputs/forecasting --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Forecasting complete → outputs/forecasting/"

run-deeplearning:
	@echo "── [run-deeplearning] Running deep learning step (quick mode)…"
	$(PYTHON) src/step05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning --quick
	@echo "✓  Deep learning complete → outputs/deeplearning/"

rerun-eda:
	@echo "── [rerun-eda] Re-running EDA (cache bypassed)…"
	$(PYTHON) src/step01_eda.py --csv $(CSV_PATH) --output_dir outputs/eda
	@echo "✓  EDA complete → outputs/eda/"

rerun-preprocessing:
	@echo "── [rerun-preprocessing] Re-running preprocessing (cache bypassed)…"
	$(PYTHON) src/step02_preprocessing.py --input_dir outputs/eda --output_dir outputs/preprocessing --train_until $(TRAIN_UNTIL)
	@echo "✓  Preprocessing complete → outputs/preprocessing/"

rerun-modeling:
	@echo "── [rerun-modeling] Re-running modeling…"
	$(SET_MLFLOW) $(PYTHON) src/step03_modeling.py --input_dir outputs/preprocessing --output_dir outputs/modeling
	@echo "✓  Modeling complete → outputs/modeling/"

rerun-forecasting:
	@echo "── [rerun-forecasting] Re-running forecasting…"
	$(SET_MLFLOW) $(PYTHON) src/step04_forecasting.py --pre_dir outputs/preprocessing --model_dir outputs/modeling --output_dir outputs/forecasting --forecast_until $(FORECAST_UNTIL)
	@echo "✓  Forecasting complete → outputs/forecasting/"

rerun-deeplearning:
	@echo "── [rerun-deeplearning] Re-running deep learning (full epochs)…"
	$(PYTHON) src/step05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning
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
# TESTS — all invoked through $(RUN) so they resolve inside the correct env
# =============================================================================
COV_FLAGS = --cov=src --cov=api --cov=pipeline --cov=mlflow_config \
            --cov-report=term-missing --cov-report=html:htmlcov \
            --cov-report=xml:coverage.xml --cov-config=.coveragerc

test:
	@echo "── [test] Running full test suite with coverage ($(ENV_MANAGER))…"
	$(RUN) pytest tests/ $(PYTEST_OPTS) -v --tb=short $(COV_FLAGS)
	@echo "✓  Tests complete. HTML report → htmlcov/index.html"

TEST_FAST_IGNORES = --ignore=tests/test_api.py --ignore=tests/test_pipeline.py \
                     --ignore=tests/test_deeplearning.py --ignore=tests/test_parity.py

test-fast:
	@echo "── [test-fast] Running fast unit tests (stop on first failure)…"
	$(RUN) pytest tests/ $(PYTEST_OPTS) -v --tb=short -x -q $(TEST_FAST_IGNORES)
	@echo "✓  Fast tests complete."

UNIT_TEST_FILES = tests/test_eda.py tests/test_preprocessing.py tests/test_modeling.py \
                   tests/test_forecasting.py tests/test_mlflow_config.py tests/test_deeplearning.py

test-unit:
	@echo "── [test-unit] Running unit tests…"
	$(RUN) pytest $(UNIT_TEST_FILES) -v --tb=short
	@echo "✓  Unit tests complete."

test-dl:
	@echo "── [test-dl] Running deep learning tests…"
	$(RUN) pytest tests/test_deeplearning.py -v --tb=short
	@echo "✓  Deep learning tests complete."

test-api:
	@echo "── [test-api] Running FastAPI endpoint tests…"
	$(RUN) pytest tests/test_api.py -v --tb=short
	@echo "✓  API tests complete."

test-pipeline:
	@echo "── [test-pipeline] Running integration tests…"
	$(RUN) pytest tests/test_pipeline.py -v --tb=short
	@echo "✓  Integration tests complete."

test-parity:
	@echo "── [test-parity] Running notebook vs script parity tests…"
	$(RUN) pytest tests/test_parity.py -v --tb=short
	@echo "✓  Parity tests complete."

test-parity-full: run
	@echo "── [test-parity-full] Full parity check (pipeline re-run + parity tests)…"
	$(RUN) pytest tests/test_parity.py -v --tb=short
	@echo "✓  Full parity check complete."

COV_FLAGS_NOAPI = --ignore=tests/test_api.py --ignore=tests/test_pipeline.py \
                   --cov=src --cov=pipeline --cov=mlflow_config \
                   --cov-report=term-missing --cov-report=html:htmlcov --cov-config=.coveragerc

test-cov:
	@echo "── [test-cov] Running coverage report (excluding API + pipeline tests)…"
	$(RUN) pytest tests/ -q $(COV_FLAGS_NOAPI)
	@echo "✓  Coverage report → htmlcov/index.html"

# =============================================================================
# CODE QUALITY — powered by RUFF (invoked through $(RUN))
# =============================================================================
RUFF_TARGETS = src/ api/ pipeline.py pipeline_cache.py mlflow_config.py

lint:
	@echo "── [lint] Running ruff check…"
	$(RUN) ruff check $(RUFF_TARGETS)
	@echo "✓  Lint passed."

lint-fix:
	@echo "── [lint-fix] Auto-fixing with ruff…"
	$(RUN) ruff check --fix $(RUFF_TARGETS)
	@echo "── [lint-fix] Formatting with ruff format…"
	$(RUN) ruff format $(RUFF_TARGETS)
	@echo "✓  Code fixed and formatted."

format: lint-fix

format-check:
	@echo "── [format-check] Checking format without changes…"
	$(RUN) ruff format --check $(RUFF_TARGETS)
	$(RUN) ruff check $(RUFF_TARGETS)
	@echo "✓  Format check passed."

typecheck:
	@echo "── [typecheck] Running mypy…"
	$(RUN) mypy src/ api/ mlflow_config.py --config-file mypy.ini
	@echo "✓  Type check passed."

check: lint typecheck test-fast
	@echo "✓  All quality gates passed."

# =============================================================================
# API
# =============================================================================
api-run:
	@echo "── [api-run] Starting FastAPI development server…"
	@echo "   Docs → http://localhost:$(PORT)/docs"
	$(SET_DEV_ENV) $(RUN) uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

# =============================================================================
# MLFLOW (invoked through $(RUN))
# =============================================================================
mlflow-ui:
	@echo "── [mlflow-ui] Starting MLflow tracking server…"
	@echo "   UI → http://127.0.0.1:$(MLFLOW_PORT)"
	MLFLOW_ALLOW_FILE_STORE=true $(RUN) mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) --backend-store-uri ./mlruns --default-artifact-root ./mlruns/artifacts

mlflow-list:
	@echo "── [mlflow-list] Listing runs in experiment 'ember-demand-forecasting'…"
	$(RUN) mlflow runs list --experiment-name ember-demand-forecasting
	@echo "✓  Run list complete."

mlflow-register:
	@echo "── [mlflow-register] Registering best models to MLflow Registry…"
	$(PYTHON) src/register_models.py --model_dir outputs/modeling --pre_dir outputs/preprocessing --dl_dir outputs/deeplearning
	@echo "✓  Registration complete. View: http://localhost:5000/#/models"

mlflow-registry-status:
	@echo "── [mlflow-registry-status] Current registry status:"
	$(PYTHON) -c "from mlflow_config import registry_summary; registry_summary()"

mlflow-register-classic:
	@echo "── Registering classical models only…"
	$(PYTHON) src/register_models.py --classic-only
	@echo "✓  Classical models registered."

mlflow-register-dl:
	@echo "── Registering DL models only…"
	$(PYTHON) src/register_models.py --dl-only
	@echo "✓  DL models registered."

mlflow-clean:
	@echo "── [mlflow-clean] Deleting local mlruns/ directory…"
	$(RM) mlruns/
	@echo "✓  mlruns/ removed."

# =============================================================================
# DVC — Data Version Control (invoked through $(RUN))
# =============================================================================
dvc-init:
	@echo "── [dvc-init] Initialising DVC in repository…"
	$(RUN) dvc init
	$(RUN) dvc config core.autostage true
	$(RUN) dvc config core.analytics false
	@echo "✓  DVC initialised. Next: make dvc-remote-add"

dvc-remote-add:
	@echo "── [dvc-remote-add] Edit .dvc/config and uncomment your preferred remote, then run:"
	@echo "     dvc remote default <name>"

dvc-add-data:
	@echo "── [dvc-add-data] Tracking raw CSV with DVC…"
	$(RUN) dvc add data/raw/yearly_full_release_long_format.csv
	@echo "✓  Data file tracked. Commit the generated .dvc pointer file."

dvc-pull:
	@echo "── [dvc-pull] Pulling data and outputs from DVC remote…"
	$(RUN) dvc pull
	@echo "✓  Pull complete."

dvc-push:
	@echo "── [dvc-push] Pushing data and outputs to DVC remote…"
	$(RUN) dvc push
	@echo "✓  Push complete."

dvc-repro:
	@echo "── [dvc-repro] Reproducing full DVC pipeline…"
	$(RUN) dvc repro
	@echo "✓  Pipeline reproduced. Run 'make dvc-push' to cache outputs."

dvc-repro-eda:
	@echo "── [dvc-repro-eda] Reproducing EDA stage…"
	$(RUN) dvc repro eda
	@echo "✓  EDA stage reproduced."

dvc-repro-preprocessing:
	@echo "── [dvc-repro-preprocessing] Reproducing preprocessing stage…"
	$(RUN) dvc repro preprocessing
	@echo "✓  Preprocessing stage reproduced."

dvc-repro-modeling:
	@echo "── [dvc-repro-modeling] Reproducing modeling stage…"
	$(RUN) dvc repro modeling
	@echo "✓  Modeling stage reproduced."

dvc-repro-forecasting:
	@echo "── [dvc-repro-forecasting] Reproducing forecasting stage…"
	$(RUN) dvc repro forecasting
	@echo "✓  Forecasting stage reproduced."

dvc-repro-deeplearning:
	@echo "── [dvc-repro-deeplearning] Reproducing deep learning stage…"
	$(RUN) dvc repro deeplearning
	@echo "✓  Deep learning stage reproduced."

dvc-dag:
	@echo "── [dvc-dag] Printing pipeline DAG…"
	$(RUN) dvc dag

dvc-status:
	@echo "── [dvc-status] DVC pipeline status…"
	$(RUN) dvc status

dvc-params:
	@echo "── [dvc-params] Showing parameter diff…"
	$(RUN) dvc params diff

dvc-metrics:
	@echo "── [dvc-metrics] Showing metrics…"
	$(RUN) dvc metrics show
	$(RUN) dvc metrics diff

dvc-plots:
	@echo "── [dvc-plots] Generating DVC plots…"
	$(RUN) dvc plots show

dvc-gc:
	@echo "── [dvc-gc] Removing unused DVC cache entries…"
	$(RUN) dvc gc --workspace --force
	@echo "✓  Garbage collection complete."

dvc-cache-info:
	@echo "── [dvc-cache-info] Local DVC cache size:"
	$(PYTHON) -c "import os; \
p='.dvc/cache'; \
total=sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs) if os.path.isdir(p) else 0; \
print(f'{total/1e6:.1f} MB' if total else 'No local cache yet.')"

# =============================================================================
# DATA MODELING — star-schema staging warehouse (src/staging.py)
# =============================================================================
staging-build:
	@echo "── [staging-build] Building star-schema warehouse from raw CSV…"
	$(PYTHON) src/staging.py build --csv $(CSV_PATH) --db warehouse/ember.db
	@echo "✓  warehouse/ember.db built. Inspect with: make staging-query"

staging-query:
	@echo "── [staging-query] Train-split rows (leak-safe view) for all countries:"
	$(PYTHON) src/staging.py query --db warehouse/ember.db --split train

staging-query-all:
	@echo "── [staging-query-all] All rows (train+val+test+forecast years):"
	$(PYTHON) src/staging.py query --db warehouse/ember.db

staging-clean:
	@echo "── [staging-clean] Removing warehouse/ember.db…"
	$(RM) warehouse/ember.db warehouse/ember.duckdb
	@echo "✓  Warehouse removed. Rebuild with: make staging-build"

# =============================================================================
# DOCKER
# =============================================================================
docker-build:
	@echo "── [docker-build] Building pipeline image: $(IMAGE_NAME):$(IMAGE_TAG)…"
	cmd.exe //c "docker build -t $(FULL_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) ."
	@echo "✓  Pipeline image built."
	@echo "── [docker-build] Building API image: ember-energy-api:$(IMAGE_TAG)…"
	cmd.exe //c "docker build -f Dockerfile.api -t $(REGISTRY)/ember-energy-api:$(IMAGE_TAG) -t ember-energy-api:$(IMAGE_TAG) ."
	@echo "✓  API image built."
	@echo "✓  Both images ready."

docker-push: docker-build
	@echo "── [docker-push] Pushing pipeline image to $(REGISTRY)…"
	cmd.exe //c "docker push $(FULL_IMAGE)"
	@echo "✓  Pipeline image pushed."
	@echo "── [docker-push] Pushing API image to $(REGISTRY)…"
	cmd.exe //c "docker push $(REGISTRY)/ember-energy-api:$(IMAGE_TAG)"
	@echo "✓  API image pushed."
	@echo "✓  Both images available at $(REGISTRY)"

docker-run: $(CSV_PATH)
	@echo "── [docker-run] Running pipeline inside Docker container…"
	@# NOTE: $(PWD) here resolves via bash to a POSIX-style path (e.g. /c/Users/...).
	@# cmd.exe/Docker Desktop usually translates this correctly for -v mounts, but
	@# if you hit a "mount path not found" error (distinct from the exec bug this
	@# wrapper fixes), switch $(PWD) to $(CURDIR) or an explicit Windows path.
	cmd.exe //c "docker run --rm -e MLFLOW_TRACKING_URI=http://host.docker.internal:$(MLFLOW_PORT) -v $(PWD)/data:/app/data:ro -v $(PWD)/outputs:/app/outputs -v $(PWD)/mlruns:/app/mlruns $(IMAGE_NAME):$(IMAGE_TAG) --csv /app/$(CSV_PATH) --train_until $(TRAIN_UNTIL) --forecast_until $(FORECAST_UNTIL)"
	@echo "✓  Pipeline container finished."

# =============================================================================
# DOCKER COMPOSE
# =============================================================================
compose-up:
	@echo "── [compose-up] Starting API + MLflow services (detached)…"
	cmd.exe //c "docker compose up mlflow api --build -d"
	@echo "✓  Services started."
	@echo "   API    → http://localhost:$(PORT)/docs"
	@echo "   MLflow → http://localhost:$(MLFLOW_PORT)"

compose-pipeline: $(CSV_PATH)
	@echo "── [compose-pipeline] Running pipeline via Docker Compose…"
	cmd.exe //c "docker compose run --rm pipeline"
	@echo "✓  Pipeline container finished."

compose-jupyter:
	@echo "── [compose-jupyter] Starting Jupyter Lab → http://localhost:8888"
	cmd.exe //c "docker compose --profile dev up jupyter --build"

compose-logs:
	@echo "── [compose-logs] Streaming Docker Compose logs (Ctrl+C to stop)…"
	cmd.exe //c "docker compose logs --follow"

compose-down:
	@echo "── [compose-down] Stopping all Compose services…"
	cmd.exe //c "docker compose down --remove-orphans"
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

k8s-rebuild-api:
	@echo "── [k8s-rebuild-api] Rebuilding API image and restarting deployment…"
	cmd.exe //c "docker build -f Dockerfile.api -t ember-energy-api:latest ."
	kubectl rollout restart deployment/ember-api -n $(NAMESPACE)
	kubectl rollout status deployment/ember-api -n $(NAMESPACE)
	@echo "✓  API rebuilt and restarted → http://localhost:30800/docs"

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

clean-all: clean mlflow-clean staging-clean
	@echo "── [clean-all] Removing Python cache files and temp directories…"
	$(PYTHON) -c "import os,shutil; \
[shutil.rmtree(os.path.join(r,d), ignore_errors=True) for r,ds,_ in os.walk('.') for d in list(ds) if d in ('__pycache__','.ipynb_checkpoints')]; \
[os.remove(os.path.join(r,f)) for r,_,fs in os.walk('.') for f in fs if f.endswith('.pyc')]"
	$(RM) .cache/
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