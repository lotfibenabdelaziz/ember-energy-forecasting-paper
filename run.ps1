# =============================================================================
# run.ps1 — Ember Energy Forecasting Pipeline (Windows / PowerShell)
# IEEE Paper | CI/CD
#
# Usage:
#   .\run.ps1 help
#   .\run.ps1 install
#   .\run.ps1 run -Csv data\ember_yearly.csv
#   .\run.ps1 mlflow-ui
#   .\run.ps1 compose-up
#   .\run.ps1 test
# =============================================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [string]$Csv            = "data\ember_yearly_full_release_long_format.csv",
    [int]   $TrainUntil     = 2019,
    [int]   $ForecastUntil  = 2025,
    [string]$Registry       = "ghcr.io/your-org",
    [string]$ImageTag       = "latest",
    [string]$Namespace      = "ember-pipeline",
    [int]   $MlflowPort     = 5000,
    [string]$Steps          = "eda preprocessing modeling forecasting"
)

$ErrorActionPreference = "Stop"
$ImageName = "ember-energy-pipeline"
$FullImage = "$Registry/$ImageName`:$ImageTag"

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Header  { param([string]$msg) Write-Host "`n  $msg" -ForegroundColor Cyan }
function Write-Success { param([string]$msg) Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn    { param([string]$msg) Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Fail    { param([string]$msg) Write-Host "  ERR $msg" -ForegroundColor Red }

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Header $Label
    $t = Get-Date
    & $Block
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Fail "$Label failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    $elapsed = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    Write-Success "$Label done in ${elapsed}s"
}

# =============================================================================
switch ($Command) {

    # ── Help ──────────────────────────────────────────────────────────────────
    "help" {
        Write-Host @"

  Ember Energy Forecasting Pipeline — PowerShell Runner
  ═══════════════════════════════════════════════════════════

  LOCAL DEVELOPMENT
    .\run.ps1 install              Install Python dependencies
    .\run.ps1 api-install          Install API + LangChain deps
    .\run.ps1 lint                 Run flake8
    .\run.ps1 format               Run black + isort
    .\run.ps1 typecheck            Run mypy
    .\run.ps1 test                 Run pytest suite
    .\run.ps1 run                  Run full pipeline locally
    .\run.ps1 run-eda              EDA step only
    .\run.ps1 run-preprocessing    Preprocessing step only
    .\run.ps1 run-modeling         Modeling step only
    .\run.ps1 run-forecasting      Forecasting step only
    .\run.ps1 api-run              Start FastAPI dev server

  MLFLOW
    .\run.ps1 mlflow-ui            Start MLflow UI (port 5000)
    .\run.ps1 mlflow-list          List experiments and runs
    .\run.ps1 mlflow-clean         Delete mlruns/ folder

  DOCKER
    .\run.ps1 docker-build         Build pipeline + API images
    .\run.ps1 docker-push          Push images to registry
    .\run.ps1 docker-run           Run pipeline in Docker

  DOCKER COMPOSE
    .\run.ps1 compose-pipeline     Run pipeline via compose
    .\run.ps1 compose-up           Start API + MLflow servers
    .\run.ps1 compose-jupyter      Start Jupyter Lab (dev)
    .\run.ps1 compose-logs         Stream service logs
    .\run.ps1 compose-down         Stop all containers

  KUBERNETES
    .\run.ps1 k8s-apply            Apply all manifests
    .\run.ps1 k8s-run-pipeline     Submit Job
    .\run.ps1 k8s-status           Jobs + Pods + PVCs
    .\run.ps1 k8s-logs             Stream pod logs
    .\run.ps1 k8s-delete           Delete namespace

  CLEANUP
    .\run.ps1 clean                Remove outputs\
    .\run.ps1 clean-all            Remove outputs\ + mlruns\ + cache

  PARAMETERS (append to any command)
    -Csv            Path to Ember CSV  (default: $Csv)
    -TrainUntil     Last train year    (default: $TrainUntil)
    -ForecastUntil  Forecast horizon   (default: $ForecastUntil)
    -MlflowPort     MLflow port        (default: $MlflowPort)
    -Registry       Docker registry    (default: $Registry)
    -ImageTag       Docker tag         (default: $ImageTag)

"@
    }

    # ── Local Development ─────────────────────────────────────────────────────
    "install" {
        Invoke-Step "Install dependencies" {
            pip install --upgrade pip
            pip install -r requirements.txt
        }
    }

    "api-install" {
        Invoke-Step "Install API + LangChain" {
            pip install -e ".[api,langchain,dev]"
        }
    }

    "lint" {
        Invoke-Step "Lint" {
            flake8 src/ api/ pipeline.py mlflow_config.py --max-line-length=100 --ignore=E501,W503
        }
    }

    "format" {
        Invoke-Step "Format" {
            black src/ api/ pipeline.py mlflow_config.py
            isort src/ api/ pipeline.py mlflow_config.py
        }
    }

    "typecheck" {
        Invoke-Step "Type check" {
            mypy src/ api/ mlflow_config.py --config-file mypy.ini
        }
    }

    "test" {
        Invoke-Step "Pytest" {
            pytest tests/ -v --tb=short --color=yes
        }
    }

    "test-unit" {
        Invoke-Step "Unit tests only" {
            pytest tests/ -v -m unit --tb=short
        }
    }

    "test-fast" {
        Invoke-Step "Fast tests (no slow)" {
            pytest tests/ -v -m "not slow" --tb=short
        }
    }

    "api-run" {
        Write-Header "FastAPI dev server → http://localhost:8000/docs"
        $env:ENV = "development"
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    }

    "run" {
        if (-not (Test-Path $Csv)) {
            Write-Fail "CSV not found: $Csv"
            Write-Warn "Usage: .\run.ps1 run -Csv path\to\ember_yearly.csv"
            exit 1
        }
        Invoke-Step "Full pipeline" {
            $env:MLFLOW_TRACKING_URI = "http://localhost:$MlflowPort"
            python pipeline.py `
                --csv            $Csv `
                --train_until    $TrainUntil `
                --forecast_until $ForecastUntil
        }
    }

    "run-eda" {
        Invoke-Step "EDA" {
            python src\01_eda.py --csv $Csv --output_dir outputs\eda
        }
    }

    "run-preprocessing" {
        Invoke-Step "Preprocessing" {
            python src\02_preprocessing.py `
                --input_dir  outputs\eda `
                --output_dir outputs\preprocessing
        }
    }

    "run-modeling" {
        Invoke-Step "Modeling" {
            $env:MLFLOW_TRACKING_URI = "http://localhost:$MlflowPort"
            python src\03_modeling.py `
                --input_dir  outputs\preprocessing `
                --output_dir outputs\modeling
        }
    }

    "run-forecasting" {
        Invoke-Step "Forecasting" {
            $env:MLFLOW_TRACKING_URI = "http://localhost:$MlflowPort"
            python src\04_forecasting.py `
                --pre_dir        outputs\preprocessing `
                --model_dir      outputs\modeling `
                --output_dir     outputs\forecasting `
                --forecast_until $ForecastUntil
        }
    }

    # ── MLflow ────────────────────────────────────────────────────────────────
    "mlflow-ui" {
        Write-Header "MLflow UI → http://localhost:$MlflowPort"
        New-Item -ItemType Directory -Force -Path mlruns | Out-Null
        mlflow server `
            --host 0.0.0.0 `
            --port $MlflowPort `
            --backend-store-uri ./mlruns `
            --default-artifact-root ./mlruns/artifacts
    }

    "mlflow-list" {
        $env:MLFLOW_TRACKING_URI = "http://localhost:$MlflowPort"
        mlflow runs list --experiment-name ember-demand-forecasting
    }

    "mlflow-clean" {
        if (Test-Path "mlruns") {
            Remove-Item -Recurse -Force "mlruns"
            Write-Success "mlruns\ deleted."
        } else {
            Write-Warn "mlruns\ does not exist."
        }
    }

    # ── Docker ────────────────────────────────────────────────────────────────
    "docker-build" {
        Invoke-Step "Build pipeline image" {
            docker build -t "$FullImage" -t "${ImageName}:${ImageTag}" .
        }
        Invoke-Step "Build API image" {
            docker build -f Dockerfile.api -t "$Registry/ember-energy-api:$ImageTag" .
        }
    }

    "docker-push" {
        .\run.ps1 docker-build
        Invoke-Step "Push images" {
            docker push $FullImage
            docker push "$Registry/ember-energy-api:$ImageTag"
        }
    }

    "docker-run" {
        if (-not (Test-Path $Csv)) {
            Write-Fail "CSV not found: $Csv"; exit 1
        }
        $absCsv = (Resolve-Path $Csv).Path
        $absData = Split-Path $absCsv -Parent
        $absOut  = (New-Item -ItemType Directory -Force -Path outputs).FullName
        Invoke-Step "Docker run pipeline" {
            docker run --rm `
                -e "MLFLOW_TRACKING_URI=http://host.docker.internal:$MlflowPort" `
                -v "${absData}:/app/data:ro" `
                -v "${absOut}:/app/outputs" `
                "${ImageName}:${ImageTag}" `
                --csv /app/data/$(Split-Path $absCsv -Leaf) `
                --train_until    $TrainUntil `
                --forecast_until $ForecastUntil
        }
    }

    # ── Docker Compose ────────────────────────────────────────────────────────
    "compose-pipeline" {
        Invoke-Step "Compose: run pipeline" {
            docker compose run --rm pipeline
        }
    }

    "compose-up" {
        Invoke-Step "Compose: start API + MLflow" {
            docker compose up mlflow api --build -d
        }
        Write-Success "API    → http://localhost:8000/docs"
        Write-Success "MLflow → http://localhost:$MlflowPort"
    }

    "compose-jupyter" {
        Write-Header "Compose: Jupyter Lab → http://localhost:8888"
        docker compose --profile dev up jupyter --build
    }

    "compose-logs" {
        docker compose logs --follow
    }

    "compose-down" {
        Invoke-Step "Compose: stop all" {
            docker compose down --remove-orphans
        }
    }

    # ── Kubernetes ────────────────────────────────────────────────────────────
    "k8s-apply" {
        Invoke-Step "K8s: apply manifests" {
            kubectl apply -f k8s\namespace.yaml
            kubectl apply -f k8s\pvc.yaml
            kubectl apply -f k8s\configmap.yaml
            kubectl apply -f k8s\
        }
        Write-Success "All manifests applied to namespace: $Namespace"
    }

    "k8s-run-pipeline" {
        Invoke-Step "K8s: submit pipeline Job" {
            kubectl delete job ember-pipeline -n $Namespace --ignore-not-found
            kubectl apply -f k8s\job-pipeline.yaml
        }
        Write-Success "Job submitted. Watch: .\run.ps1 k8s-logs"
    }

    "k8s-status" {
        Write-Header "K8s: Jobs"
        kubectl get jobs -n $Namespace
        Write-Header "K8s: Pods"
        kubectl get pods -n $Namespace
        Write-Header "K8s: PVCs"
        kubectl get pvc -n $Namespace
    }

    "k8s-logs" {
        kubectl logs -n $Namespace `
            -l app=ember-pipeline `
            --follow --tail=100
    }

    "k8s-delete" {
        Invoke-Step "K8s: delete namespace $Namespace" {
            kubectl delete namespace $Namespace --ignore-not-found
        }
    }

    # ── Cleanup ───────────────────────────────────────────────────────────────
    "clean" {
        if (Test-Path "outputs") {
            Remove-Item -Recurse -Force "outputs"
            Write-Success "outputs\ removed."
        }
    }

    "clean-all" {
        .\run.ps1 clean
        .\run.ps1 mlflow-clean
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory |
            Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Filter "*.pyc" |
            Remove-Item -Force
        Get-ChildItem -Recurse -Filter ".ipynb_checkpoints" -Directory |
            Remove-Item -Recurse -Force
        Write-Success "All generated files removed."
    }

    # ── Unknown command ───────────────────────────────────────────────────────
    default {
        Write-Fail "Unknown command: $Command"
        Write-Warn "Run '.\run.ps1 help' for available commands."
        exit 1
    }
}
