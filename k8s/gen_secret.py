"""
k8s/gen_secret.py — Generate k8s/secret.yaml from .env
Run: python k8s/gen_secret.py
Called automatically by: make k8s-secret-gen
"""

import base64
import os
import sys
from pathlib import Path

# Try dotenv, fall back to manual parsing
try:
    from dotenv import dotenv_values
    env = dotenv_values(".env")
except ImportError:
    # Manual .env parser if python-dotenv not available
    env = {}
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")


def b64(val: str) -> str:
    """Base64-encode a string value."""
    return base64.b64encode((val or "placeholder").strip().encode()).decode()


# ── Read values from .env ─────────────────────────────────────────────────────
openai_key    = b64(env.get("OPENAI_API_KEY",  "placeholder"))
csv_path      = b64(env.get("CSV_PATH",         "data/raw/yearly_full_release_long_format.csv"))
admin_pwd     = b64(env.get("ADMIN_PASSWORD",   "changeme"))
secret_key    = b64(env.get("SECRET_KEY",       "dev-secret-key-change-in-production"))
admin_user    = b64(env.get("ADMIN_USERNAME",   "admin"))
mlflow_uri = b64("http://mlflow-service:5000")

# ── Generate YAML ─────────────────────────────────────────────────────────────
secret_yaml = f"""# k8s/secret.yaml — AUTO-GENERATED from .env by make k8s-secret-gen
# DO NOT edit manually — run: make k8s-secret-gen
# DO NOT commit this file — add k8s/secret.yaml to .gitignore

apiVersion: v1
kind: Secret
metadata:
  name: ember-secrets
  namespace: ember-pipeline
type: Opaque
data:
  OPENAI_API_KEY:       {openai_key}
  CSV_PATH:             {csv_path}
  ADMIN_PASSWORD:       {admin_pwd}
  SECRET_KEY:           {secret_key}
  ADMIN_USERNAME:       {admin_user}
  MLFLOW_TRACKING_URI:  {mlflow_uri}
"""

# ── Write ─────────────────────────────────────────────────────────────────────
out_path = Path("k8s/secret.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(secret_yaml, encoding="utf-8")

print(f"✓ k8s/secret.yaml generated from .env")
print(f"  ADMIN_USERNAME : {env.get('ADMIN_USERNAME', 'admin')}")
print(f"  ADMIN_PASSWORD : {'*' * len(env.get('ADMIN_PASSWORD', ''))}")
print(f"  SECRET_KEY     : {'*' * len(env.get('SECRET_KEY', ''))}")
print(f"  OPENAI_API_KEY : {'set' if env.get('OPENAI_API_KEY','').startswith('sk-') else 'placeholder'}")