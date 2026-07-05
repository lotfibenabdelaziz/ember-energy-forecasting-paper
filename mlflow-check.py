import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

runs = client.search_runs(["624796449716303392"])
print("Total runs:", len(runs))
for r in runs[:5]:
    print(" ", r.info.run_name, "| MAPE=", r.data.metrics.get("MAPE", "N/A"))

print()
print("Registry models:")
for rm in client.search_registered_models():
    # Use new API — no deprecation warning
    versions = client.search_model_versions(f"name='{rm.name}'")
    for mv in versions:
        print(" ", rm.name, f"v{mv.version}", f"stage={mv.current_stage}")