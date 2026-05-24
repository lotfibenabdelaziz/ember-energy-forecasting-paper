# # =============================================================================
# # DVC — Data Version Control
# # =============================================================================

# # ── Setup ─────────────────────────────────────────────────────────────────────
# dvc-init:
# 	dvc init
# 	dvc config core.autostage true
# 	dvc config core.analytics false
# 	@echo "✓ DVC initialised. Add a remote: make dvc-remote-add"

# dvc-remote-add:
# 	@echo "Edit .dvc/config and uncomment your preferred remote, then run:"
# 	@echo "  dvc remote default <name>"

# # ── Data ──────────────────────────────────────────────────────────────────────
# dvc-add-data:
# 	dvc add data/raw/yearly_full_release_long_format.csv
# 	@echo "✓ Data file tracked by DVC. Commit the .dvc pointer file."

# dvc-pull:
# 	dvc pull
# 	@echo "✓ Data and outputs pulled from remote."

# dvc-push:
# 	dvc push
# 	@echo "✓ Data and outputs pushed to remote."

# # ── Pipeline ──────────────────────────────────────────────────────────────────
# dvc-repro:
# 	dvc repro
# 	@echo "✓ Pipeline reproduced. Run 'dvc push' to cache outputs."

# dvc-repro-eda:
# 	dvc repro eda

# dvc-repro-preprocessing:
# 	dvc repro preprocessing

# dvc-repro-modeling:
# 	dvc repro modeling

# dvc-repro-forecasting:
# 	dvc repro forecasting

# dvc-repro-deeplearning:
# 	dvc repro deeplearning

# # ── Inspection ────────────────────────────────────────────────────────────────
# dvc-dag:
# 	dvc dag

# dvc-status:
# 	dvc status

# dvc-params:
# 	dvc params diff

# dvc-metrics:
# 	dvc metrics show
# 	dvc metrics diff

# dvc-plots:
# 	dvc plots show

# # ── Cache ─────────────────────────────────────────────────────────────────────
# dvc-gc:
# 	dvc gc --workspace --force
# 	@echo "✓ Unused cache entries removed."

# dvc-cache-info:
# 	du -sh .dvc/cache 2>/dev/null || echo "No local cache yet."
