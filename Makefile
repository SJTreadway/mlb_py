.PHONY: test test-verbose test-file clean install

# Run all tests
test:
	python3.11 -m pytest tests/

# Run tests with verbose output
test-verbose:
	python3.11 -m pytest tests/ -v

# Run specific test file (usage: make test-file FILE=test_batters)
test-file:
	python3.11 -m pytest tests/$(FILE).py -v

# Run tests matching pattern (usage: make test-pattern PATTERN=retro)
test-pattern:
	python3.11 -m pytest tests/ -k "$(PATTERN)" -v

# Install test dependencies
install-test:
	python3.11 -m pip install pytest --user

# Run with coverage (install pytest-cov first)
test-cov:
	python3.11 -m pytest tests/ --cov=. --cov-report=term-missing

# Run pipeline
run:
	REFRESH_DATA=0 python3.11 results_tracker.py && python3.11 pipeline.py

run\:force:
	REFRESH_DATA=1 python3.11 results_tracker.py && python3.11 pipeline.py

# Run only passing tests (useful after fixes)
test-failed:
	python3.11 -m pytest tests/ --lf

# ──────────────────────────────────────────────────────────────────────────────
# Single-Model Training
# ──────────────────────────────────────────────────────────────────────────────
train\:win:
	python3.11 train/win/train_win_model.py

train\:homerun:
	python3.11 train/homerun/train_homerun_model.py

# w/ Optuna Hyperparams Tuning
train\:win\:tune:
	TUNE_HYPERPARAMS=1 python3.11  train/win/train_win_model.py

train\:homerun\:tune:
	TUNE_HYPERPARAMS=1 python3.11  train/homerun/train_homerun_model.py

# Run Model Training Script Without Training
dryrun\:win:
	python3.11 train/win/train_win_model.py --dry-run

dryrun\:homerun:
	python3.11 train/homerun/train_homerun_model.py --dry-run

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Fleet Training
#   Trains a fleet of models (N seed-bagged XGBoost + per-feature-view XGBoost +
#   LightGBM) and saves models/ensemble_<model>_2026v1.pkl
#   Pass extra flags via ARGS, e.g.:
#     make ensemble:win ARGS="--n-seeds 15"
#     make ensemble:homerun ARGS="--n-seeds 15 --skip-lightgbm --min-pa 50"
# ──────────────────────────────────────────────────────────────────────────────
ensemble\:win:
	python3.11 train/win/ensemble_win.py $(ARGS)

ensemble\:homerun:
	python3.11 train/homerun/ensemble_homerun.py $(ARGS)

# ──────────────────────────────────────────────────────────────────────────────
# Stacking / Strategy Comparison
#   Loads the fleet artifact, scores every base model + averaging + stacked
#   meta-learner on AUC / log-loss / Brier, prints the winner, and saves
#   models/best_ensemble_strategy_<model>_2026v1.pkl
# ──────────────────────────────────────────────────────────────────────────────
stack\:win:
	python3.11 train/stack_models.py --ensemble-path models/ensemble_win_2026v1.pkl

stack\:homerun:
	python3.11 train/stack_models.py --ensemble-path models/ensemble_homerun_2026v1.pkl

# Train fleet, then immediately compare strategies (end-to-end)
ensemble\:win\:all:
	make ensemble\:win && make stack\:win

ensemble\:homerun\:all:
	make ensemble\:homerun && make stack\:homerun

# ──────────────────────────────────────────────────────────────────────────────
# Model Results Tracking
# ──────────────────────────────────────────────────────────────────────────────
results:
	python3.11 results_tracker.py $(DATE)

results\:today:
	python3.11 results_tracker.py

results\:summary:
	python3.11 -c "from results_tracker import print_summary; print_summary()"

# Backfill Weather Data Cache For Model Training
backfill\:weather:
	python3.11 api/weather.py --cache cache/hr_training_data_weather_cache.pkl

# ──────────────────────────────────────────────────────────────────────────────
# Backups
# ──────────────────────────────────────────────────────────────────────────────
# Backup Training Data
backup\:cache:
	cp data/hr_training_data.csv data/hr_training_data_backup.csv

# Backup Model File
backup\:model:
	cp models/homerun_model_2026v1.pkl models/homerun_model_2026v1_backup.pkl

# Backup Ensemble Fleet + Winning-Strategy Artifacts
backup\:ensemble:
	cp models/ensemble_win_2026v1.pkl models/ensemble_win_2026v1_backup.pkl 2>/dev/null || true
	cp models/ensemble_homerun_2026v1.pkl models/ensemble_homerun_2026v1_backup.pkl 2>/dev/null || true
	cp models/best_ensemble_strategy_win_2026v1.pkl models/best_ensemble_strategy_win_2026v1_backup.pkl 2>/dev/null || true
	cp models/best_ensemble_strategy_homerun_2026v1.pkl models/best_ensemble_strategy_homerun_2026v1_backup.pkl 2>/dev/null || true

# Generate Backups
backup:
	make backup\:cache
	make backup\:model

# Delete Daily Files
clean:
	python3.11 cleanup.py

# Help
help:
	@echo "Available commands:"
	@echo "  make test                       - Run all tests"
	@echo "  make test-verbose               - Run tests with verbose output"
	@echo "  make test-file FILE=name        - Run specific test file"
	@echo "  make test-pattern PATTERN=key   - Run tests matching pattern"
	@echo "  make test-cov                   - Run tests with coverage report"
	@echo "  make run                        - Run the pipeline"
	@echo ""
	@echo "  Single-model training:"
	@echo "  make train:win                  - Train the win model"
	@echo "  make train:homerun              - Train the HR model"
	@echo "  make train:win:tune             - Train win model w/ Optuna tuning"
	@echo "  make train:homerun:tune         - Train HR model w/ Optuna tuning"
	@echo ""
	@echo "  Ensemble training & stacking:"
	@echo "  make ensemble:win               - Train win-model fleet"
	@echo "  make ensemble:homerun           - Train HR-model fleet"
	@echo "  make ensemble:win ARGS=\"...\"    - Pass flags (e.g. --n-seeds 15)"
	@echo "  make stack:win                  - Compare strategies on win fleet"
	@echo "  make stack:homerun              - Compare strategies on HR fleet"
	@echo "  make ensemble:win:all           - Train win fleet + stack in one go"
	@echo "  make ensemble:homerun:all       - Train HR fleet + stack in one go"
	@echo ""
	@echo "  make clean                      - Clean daily files"