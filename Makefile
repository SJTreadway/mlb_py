.PHONY: test test-verbose test-file clean install

# Run all tests
test:
	python3.11 -m pytest tests/

# Run tests with verbose output
test-verbose:
	python3.11 -m pytest tests/ -v

# Run specific test file (usage: make test-file FILE=test_batters_v2)
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
	python3.11 pipeline.py

run\:force:
	REFRESH_DATA=1 python3.11 pipeline.py

# Run only passing tests (useful after fixes)
test-failed:
	python3.11 -m pytest tests/ --lf

# Model Training
train\:homerun:
	python3.11 train/train_homerun_model.py

# Backup Training Data
backup\:cache:
	cp data/hr_training_data.csv data/hr_training_data_backup.csv

# Backup Model File
backup\:model:
	cp models/homerun_model_2026v1.pkl models/homerun_model_2026v1_backup.pkl

# Generate Backups
backup:
	make backup\:cache
	make backup\:model

# Delete Cache Files
clean\:cache:
	rm -f data/hr_training_data.csv
	rm -f data/hr_training_data_batter_checkpoint.pkl
	rm -f data/hr_training_data_pitcher_checkpoint.pkl

clean\:daily:
	rm -f data/daily/*.pkl

clean:
	make clean\:cache
	make clean\:daily

# Help
help:
	@echo "Available commands:"
	@echo "  make test              - Run all tests"
	@echo "  make test-verbose      - Run tests with verbose output"
	@echo "  make test-file FILE=name - Run specific test file"
	@echo "  make test-pattern PATTERN=keyword - Run tests matching pattern"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo "  make clean             - Clean test cache"
	@echo "  make run               - Run the pipeline"
