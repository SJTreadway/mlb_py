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

# Clean test cache
clean:
	rm -rf .pytest_cache tests/__pycache__ tests/*.pyc

# Run with coverage (install pytest-cov first)
test-cov:
	python3.11 -m pytest tests/ --cov=. --cov-report=term-missing

# Run pipeline
run:
	python3.11 pipeline.py

# Run only passing tests (useful after fixes)
test-failed:
	python3.11 -m pytest tests/ --lf

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
