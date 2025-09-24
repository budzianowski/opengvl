.PHONY: install format lint test test-coverage test-fast test-verbose test-specific clean help pyright

# Install dependencies
install:
	uv sync

# Format code
format:
	uv run isort .
	uv run black .
	uv run ruff check --fix .
	uv run ruff format .

# Lint code (check only)
lint:
	uv run isort --check-only .
	uv run black --check .
	uv run ruff check .

# Run tests
test:
	uv run python3 -m pytest tests/ -v

# Run tests with coverage report
test-coverage:
	uv run python3 -m pytest tests/ -v --cov=opengvl --cov-report=term-missing --cov-report=html

# Run tests quickly (no verbose output)
test-fast:
	uv run python3 -m pytest tests/ -q

# Run tests with extra verbose output and show local variables on failure
test-verbose:
	uv run python3 -m pytest tests/ -vvv --tb=long

# Run specific test file or pattern
# Usage: make test-specific TEST=test_voc_metric.py
# Usage: make test-specific TEST="test_data_types.py::TestEpisode"
test-specific:
	uv run python3 -m pytest tests/$(TEST) -v

# Run tests and generate XML report for CI
test-ci:
	uv run python3 -m pytest tests/ --junitxml=test-results.xml --cov=opengvl --cov-report=xml

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	uv run python3 -m pytest tests/ -v -n auto

# Clean up
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf test-results.xml
	rm -rf coverage.xml
	find . -name "*.pyc" -delete

# Typing checking
pyright:
	uv run pyright opengvl

# Run pylint (static analysis using .pylintrc). Does not fail the build by default; remove '|| true' to enforce.
pylint:
	uv run pylint opengvl tests --rcfile=.pylintrc

# Show help
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies with uv"
	@echo "  format         - Format code with isort, black, and ruff"
	@echo "  lint           - Lint code (check only)"
	@echo "  test           - Run all tests with verbose output"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-fast      - Run tests quickly (quiet mode)"
	@echo "  test-verbose   - Run tests with extra verbose output"
	@echo "  test-specific  - Run specific test (use TEST=filename or pattern)"
	@echo "  test-ci        - Run tests with CI-friendly output (XML reports)"
	@echo "  test-parallel  - Run tests in parallel (requires pytest-xdist)"
	@echo "  clean          - Clean up generated files"
	@echo "  pyright        - Run type checking"
	@echo "  pylint         - Run pylint static analysis (non-fatal)"
	@echo "  help           - Show this help message"