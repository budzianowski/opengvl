.PHONY: install format lint test clean help

# Install dependencies
install:p
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
	uv run pytest

# Clean up
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete

# Typing checking
pyright:
	uv run pyright opengvl