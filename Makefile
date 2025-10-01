.PHONY: install sync format lint test clean pyright help guard-uv banner
.DEFAULT_GOAL := help

# ------------------------------------------------------------
# Styling / UX helpers (colorless verbose mode)
# ------------------------------------------------------------
CHECK := [OK]
CROSS := [FAIL]
ARROW := ->
DOT   := *

define _echo_step
	@echo "$(DOT) $1"
endef

define _echo_ok
	@echo "  $(CHECK) $1"
endef

define _echo_fail
	@echo "  $(CROSS) $1" >&2
endef

define _banner
	@echo ""; \
	echo "==================== $1 ====================="; \
	echo ""
endef

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV := $(shell command -v uv 2>/dev/null)

# Guard to ensure 'uv' is installed
guard-uv:
ifndef UV
	$(error 'uv' not found in PATH. Install from https://github.com/astral-sh/uv)
endif

# ------------------------------------------------------------
# Primary targets
# ------------------------------------------------------------

install: guard-uv
	$(call _banner,Install)
	$(call _echo_step,Verifying environment)
	@echo "Using uv at: $(UV)" && echo "Python: $$(python3 --version 2>/dev/null || echo 'n/a')"
	$(call _echo_step,Running placeholder install script)
	@# Keep the *same functionality* (a simple success message) as requested
	@uv run python3 -c "print('installed dependencies successfully')"
	$(call _echo_ok,Done)

# Actual sync (resolve + lock install) if desired
sync: guard-uv
	$(call _banner,Sync Dependencies)
	$(call _echo_step,Syncing project environment via uv)
	@uv sync
	$(call _echo_ok,Synchronized)

# Format code (auto-fix)
format: guard-uv
	$(call _banner,Code Format)
	$(call _echo_step,Black format)
	@uv run black . >/dev/null
	$(call _echo_ok,Black complete)
	$(call _echo_step,Ruff (fix))
	@uv run ruff check --fix . >/dev/null
	$(call _echo_ok,Ruff fixes applied)
	$(call _echo_step,Ruff format (doc/style))
	@uv run ruff format . >/dev/null
	$(call _echo_ok,Ruff format done)
	$(call _echo_step,isort)
	@uv run isort . >/dev/null
	$(call _echo_ok,Imports sorted)

# Lint only (no modifications)
lint: guard-uv
	$(call _banner,Lint)
	$(call _echo_step,isort check)
	@uv run isort --check-only .
	$(call _echo_ok,isort clean)
	$(call _echo_step,black check)
	@uv run black --check .
	$(call _echo_ok,black clean)
	$(call _echo_step,ruff check)
	@uv run ruff check .
	$(call _echo_ok,ruff clean)

# Run tests with pytest
test: guard-uv
	$(call _banner,Tests)
	$(call _echo_step,Running pytest)
	@uv run pytest || { $(call _echo_fail,Tests failed); exit 1; }
	$(call _echo_ok,All tests passed)

# Static type checking
pyright: guard-uv
	$(call _banner,Type Check)
	$(call _echo_step,Running pyright)
	@uv run pyright opengvl || { $(call _echo_fail,Pyright failed); exit 1; }
	$(call _echo_ok,No type errors)

# Clean build / cache artifacts
clean:
	$(call _banner,Clean)
	$(call _echo_step,Removing virtual env .venv (if exists))
	@rm -rf .venv || true
	$(call _echo_step,Removing __pycache__ + pytest cache)
	@rm -rf __pycache__ .pytest_cache || true
	$(call _echo_step,Removing stray *.pyc files)
	@find . -name "*.pyc" -delete || true
	$(call _echo_ok,Clean complete)

# ------------------------------------------------------------
# Help & meta
# ------------------------------------------------------------
help:
	@echo "Project Make Targets"; \
	echo "Usage: make <target> [VAR=val]"; echo ""; \
	echo "Core"; \
	echo "  install   Placeholder install (kept same behavior)"; \
	echo "  sync      Sync dependencies with uv (actual env setup)"; \
	echo "  format    Auto-format (black, ruff, isort)"; \
	echo "  lint      Lint only (no changes)"; \
	echo "  test      Run test suite"; \
	echo "  pyright   Static type checking"; \
	echo "  clean     Remove caches & artifacts"; \
	echo ""; \
	echo "Environment"; \
	echo "  UV path : $(UV)"; \
	echo ""; \
	echo "Examples"; \
	echo "  make format"; \
	echo "  make test"; \
	echo ""; \
	echo "Tip: use 'make sync' to actually install dependencies."

# Silence make's own output for recipe lines but keep our echos
.SILENT: help install sync format lint test pyright clean guard-uv