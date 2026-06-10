# =============================================================================
# Makefile — convenience targets for the Fusion Model project.
#
# Requires: uv (https://docs.astral.sh/uv/)
#
# Usage examples:
#     make install      # create venv + install all deps (incl. dev)
#     make lint         # run ruff and mypy
#     make test         # run pytest
#     make format       # auto-format code with ruff
#     make precommit    # install pre-commit hooks
#     make train        # launch a quick training run (100 samples)
#     make all          # install + lint + test
# =============================================================================

.PHONY: all install lint format typecheck test precommit train clean help

# Default target: install, lint, then test.
all: install lint test

# --------------------------------------------------------------------------
# install — create a virtual environment and install all dependencies.
# --------------------------------------------------------------------------
install:
	uv sync --all-extras

# --------------------------------------------------------------------------
# lint — run the ruff linter (without auto-fixing).
# --------------------------------------------------------------------------
lint: typecheck
	uv run ruff check .

# --------------------------------------------------------------------------
# format — auto-format all Python files with ruff.
# --------------------------------------------------------------------------
format:
	uv run ruff format .
	uv run ruff check --fix .

# --------------------------------------------------------------------------
# typecheck — run mypy static type checking.
# --------------------------------------------------------------------------
typecheck:
	uv run mypy fusion_model/ tasks/ train.py evaluate.py

# --------------------------------------------------------------------------
# test — run the pytest test suite.
# --------------------------------------------------------------------------
test:
	uv run pytest -v

# --------------------------------------------------------------------------
# precommit — install git pre-commit hooks.
# --------------------------------------------------------------------------
precommit:
	uv run pre-commit install

# --------------------------------------------------------------------------
# train — launch a quick debug training run with a small data subset.
# --------------------------------------------------------------------------
train:
	uv run python train.py --max_samples 20 --epochs 2 --batch_size 4

# --------------------------------------------------------------------------
# clean — remove generated artefacts.
# --------------------------------------------------------------------------
clean:
	rm -rf outputs/ __pycache__ fusion_model/__pycache__ tasks/__pycache__
	rm -rf .mypy_cache .ruff_cache .pytest_cache

# --------------------------------------------------------------------------
# help — print available targets.
# --------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  install     Create venv + install all deps via uv"
	@echo "  lint        Run ruff linter + mypy type-checker"
	@echo "  format      Auto-format code with ruff"
	@echo "  typecheck   Run mypy only"
	@echo "  test        Run pytest test suite"
	@echo "  precommit   Install pre-commit hooks"
	@echo "  train       Quick debug training run (100 samples)"
	@echo "  clean       Remove generated artefacts"
	@echo "  all         install + lint + test"
