# =============================================================================
# Make.ps1 — PowerShell convenience script for the Fusion Model project.
#
# Requires: uv (https://docs.astral.sh/uv/)
#
# Usage examples:
#     .\Make.ps1 install      # create venv + install all deps (incl. dev)
#     .\Make.ps1 lint         # run ruff and mypy
#     .\Make.ps1 test         # run pytest
#     .\Make.ps1 format       # auto-format code with ruff
#     .\Make.ps1 precommit    # install pre-commit hooks
#     .\Make.ps1 train        # launch a quick training run (100 samples)
#     .\Make.ps1 all          # install + lint + test
# =============================================================================

param(
    # The target to execute — one of the function names below.
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

# ── Helper: abort on first error ────────────────────────────────────────────
$ErrorActionPreference = "Stop"

# ── Targets ─────────────────────────────────────────────────────────────────

function Install {
    <# Create a virtual environment and install all dependencies. #>
    Write-Host ">> Installing dependencies with uv..." -ForegroundColor Cyan
    uv sync --all-extras
}

function Lint {
    <# Run the ruff linter and mypy type-checker. #>
    Typecheck
    Write-Host ">> Running ruff linter..." -ForegroundColor Cyan
    uv run ruff check .
}

function Format {
    <# Auto-format all Python files with ruff. #>
    Write-Host ">> Formatting with ruff..." -ForegroundColor Cyan
    uv run ruff format .
    uv run ruff check --fix .
}

function Typecheck {
    <# Run mypy static type checking. #>
    Write-Host ">> Running mypy..." -ForegroundColor Cyan
    uv run mypy fusion_model/ tasks/ train.py evaluate.py
}

function Test {
    <# Run the pytest test suite. #>
    Write-Host ">> Running tests..." -ForegroundColor Cyan
    uv run pytest -v
}

function Precommit {
    <# Install git pre-commit hooks. #>
    Write-Host ">> Installing pre-commit hooks..." -ForegroundColor Cyan
    uv run pre-commit install
}

function Train {
    <# Launch a quick debug training run with a small data subset. #>
    Write-Host ">> Starting debug training run..." -ForegroundColor Cyan
    uv run python train.py --max_samples 100 --epochs 2 --batch_size 16
}

function Clean {
    <# Remove generated artefacts. #>
    Write-Host ">> Cleaning up..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue outputs, __pycache__, `
        fusion_model/__pycache__, tasks/__pycache__, .mypy_cache, .ruff_cache, .pytest_cache
}

function All {
    <# Run install, lint, and test in sequence. #>
    Install
    Lint
    Test
}

function Help {
    <# Print available targets. #>
    Write-Host "Available targets:" -ForegroundColor Green
    Write-Host "  install     Create venv + install all deps via uv"
    Write-Host "  lint        Run ruff linter + mypy type-checker"
    Write-Host "  format      Auto-format code with ruff"
    Write-Host "  typecheck   Run mypy only"
    Write-Host "  test        Run pytest test suite"
    Write-Host "  precommit   Install pre-commit hooks"
    Write-Host "  train       Quick debug training run (100 samples)"
    Write-Host "  clean       Remove generated artefacts"
    Write-Host "  all         install + lint + test"
}

# ── Dispatch ────────────────────────────────────────────────────────────────

switch ($Target.ToLower()) {
    "install"    { Install }
    "lint"       { Lint }
    "format"     { Format }
    "typecheck"  { Typecheck }
    "test"       { Test }
    "precommit"  { Precommit }
    "train"      { Train }
    "clean"      { Clean }
    "all"        { All }
    "help"       { Help }
    default      { Write-Host "Unknown target: $Target" -ForegroundColor Red; Help }
}
