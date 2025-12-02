.PHONY: install install-dev sync notebook lab test lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Installation (choose one):"
	@echo "  make sync         - Install with uv (recommended, uses uv.lock)"
	@echo "  make install      - Install with pip"
	@echo "  make install-dev  - Install with pip (all dependencies)"
	@echo ""
	@echo "Jupyter:"
	@echo "  make notebook     - Start Jupyter Notebook server"
	@echo "  make lab          - Start JupyterLab server"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters (ruff, mypy)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove build artifacts"

# Installation with uv (recommended)
sync:
	uv sync --all-extras

# Installation with pip
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

# Jupyter (uses uv if available, falls back to direct command)
notebook:
	@command -v uv >/dev/null 2>&1 && uv run jupyter notebook --notebook-dir=notebooks || jupyter notebook --notebook-dir=notebooks

lab:
	@command -v uv >/dev/null 2>&1 && uv run jupyter lab --notebook-dir=notebooks || jupyter lab --notebook-dir=notebooks

# Development
test:
	@command -v uv >/dev/null 2>&1 && uv run pytest || pytest

lint:
	@command -v uv >/dev/null 2>&1 && uv run ruff check src/ && uv run mypy src/ || (ruff check src/ && mypy src/)

format:
	@command -v uv >/dev/null 2>&1 && uv run black src/ tests/ && uv run ruff check --fix src/ || (black src/ tests/ && ruff check --fix src/)

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
