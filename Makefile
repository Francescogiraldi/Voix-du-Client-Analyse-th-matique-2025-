# Makefile for Voix du Client project

.PHONY: help install install-dev setup test test-cov lint format clean run dashboard cli docs build

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Complete project setup"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean temporary files"
	@echo "  run          - Run dashboard"
	@echo "  dashboard    - Run Streamlit dashboard"
	@echo "  cli          - Show CLI help"
	@echo "  docs         - Generate documentation"
	@echo "  build        - Build package"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Setup
setup: install-dev
	python -m spacy download fr_core_news_sm
	mkdir -p data models logs
	cp .env.example .env
	@echo "Setup complete! Edit .env file if needed."

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/voix_du_client --cov-report=html --cov-report=term

# Code quality
lint:
	ruff check src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/
	ruff check src/ tests/ --fix

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

# Running
run: dashboard

dashboard:
	streamlit run src/voix_du_client/dashboard.py

cli:
	python -m src.voix_du_client.main --help

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation available in README.md"

# Building
build: clean
	python -m build

# Development helpers
dev-install-spacy:
	python -m spacy download fr_core_news_sm

dev-test-sample:
	python -m src.voix_du_client.main analyze --csv data/feedback.csv --k 5 --output results.csv

dev-check-all: format lint test
	@echo "All checks passed!"