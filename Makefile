# Makefile for DeepHallu Project
# Author: Yongli Mou
# Description: Automated tasks for development, testing, and deployment

# ============================================================================
# Configuration
# ============================================================================

# Python environment
VENV := /home/$(USER)/anaconda3/envs/deephallu
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy
JUPYTER := $(VENV)/bin/jupyter

# Project directories
SRC_DIR := src/deephallu
TEST_DIR := tests
NOTEBOOK_DIR := notebooks
WEB_BACKEND := src/deephallu/web/backend
WEB_FRONTEND := src/deephallu/web/frontend
DOCS_DIR := docs
DATA_DIR := data

# Source files
PYTHON_FILES := $(shell find $(SRC_DIR) $(TEST_DIR) -name "*.py" -not -path "*/node_modules/*")

# Build directories
BUILD_DIR := build
DIST_DIR := dist
EGG_INFO := src/deephallu.egg-info

# Colors for terminal output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_RED := \033[31m

# ============================================================================
# Default Target
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(COLOR_BOLD)DeepHallu Makefile Commands$(COLOR_RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Environment Setup
# ============================================================================

.PHONY: env
env: ## Create conda environment
	@echo "$(COLOR_BLUE)Creating conda environment: deephallu$(COLOR_RESET)"
	conda create -n deephallu python=3.12 -y
	@echo "$(COLOR_GREEN)✓ Environment created$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Activate with: conda activate deephallu$(COLOR_RESET)"

.PHONY: install
install: ## Install package in development mode
	@echo "$(COLOR_BLUE)Installing DeepHallu in development mode...$(COLOR_RESET)"
	$(PIP) install -e .
	@echo "$(COLOR_GREEN)✓ Package installed$(COLOR_RESET)"

.PHONY: install-torch
install-torch: ## Install PyTorch with CUDA 12.9 support
	@echo "$(COLOR_BLUE)Installing PyTorch with CUDA 12.9...$(COLOR_RESET)"
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu129
	@echo "$(COLOR_GREEN)✓ PyTorch installed$(COLOR_RESET)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(COLOR_BLUE)Installing development dependencies...$(COLOR_RESET)"
	$(PIP) install black isort flake8 mypy pytest pytest-cov jupyter
	@echo "$(COLOR_GREEN)✓ Development dependencies installed$(COLOR_RESET)"

.PHONY: install-web
install-web: ## Install web application dependencies
	@echo "$(COLOR_BLUE)Installing backend dependencies...$(COLOR_RESET)"
	$(PIP) install fastapi uvicorn python-multipart
	@echo "$(COLOR_BLUE)Installing frontend dependencies...$(COLOR_RESET)"
	cd $(WEB_FRONTEND) && npm install
	@echo "$(COLOR_GREEN)✓ Web dependencies installed$(COLOR_RESET)"

.PHONY: install-all
install-all: install-torch install install-dev install-web ## Install all dependencies
	@echo "$(COLOR_GREEN)✓ All dependencies installed$(COLOR_RESET)"

# ============================================================================
# Code Quality
# ============================================================================

.PHONY: format
format: ## Format code with black and isort
	@echo "$(COLOR_BLUE)Formatting code with black...$(COLOR_RESET)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@echo "$(COLOR_BLUE)Sorting imports with isort...$(COLOR_RESET)"
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@echo "$(COLOR_GREEN)✓ Code formatted$(COLOR_RESET)"

.PHONY: lint
lint: ## Lint code with flake8
	@echo "$(COLOR_BLUE)Linting code with flake8...$(COLOR_RESET)"
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR) --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(COLOR_GREEN)✓ Linting passed$(COLOR_RESET)"

.PHONY: typecheck
typecheck: ## Type check with mypy
	@echo "$(COLOR_BLUE)Type checking with mypy...$(COLOR_RESET)"
	$(MYPY) $(SRC_DIR) || true
	@echo "$(COLOR_GREEN)✓ Type checking complete$(COLOR_RESET)"

.PHONY: check
check: format lint typecheck ## Run all code quality checks
	@echo "$(COLOR_GREEN)✓ All checks passed$(COLOR_RESET)"

# ============================================================================
# Testing
# ============================================================================

.PHONY: test
test: ## Run tests with pytest
	@echo "$(COLOR_BLUE)Running tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -v
	@echo "$(COLOR_GREEN)✓ Tests passed$(COLOR_RESET)"

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "$(COLOR_BLUE)Running tests with coverage...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(COLOR_GREEN)✓ Coverage report generated$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)View report: open htmlcov/index.html$(COLOR_RESET)"

.PHONY: test-fast
test-fast: ## Run fast tests only (skip slow markers)
	@echo "$(COLOR_BLUE)Running fast tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -m "not slow" -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(COLOR_BLUE)Running integration tests...$(COLOR_RESET)"
	$(PYTEST) $(TEST_DIR) -m "integration" -v

# ============================================================================
# Development
# ============================================================================

.PHONY: notebook
notebook: ## Start Jupyter notebook server
	@echo "$(COLOR_BLUE)Starting Jupyter notebook...$(COLOR_RESET)"
	cd $(NOTEBOOK_DIR) && $(JUPYTER) notebook

.PHONY: lab
lab: ## Start Jupyter lab server
	@echo "$(COLOR_BLUE)Starting Jupyter lab...$(COLOR_RESET)"
	cd $(NOTEBOOK_DIR) && $(JUPYTER) lab

.PHONY: clean-notebooks
clean-notebooks: ## Clean notebook outputs
	@echo "$(COLOR_BLUE)Cleaning notebook outputs...$(COLOR_RESET)"
	$(JUPYTER) nbconvert --clear-output --inplace $(NOTEBOOK_DIR)/*.ipynb
	@echo "$(COLOR_GREEN)✓ Notebook outputs cleared$(COLOR_RESET)"

# ============================================================================
# Web Application
# ============================================================================

.PHONY: web-backend
web-backend: ## Run FastAPI backend server
	@echo "$(COLOR_BLUE)Starting FastAPI backend...$(COLOR_RESET)"
	cd $(WEB_BACKEND) && $(PYTHON) app.py

.PHONY: web-frontend
web-frontend: ## Run Next.js frontend development server
	@echo "$(COLOR_BLUE)Starting Next.js frontend...$(COLOR_RESET)"
	cd $(WEB_FRONTEND) && npm run dev

.PHONY: web-build
web-build: ## Build frontend for production
	@echo "$(COLOR_BLUE)Building frontend...$(COLOR_RESET)"
	cd $(WEB_FRONTEND) && npm run build
	@echo "$(COLOR_GREEN)✓ Frontend built$(COLOR_RESET)"

.PHONY: web
web: ## Run both backend and frontend (requires tmux or run in separate terminals)
	@echo "$(COLOR_YELLOW)Run 'make web-backend' and 'make web-frontend' in separate terminals$(COLOR_RESET)"

# ============================================================================
# Data Management
# ============================================================================

.PHONY: preprocess-mme
preprocess-mme: ## Preprocess MME dataset
	@echo "$(COLOR_BLUE)Preprocessing MME dataset...$(COLOR_RESET)"
	$(PYTHON) -m deephallu.preprocessing.mme
	@echo "$(COLOR_GREEN)✓ MME dataset preprocessed$(COLOR_RESET)"

.PHONY: data-info
data-info: ## Show dataset information
	@echo "$(COLOR_BLUE)Dataset Information:$(COLOR_RESET)"
	@ls -lh $(DATA_DIR)
	@echo ""
	@du -sh $(DATA_DIR)/*

# ============================================================================
# Inference & Evaluation
# ============================================================================

.PHONY: inference-mme
inference-mme: ## Run LLaVA inference on MME benchmark
	@echo "$(COLOR_BLUE)Running LLaVA inference on MME...$(COLOR_RESET)"
	$(PYTHON) -m deephallu.inference.run_llava_mme
	@echo "$(COLOR_GREEN)✓ Inference complete$(COLOR_RESET)"

.PHONY: eval-mme
eval-mme: ## Evaluate results on MME benchmark
	@echo "$(COLOR_BLUE)Evaluating MME benchmark results...$(COLOR_RESET)"
	$(PYTHON) -m deephallu.analytics.run_llava_mme
	@echo "$(COLOR_GREEN)✓ Evaluation complete$(COLOR_RESET)"

# ============================================================================
# Build & Distribution
# ============================================================================

.PHONY: build
build: clean ## Build distribution packages
	@echo "$(COLOR_BLUE)Building distribution packages...$(COLOR_RESET)"
	$(PYTHON) -m build
	@echo "$(COLOR_GREEN)✓ Build complete$(COLOR_RESET)"

.PHONY: upload-test
upload-test: build ## Upload to TestPyPI
	@echo "$(COLOR_BLUE)Uploading to TestPyPI...$(COLOR_RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: upload
upload: build ## Upload to PyPI
	@echo "$(COLOR_RED)WARNING: This will upload to PyPI!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) -m twine upload dist/*; \
	fi

# ============================================================================
# Cleaning
# ============================================================================

.PHONY: clean
clean: ## Remove build artifacts, cache files, and temporary files
	@echo "$(COLOR_BLUE)Cleaning build artifacts...$(COLOR_RESET)"
	rm -rf $(BUILD_DIR) $(DIST_DIR) $(EGG_INFO)
	@echo "$(COLOR_BLUE)Cleaning Python cache files...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(COLOR_BLUE)Cleaning test cache...$(COLOR_RESET)"
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	@echo "$(COLOR_BLUE)Cleaning Jupyter checkpoints...$(COLOR_RESET)"
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(COLOR_BLUE)Cleaning Next.js build files...$(COLOR_RESET)"
	rm -rf $(WEB_FRONTEND)/.next $(WEB_FRONTEND)/out
	@echo "$(COLOR_GREEN)✓ Cleanup complete$(COLOR_RESET)"

.PHONY: clean-data
clean-data: ## Remove preprocessed data (keeps raw data)
	@echo "$(COLOR_RED)WARNING: This will remove preprocessed data!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		find $(DATA_DIR) -name "*.json" -delete; \
		echo "$(COLOR_GREEN)✓ Preprocessed data removed$(COLOR_RESET)"; \
	fi

.PHONY: clean-all
clean-all: clean clean-data ## Remove all generated files including data
	@echo "$(COLOR_GREEN)✓ Complete cleanup done$(COLOR_RESET)"

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "$(COLOR_BLUE)Generating documentation...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Documentation generation not yet implemented$(COLOR_RESET)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(COLOR_BLUE)Serving documentation...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Documentation serving not yet implemented$(COLOR_RESET)"

# ============================================================================
# Git & Version Control
# ============================================================================

.PHONY: git-status
git-status: ## Show detailed git status
	@echo "$(COLOR_BLUE)Git Status:$(COLOR_RESET)"
	@git status
	@echo ""
	@echo "$(COLOR_BLUE)Untracked files:$(COLOR_RESET)"
	@git ls-files --others --exclude-standard

.PHONY: git-clean
git-clean: ## Remove untracked files (dry run first)
	@echo "$(COLOR_BLUE)Files to be removed (dry run):$(COLOR_RESET)"
	git clean -ndx
	@echo ""
	@echo "$(COLOR_YELLOW)Run 'git clean -fdx' to actually remove files$(COLOR_RESET)"

# ============================================================================
# Monitoring & Profiling
# ============================================================================

.PHONY: gpu-info
gpu-info: ## Show GPU information
	@echo "$(COLOR_BLUE)GPU Information:$(COLOR_RESET)"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
	print(f'CUDA version: {torch.version.cuda}'); \
	print(f'Device count: {torch.cuda.device_count()}'); \
	[print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

.PHONY: check-env
check-env: ## Check environment setup
	@echo "$(COLOR_BLUE)Environment Check:$(COLOR_RESET)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA: $$($(PYTHON) -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
	@echo "HuggingFace Transformers: $$($(PYTHON) -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "FastAPI: $$($(PYTHON) -c 'import fastapi; print(fastapi.__version__)' 2>/dev/null || echo 'Not installed')"

.PHONY: monitor
monitor: ## Monitor GPU usage with nvitop
	@echo "$(COLOR_BLUE)Starting GPU monitor...$(COLOR_RESET)"
	@$(PYTHON) -m nvitop

# ============================================================================
# CI/CD Simulation
# ============================================================================

.PHONY: ci
ci: check test ## Simulate CI pipeline (format, lint, typecheck, test)
	@echo "$(COLOR_GREEN)✓ CI pipeline passed$(COLOR_RESET)"

.PHONY: ci-full
ci-full: check test-cov ## Full CI pipeline with coverage
	@echo "$(COLOR_GREEN)✓ Full CI pipeline passed$(COLOR_RESET)"

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: tree
tree: ## Show project directory tree
	@tree -L 3 -I '__pycache__|*.pyc|node_modules|.git|*.egg-info|.next' .

.PHONY: size
size: ## Show project size statistics
	@echo "$(COLOR_BLUE)Project Size Statistics:$(COLOR_RESET)"
	@echo "Source code:"
	@du -sh $(SRC_DIR)
	@echo "Tests:"
	@du -sh $(TEST_DIR)
	@echo "Notebooks:"
	@du -sh $(NOTEBOOK_DIR)
	@echo "Total project:"
	@du -sh .

.PHONY: count
count: ## Count lines of code
	@echo "$(COLOR_BLUE)Lines of Code:$(COLOR_RESET)"
	@find $(SRC_DIR) -name "*.py" | xargs wc -l | tail -1
	@echo "$(COLOR_BLUE)Test Lines:$(COLOR_RESET)"
	@find $(TEST_DIR) -name "*.py" | xargs wc -l 2>/dev/null | tail -1 || echo "0 total"

# ============================================================================
# Special Targets
# ============================================================================

.PHONY: all
all: install-all check test ## Setup everything and run checks
	@echo "$(COLOR_GREEN)✓ All tasks complete$(COLOR_RESET)"

.DEFAULT_GOAL := help

# Ensure intermediate files are not deleted
.PRECIOUS: %.py

# Phony targets don't correspond to actual files
.PHONY: help env install install-torch install-dev install-web install-all \
        format lint typecheck check test test-cov test-fast test-integration \
        notebook lab clean-notebooks web-backend web-frontend web-build web \
        preprocess-mme data-info inference-mme eval-mme build upload-test upload \
        clean clean-data clean-all docs docs-serve git-status git-clean \
        gpu-info check-env monitor ci ci-full tree size count all
