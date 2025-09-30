# Contributing to DeepHallu

Thank you for your interest in contributing to DeepHallu! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub

2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/DeepHallu.git
   cd DeepHallu
   export PROJECT_ROOT=$(pwd)
   ```

3. Create a conda environment:
   ```bash
   conda create -n deephallu python=3.12
   conda activate deephallu
   ```

4. Install PyTorch according to your own compute configuration.
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
   ```

5. Install development dependencies:
   ```bash
   pip install -e .
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for your changes
4. Run tests to ensure everything works:
   ```bash
   pytest
   ```
5. Run code formatting:
   ```bash
   black deephallu tests
   isort deephallu tests
   ```
6. Run linting:
   ```bash
   flake8 deephallu tests
   mypy deephallu
   ```
7. Commit your changes:
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
8. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. Create a Pull Request on GitHub

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [isort](https://isort.readthedocs.io/) for import sorting
- We use [flake8](https://flake8.pycqa.org/) for linting
- We use [mypy](https://mypy.readthedocs.io/) for type checking

```toml
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=21.0",
    "isort>=5.0",
    "flake8>=3.8",
    "mypy>=0.900",
    "pre-commit>=2.15",
]
docs = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "myst-parser>=0.15",
]
```

## Testing

- All new code should include appropriate tests
- Tests should be placed in the `tests/` directory
- Use descriptive test names that explain what is being tested
- Aim for high test coverage

## Documentation

- Document all public functions, classes, and methods
- Use Google-style docstrings
- Update README.md

## Issue Reporting

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

## Feature Requests

Before implementing new features:
1. Open an issue to discuss the feature
2. Wait for feedback from maintainers
3. Implement the feature following these guidelines

## Questions?

If you have questions about contributing, please open an issue with the "question" label.
Thank you for contributing to DeepHallu!