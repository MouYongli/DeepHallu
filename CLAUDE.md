# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepHallu is a research project focused on analyzing and mitigating hallucinations in Vision Language Models (VLMs). The project provides tools for:
- Detecting hallucinations in VLM outputs
- Implementing mitigation strategies
- Evaluating model reliability with comprehensive benchmarks
- Supporting research with multiple datasets (MME, VQA v2.0, CHAIR, POPE, LLaVA Bench)

## Environment Setup

The project uses Python 3.12 with conda for environment management:

```bash
conda create -n deephallu python=3.12
conda activate deephallu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -e .
```

HuggingFace models are cached in `/DATA2/HuggingFace` (set via `HF_HOME` environment variable).

## Common Development Commands

### Package Management
```bash
pip install -e .                    # Install package in development mode
```

### Code Quality
```bash
black deephallu tests               # Format code
isort deephallu tests              # Sort imports  
flake8 deephallu tests            # Lint code
mypy deephallu                    # Type checking
pytest                            # Run tests
pytest --cov=deephallu --cov-report=term-missing  # Run tests with coverage
```

### Cleanup
```bash
make clean                        # Remove cache files, build artifacts
```

## Architecture Overview

### Package Structure
```
src/deephallu/
├── __init__.py                   # Package initialization
├── data/                         # Dataset loaders and utilities
│   ├── __init__.py
│   └── mme.py                   # MME benchmark dataset loader
├── models/                       # VLM model implementations and analysis
│   ├── llava_next.py           # LLaVA-Next model integration
│   └── image_processor_analysis.py  # Image processing pipeline analysis
└── preprocessing/               # Data preprocessing utilities
    ├── __init__.py
    └── mme.py                  # MME benchmark preprocessor
```

### Key Components

**Dataset Management:**
- `MMEDataset`: PyTorch dataset for MME benchmark with 14 categories
- `MMEPreprocessor`: Converts MME data structure to standardized JSON format
- Data path: `data/mme/MME_Benchmark_release_version/MME_Benchmark/`

**VLM Integration:**
- LLaVA-Next model integration with HuggingFace transformers
- Image processor analysis for multi-scale processing (336x336 base resolution)
- Support for various VLM architectures via notebooks

**Research Workflow:**
- Notebooks in `notebooks/` contain baseline implementations and benchmark evaluations
- Each VLM (LLaVA, DeepSeek-VL2, Qwen2.5-VL) has dedicated baseline notebooks
- Dataset exploration notebooks for each supported benchmark

### Data Flow

1. **Preprocessing:** Raw benchmark data → standardized JSON format via preprocessing modules
2. **Loading:** Dataset classes load preprocessed data and return (image, metadata, question, answer) tuples
3. **Model Processing:** Images processed through VLM-specific pipelines (e.g., multi-scale for LLaVA-Next)
4. **Evaluation:** Model outputs compared against ground truth for hallucination analysis

## Development Notes

### VLM Model Integration
- Models are loaded with specific configurations (e.g., `torch_dtype=torch.float16`, `low_cpu_mem_usage=True`)
- CUDA device placement is explicit (usually `cuda:0`)
- Image processing involves multi-scale transformations and token generation

### Dataset Conventions
- All datasets follow PyTorch Dataset interface
- Return format: `(image, id, image_name, category, question, answer)`
- Images are PIL Image objects, not tensors

### Research Documentation
- Extensive research documentation in `docs/research/` including problem formulation and literature reviews
- Papers and references stored in `docs/research/papers/`
- Chinese documentation available for some components

### Code Quality Standards
- Type hints required (mypy configuration in pyproject.toml)
- Black formatting with 88-character line length
- Test coverage tracking enabled
- Import sorting with isort using black profile

## Testing Strategy

- Tests located in `tests/` directory
- Use pytest with coverage reporting
- Test naming convention: `test_*.py` files with `test_*` functions
- Coverage target: comprehensive coverage for core functionality

## Performance Considerations

- Large VLMs require significant GPU memory
- Image processing can be memory-intensive due to multi-scale approaches
- HuggingFace model caching reduces download times for repeated experiments
- Consider batch processing for dataset evaluation to optimize GPU utilization