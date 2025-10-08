# DeepHallu System Architecture

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Core Modules](#core-modules)
- [Data Flow](#data-flow)
- [Development Workflow](#development-workflow)
- [Deployment](#deployment)

## System Overview

DeepHallu is a research-oriented system for analyzing and mitigating hallucinations in Vision Language Models (VLMs). The system provides:

1. **Dataset Management**: Unified interface for multiple VLM benchmarks (MME, VQA v2.0, CHAIR, POPE, LLaVA Bench)
2. **Model Integration**: Support for state-of-the-art VLMs (LLaVA-Next, DeepSeek-VL2, Qwen2.5-VL)
3. **Analysis Tools**: Token-to-patch mapping, attention visualization, hallucination detection
4. **Web Interface**: Interactive platform for visualizing model behavior and hallucination patterns
5. **Research Workflow**: Jupyter notebooks for exploratory analysis and benchmarking

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌──────────────────┐              ┌──────────────────────┐     │
│  │  Jupyter         │              │  Web Application     │     │
│  │  Notebooks       │              │  (Next.js + FastAPI) │     │
│  └──────────────────┘              └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Core Python Package                        │
│  ┌──────────┬──────────┬────────────────┬──────────┬──────────┐ │
│  │   Data   │  Models  │      Data      │ Analytics│ Inference│ │
│  │  Loaders │ (VLMs)   │  Preprocessing │          │          │ │
│  └──────────┴──────────┴────────────────┴──────────┴──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌──────────┬──────────┬──────────┬──────────────────────┐      │
│  │ PyTorch  │  CUDA    │ Hugging  │  Datasets            │      │
│  │          │          │  Face    │  (MME, VQA, etc.)    │      │
│  └──────────┴──────────┴──────────┴──────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Tier Architecture

1. **Presentation Layer**
   - Jupyter notebooks for research and experimentation
   - Next.js web frontend for interactive visualization
   - FastAPI backend serving model inference and data APIs

2. **Application Layer**
   - Data preprocessing and loading modules
   - VLM model wrappers and analysis tools
   - Token-to-patch mapping and attention mechanisms
   - Hallucination detection and analysis algorithms

3. **Data Layer**
   - Benchmark datasets (MME, VQA v2.0, CHAIR, POPE, LLaVA Bench)
   - Preprocessed JSON data
   - HuggingFace model cache (`/DATA2/HuggingFace`)

## Project Structure

```
DeepHallu/
├── src/deephallu/              # Main package
│   ├── data/                   # Dataset loaders
│   │   ├── __init__.py
│   │   └── mme.py             # MME benchmark dataset
│   ├── preprocessing/          # Data preprocessing
│   │   ├── __init__.py
│   │   └── mme.py             # MME preprocessor
│   ├── models/                 # VLM integrations
│   │   ├── llava_next.py      # LLaVA-Next model
│   │   └── llava_next_t2p_mapper.py  # Token-to-patch mapper
│   ├── analysis/               # Analysis tools
│   │   ├── __init__.py
│   │   └── image_processor_analysis.py
│   ├── inference/              # Inference scripts (planned)
│   └── web/                    # Web application
│       ├── backend/            # FastAPI backend
│       │   ├── app.py
│       │   ├── api/           # API endpoints
│       │   ├── core/          # Configuration
│       │   ├── models/        # Pydantic models
│       │   ├── services/      # Business logic
│       │   └── utils/
│       └── frontend/           # Next.js frontend
│           ├── src/
│           │   ├── app/       # Next.js app router
│           │   ├── components/
│           │   └── lib/
│           └── package.json
├── notebooks/                  # Research notebooks
│   ├── baseline_*.ipynb       # Baseline model notebooks
│   ├── benchmark_*.ipynb      # Benchmark evaluations
│   ├── datasets_*.ipynb       # Dataset exploration
│   └── exploration_*.ipynb    # Exploratory analysis
├── data/                       # Data directory
│   └── mme -> /path/to/MME     # Symlink to datasets
├── docs/                       # Documentation
│   ├── research/              # Research documentation
│   └── *.md                   # Technical docs
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
├── examples/                   # Example usage
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
├── Makefile                   # Build and task automation
└── README.md                  # Project overview
```

## Technology Stack

### Backend
- **Python 3.12**: Core programming language
- **PyTorch 2.4.0 + CUDA 12.9**: Deep learning framework
- **HuggingFace Transformers**: VLM model integration
- **FastAPI**: Web API framework
- **Uvicorn**: ASGI server

### Frontend
- **Next.js**: React framework for web interface
- **TypeScript**: Type-safe frontend development
- **Tailwind CSS**: Styling framework
- **D3.js / Three.js**: Data visualization (planned)

### Development Tools
- **Conda**: Environment management
- **Black + isort**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Jupyter**: Interactive development

### Infrastructure
- **Git**: Version control
- **HuggingFace Hub**: Model hosting
- **CUDA**: GPU acceleration

## Core Modules

### 1. Data Module (`deephallu.data`)

**Purpose**: Unified interface for loading benchmark datasets

**Components**:
- `MMEDataset`: PyTorch dataset for MME benchmark
  - 14 categories (OCR, Scene Text, Artwork, etc.)
  - Returns: `(image, id, image_name, category, question, answer)`

**Design Pattern**: Factory pattern for dataset creation

### 2. Preprocessing Module (`deephallu.preprocessing`)

**Purpose**: Convert raw benchmark data to standardized JSON format

**Components**:
- `MMEPreprocessor`: Converts MME directory structure to JSON
  - Input: Raw dataset folders
  - Output: Standardized JSON with metadata

**Data Flow**:
```
Raw Dataset → Preprocessor → Standardized JSON → Dataset Loader
```

### 3. Models Module (`deephallu.models`)

**Purpose**: VLM model wrappers and analysis tools

**Components**:
- `llava_next.py`: LLaVA-Next model integration
  - Multi-scale image processing (336×336 base resolution)
  - Token generation and inference
- `llava_next_t2p_mapper.py`: Token-to-patch mapping
  - Maps text tokens to image patches
  - Enables attention analysis

**Key Features**:
- `torch.float16` precision for memory efficiency
- CUDA device placement (`cuda:0`)
- Low CPU memory usage optimization

### 4. Analysis Module (`deephallu.analysis`)

**Purpose**: Analyze model internals and hallucination patterns

**Components**:
- `image_processor_analysis.py`: Image processing pipeline analysis
  - Multi-scale transformation analysis
  - Patch-level feature extraction

### 5. Web Module (`deephallu.web`)

**Purpose**: Interactive web platform for hallucination analysis

**Backend (`backend/`)** :
- FastAPI application with CORS support
- RESTful API endpoints for datasets and models
- Configuration management via Pydantic

**Frontend (`frontend/`)**:
- Next.js application with TypeScript
- Interactive visualization components
- Real-time model output display

**Features** (from `web/CLAUDE.md`):
- Image upload and dataset selection
- Model selection and inference
- Token hover interaction showing:
  - Causal attention distribution (heatmap)
  - Next-token prediction distribution (Top-K bar chart)
- Attention threshold highlighting
- Dynamic patch visualization

## Data Flow

### 1. Dataset Preprocessing Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Raw Dataset  │ ──→ │ Preprocessor │ ──→ │ JSON Output  │
│ (MME/VQA)    │     │ (mme.py)     │     │ (standardized)│
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2. Model Inference Flow

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Dataset  │ ──→ │ DataLoader   │ ──→ │ VLM Model    │
│ Loader   │     │ (PyTorch)    │     │ (LLaVA-Next) │
└──────────┘     └──────────────┘     └──────────────┘
                                              ↓
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Evaluation   │ ←── │ Analysis     │ ←── │ Outputs:     │
│ & Metrics    │     │ Tools        │     │ - Tokens     │
└──────────────┘     └──────────────┘     │ - Attention  │
                                            │ - Logits     │
                                            └──────────────┘
```

### 3. Web Application Flow

```
┌───────────┐     ┌─────────────┐     ┌──────────────┐
│ Frontend  │ ──→ │ FastAPI     │ ──→ │ Model        │
│ (Next.js) │     │ Backend     │     │ Inference    │
└───────────┘     └─────────────┘     └──────────────┘
     ↑                                        ↓
     │                                  ┌──────────────┐
     └──────────────────────────────── │ Response:    │
                                        │ - Answer     │
                                        │ - Attention  │
                                        │ - Predictions│
                                        └──────────────┘
```

## Development Workflow

### 1. Research Workflow

```
Hypothesis → Notebook Exploration → Implementation → Testing → Integration
    ↓              ↓                     ↓              ↓           ↓
  Docs        baseline_*.ipynb    src/deephallu/   pytest      Web UI
```

### 2. Feature Development Cycle

1. **Exploration**: Use Jupyter notebooks to prototype
2. **Implementation**: Move stable code to `src/deephallu/`
3. **Testing**: Write unit tests in `tests/`
4. **Documentation**: Update relevant `.md` files
5. **Integration**: Integrate with web interface if applicable

### 3. Code Quality Pipeline

```bash
# Format code
black src/deephallu tests
isort src/deephallu tests

# Check code quality
flake8 src/deephallu tests
mypy src/deephallu

# Run tests
pytest --cov=deephallu --cov-report=term-missing
```

## Deployment

### Development Environment

```bash
# 1. Create conda environment
conda create -n deephallu python=3.12
conda activate deephallu

# 2. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 3. Install package in editable mode
pip install -e .

# 4. Set HuggingFace cache
export HF_HOME=/DATA2/HuggingFace
```

### Running Web Application

```bash
# Backend (FastAPI)
cd src/deephallu/web/backend
python app.py

# Frontend (Next.js)
cd src/deephallu/web/frontend
npm install
npm run dev
```

### Running Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Configuration

### Environment Variables

- `HF_HOME`: HuggingFace model cache directory (default: `/DATA2/HuggingFace`)
- `PROJECT_ROOT`: Project root directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Model Configuration

Models are configured with:
- `torch_dtype=torch.float16`: Memory optimization
- `low_cpu_mem_usage=True`: Reduce CPU memory footprint
- `device_map="cuda:0"`: Explicit GPU placement

### Dataset Paths

- MME: `data/mme/MME_Benchmark_release_version/MME_Benchmark/`
- Preprocessed data: `data/<dataset>/preprocessed.json`

## Performance Considerations

1. **GPU Memory Management**
   - Use FP16 precision for large models
   - Enable gradient checkpointing for training
   - Batch processing for dataset evaluation

2. **Data Loading**
   - Use PyTorch DataLoader with multiple workers
   - Cache preprocessed data to avoid repeated processing

3. **Model Caching**
   - HuggingFace models cached in `/path/to/HuggingFace`
   - Avoid repeated downloads

4. **Web Application**
   - Frontend response < 200ms (interaction level)
   - Model inference time depends on hardware
   - CORS enabled for local development

## Security & Best Practices

1. **Code Quality**
   - Type hints required (enforced by mypy)
   - Black formatting (88-char line length)
   - Import sorting with isort
   - Test coverage tracking

2. **Version Control**
   - Git branching strategy (main branch)
   - Commit message conventions
   - Pull request reviews

3. **Data Management**
   - Symbolic links for large datasets
   - Separate data storage (`/DATA2/DeepHallu/`)
   - Version control for preprocessing scripts

4. **Documentation**
   - Code-level docstrings
   - Research documentation in `docs/research/`
   - CLAUDE.md for AI assistant guidance
   - System architecture documentation (this file)

## Future Enhancements

1. **Model Support**
   - Add more VLM architectures
   - Support for custom fine-tuned models

2. **Datasets**
   - Additional benchmarks (COCO-Captions, NoCaps, etc.)
   - Custom dataset upload functionality

3. **Analysis Tools**
   - Advanced attention visualization
   - Hallucination pattern clustering
   - Automated hallucination detection

4. **Web Interface**
   - 3D attention visualization with Three.js
   - Real-time model comparison
   - Export analysis results

5. **Infrastructure**
   - Docker containerization
   - Multi-GPU support
   - Cloud deployment options

## References

- Project Repository: https://github.com/MouYongli/DeepHallu
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch Documentation: https://pytorch.org/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- Next.js Documentation: https://nextjs.org/docs

---

**Last Updated**: 2025-10-06
**Version**: 1.0.0
**Maintainer**: Yongli Mou (mou@dbis.rwth-aachen.de)
