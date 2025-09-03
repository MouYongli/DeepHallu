# DeepHallu

On the anaylsis and mitigation of hallucinations in Vision Language Models (VLMs).

## About

DeepHallu is a research project focused on developing advanced techniques for analysing and mitigating hallucinations in VLMs.

## Features

- **Hallucination Analysis**: Advanced algorithms to analyse the hallucination in VLMS, for example, identifying hallucinated content in model outputs and the patterns inside the VLMs.
- **Mitigation Strategies**: Techniques to reduce hallucination rates in VLMs.
- **Evaluation Metrics**: Comprehensive benchmarks for measuring hallucination rates and model reliability

## Datasets and Benchmarks

The following datasets and benchmarks are used in the project:
1. MME
2. VQA v2.0
3. CHAIR
4. POPE
5. Llava Bench in the Wild

Details of the datasets and benchmarks are in the [data/datasets](data/datasets) directory.

## Installation

1. Clone the repository
```bash
# Clone the repository
git clone https://github.com/MouYongli/DeepHallu.git
cd DeepHallu
export PROJECT_ROOT=$(pwd)
```

2. Setup the environment
```bash
conda create -n deephallu python=3.12
conda activate deephallu
```

3. Install PyTorch according to your own compute configuration.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

4. Install the package locally
```bash
pip install -e .  
```



## Quick Start

Coming soon! This project is currently in the initial setup phase.

## Project Structure

```
DeepHallu/
├── data/               # Sample datasets and benchmarks
├── docs/               # Documentation
├── examples/           # Example scripts and notebooks
├── notebooks/          # Notebooks
├── scripts/            # Scripts
├── src/                # Main package source code
|   └── deephallu/
|       ├── __init__.py
|       ├── data/
|       └── models/
├── tests/              # Unit tests
├── README.md           # README
├── pyproject.toml      # Project configuration
├── requirements.txt    # Requirements
├── .gitignore          # Git ignore
└── LICENSE             # License
```

## Contributing

We welcome contributions to DeepHallu! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Citation

If you use DeepHallu in your research, please cite:

```bibtex
@article{deephallu2025,
  author = {Mou, Yongli},
  title = {DeepHallu: On the Analysis and Mitigation of Hallucinations in Vision Language Models},
  year = {2025},
  journal = {arXiv preprint arXiv:2509.00000},
}
``` -->

## Contact

- Author: Yongli Mou, Er Jin, Johannes Stegmaier, Shin'ichi Satoh, Stefan Decker
- Email: mou@dbis.rwth-aachen.de
- GitHub: [@MouYongli](https://github.com/MouYongli)
- Website: [https://mouyongli.github.io/](https://mouyongli.github.io/)

---

**Note**: This project is currently in active development. Features and API may change.