# DeepHallu: Deep Foundation Models for Single Cell RNA Sequencing

This is official repo for "DeepHallu: Hallucination in Vision Language Models" by DBIS at RWTH Aachen University 
[Yongli Mou*](mou@dbis.rwth-aachen.de), Jan Ebigt, Stefan Decker

## Python Environment Setup

1. conda environment
```bash
conda create --name deephallu python=3.10
conda activate deephallu
```

We also create individual environments for each baseline due to different requirements of the package dependencies.
```bash
conda create --name deepseek-vl2 python=3.10
conda activate deephallu
```
```bash
conda create --name qwen2.5-vl python=3.10
conda activate deephallu
```

2. jupyter lab and kernel
```bash
conda install -c conda-forge jupyterlab
conda install ipykernel
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # torch==2.6.0+cu124, torchaudio==2.6.0+cu124, torchvision==0.21.0+cu124
pip install -e .
```


## Datasets

VQA v2

