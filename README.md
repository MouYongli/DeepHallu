# DeepHallu# Deep Foundation Models for Single Cell RNA Sequencing

This is official repo for "DeepHallu: Hallucination in Vision Language Models" by DBIS at RWTH Aachen University 
[Yongli Mou*](mou@dbis.rwth-aachen.de), Jan Ebigt, Stefan Decker

## Python Environment Setup

1. conda environment
```
conda create --name deephallu python=3.11
conda activate deephallu
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
pip install -r requirements.txt
pip install -e .
```


## Datasets

VQA v2

