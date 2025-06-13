# DeepHallu: Deep Foundation Models for Single Cell RNA Sequencing

[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-green)](https://pytorch.org/get-started/locally/)

[![Forks](https://img.shields.io/github/forks/MouYongli/DeepHallu?style=social)](https://github.com/MouYongli/DeepHallu/network/members)
[![Stars](https://img.shields.io/github/stars/MouYongli/DeepHallu?style=social)](https://github.com/MouYongli/DeepHallu/stargazers)
[![Issues](https://img.shields.io/github/issues/MouYongli/DeepHallu)](https://github.com/MouYongli/DeepHallu/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/MouYongli/DeepHallu)](https://github.com/MouYongli/DeepHallu/pulls)
[![Contributors](https://img.shields.io/github/contributors/MouYongli/DeepHallu)](https://github.com/MouYongli/DeepHallu/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/MouYongli/DeepHallu)](https://github.com/MouYongli/DeepHallu/commits/main)
<!-- [![Build Status](https://img.shields.io/github/actions/workflow/status/MouYongli/DeepHallu/ci.yml)](https://github.com/MouYongli/DeepHallu/actions)
[![Code Quality](https://img.shields.io/lgtm/grade/python/g/MouYongli/DeepHallu.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MouYongli/DeepHallu/context:python) -->

[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://hub.docker.com/r/YOUR_DOCKER_IMAGE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-yellow)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)


[![WeChat](https://img.shields.io/badge/WeChat-公众号名称-green)](https://your-wechat-link.com)
[![Weibo](https://img.shields.io/badge/Weibo-关注-red)](https://weibo.com/YOUR_WEIBO_LINK)
<!-- [![Discord](https://img.shields.io/discord/YOUR_DISCORD_SERVER_ID?label=Discord&logo=discord&color=5865F2)](https://discord.gg/YOUR_INVITE_LINK) -->
<!-- [![Twitter](https://img.shields.io/twitter/follow/YOUR_TWITTER_HANDLE?style=social)](https://twitter.com/YOUR_TWITTER_HANDLE) -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


This is official repo for "DeepHallu: Hallucination in Vision Language Models" by DBIS at RWTH Aachen University 
([Yongli Mou*](mou@dbis.rwth-aachen.de), Jan Ebigt, Er Jin, Shin'ichi Satoh and Stefan Decker)


## 1. Overview

DeepHallu is a research project focused on the analysis and mitigation of hallucination in Vision Language Models (VLMs).

## 2. Installation

### 2.1 Manual Installation

We use DeepSeek-VL2 and Qwen2.5-VL as our baselines and use conda to create individual python environments for each baseline due to different requirements of the package dependencies.

```bash
git clone https://github.com/MouYongli/DeepHallu.git
cd DeepHallu
export PROJECT_ROOT=$(pwd)
```

#### DeepSeek-VL2
```bash
cd $PROJECT_ROOT/baselines
mkdir deepseek
# Clone the DeepSeek-VL2 repository
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
mv DeepSeek-VL2/* deepseek
rm -rf DeepSeek-VL2
# Update requirements.txt in DeepSeek-VL2 folder
cp requirements.deepseek.txt deepseek/requirements.txt
#  Update pyproject.toml in DeepSeek-VL2 folder
cp pyproject.deepseek.toml deepseek/pyproject.toml
# Install dependencies and install the deepseek-vl2 package
cd deepseek
# Create a new conda environment for DeepSeek-VL2
conda create --name deepseekenv python=3.10
conda activate deepseekenv
pip install -r requirements.txt
pip install -e .
```
Optional:

```bash
# Install jupyterlab and ipykernel for Jupyter Notebooks
conda install -c conda-forge jupyterlab
conda install ipykernel
# Install PyTorch with CUDA 12.6
# For other version, please refer to https://pytorch.org/get-started/locally, for example:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch torchvision torchaudio 
```

#### Qwen2.5-VL

```bash
cd $PROJECT_ROOT/baselines
mkdir qwen
conda create --name qwenenv python=3.10
conda activate qwenenv
cp requirements.qwen.txt qwen/requirements.txt
cd qwen
pip install -r requirements.txt
```

Optional:

```bash
# Install jupyterlab and ipykernel for Jupyter Notebooks
conda install -c conda-forge jupyterlab
conda install ipykernel
# Install PyTorch with CUDA 12.6
# For other version, please refer to https://pytorch.org/get-started/locally, for example:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch torchvision torchaudio 
```


# Install our project and dependencies


```bash
cd $PROJECT_ROOT
conda activate deepseekenv
pip install -e .
conda activate qwenenv
pip install -e .
```

### 2.2 Docker Installation

```bash
# docker pull mouyongli/deephallu:latest
# docker run--gpus all -it --rm mouyongli/deephallu:latest
```

### 2.3 Colab Installation

```bash
# !pip install git+https://github.com/MouYongli/DeepHallu.git
```

## 3. Datasets

We use the following datasets for our experiments:

- [VQAv2](https://visualqa.org/vqa_download.html)
- [POPE](https://github.com/google-research/google-research/tree/master/pope)

```bash
cd $PROJECT_ROOT/scripts/data
python prepare_data.py
``` 









