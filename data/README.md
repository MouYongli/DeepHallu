# Data Directory

This directory will contain sample datasets and benchmarks for hallucination detection and mitigation research.

## Planned Datasets

- **Text Hallucination**: Examples of model-generated text with hallucinated facts
- **Image Captioning**: Vision-language model outputs with visual hallucinations
- **Question Answering**: Q&A pairs with factually incorrect responses
- **Benchmarks**: Standard evaluation datasets for reproducible research

## Data Format

Datasets will follow a standardized JSON format:
```json
{
  "id": "unique_identifier",
  "input": "original_input_data", 
  "output": "model_generated_output",
  "ground_truth": "reference_truth",
  "hallucination_label": true/false,
  "metadata": {
    "model": "source_model_name",
    "domain": "application_domain"
  }
}
```

## Usage

Place your datasets in this directory and use the DeepHallu utilities to load and process them:

```python
from deephallu.utils import load_dataset
dataset = load_dataset("data/sample_hallucinations.json")
```