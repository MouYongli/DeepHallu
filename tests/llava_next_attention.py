import os

os.environ["HF_HOME"] = "/DATA2/HuggingFace"

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import numpy as np
