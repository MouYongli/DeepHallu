
import os
os.environ["HF_HOME"] = "/DATA2/HuggingFace"
import os.path as osp
import numpy as np
import torch
from PIL import Image

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from deephallu.data.mme import MMEDataset

def main():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")

    dataset = MMEDataset()
    image, id, image_name, category, question, answer = dataset[0]
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=1000)
    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()