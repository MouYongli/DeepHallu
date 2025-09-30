# LlavaNext

## 1. Image Processor

假设我们的图像大小为 (808, 1100)，我们输入的图像和文本会经过以下处理：

```python
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from deephallu.data.mme import MMEDataset
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")

image_processor = processor.image_processor
tokenizer = processor.tokenizer
vision_tower = model.vision_tower
image = Image.open("path/to/image.jpg")
question = "Is this artwork created by linard, jacques? Please answer yes or no."
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
```
通过`processor`，我们得到了`inputs`，这是一个字典，包含了输入的文本和图像的编码结果。

`inputs` 是一个字典，包含了以下键值对：
- `input_ids`: 输入的文本和图像的编码结果，形状为 [1, 2369]
- `attention_mask`: 输入的文本和图像的注意力掩码，形状为 [1, 2369]
- `pixel_values`: 输入的图像的像素值，形状为 [1, 5, 3, 336, 336]
- `image_sizes`: 输入的图像的大小，形状为 [1, 2]

那么我们进一步拆解`LlavaNextProcessor`的代码，看看它是如何处理图像和文本的。主要的代码有两个部分：
- `transformers/src/transformers/models/llava_next/processing_llava_next.py`
- `transformers/src/transformers/models/llava_next/image_processing_llava_next.py`

### 1.1 LlavaNextProcessor

#### 参数

```python
for k, v in processor.image_processor.__dict__.items():
    print(f'{k} = {v}')
```

得到的结果如下：
```
patch_size = 14
num_additional_image_tokens = 1
vision_feature_select_strategy = default
image_token = <image>
image_token_id = 32000
chat_template = {% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>
' + message['content'][0]['text'] + '
<</SYS>>

' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>
' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '</s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
audio_tokenizer = None
image_processor = LlavaNextImageProcessor {
  "aspect_ratio_setting": "anyres",
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_grid_pinpoints": [
    [
      336,
      672
    ],
    [
      672,
      336
    ],
    [
      672,
      672
    ],
    [
      1008,
      336
    ],
    [
      336,
      1008
    ]
  ],
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "LlavaNextImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "processor_class": "LlavaNextProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 336
  }
}

tokenizer = LlamaTokenizerFast(name_or_path='llava-hf/llava-v1.6-mistral-7b-hf', vocab_size=32000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'image_token': '<image>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32000: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32001: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
```

#### Prompt的生成
我们通过调用`LlavaNextProcessor`的`apply_chat_template`方法，得到了`prompt`，这是一个字符串，包含了输入的文本和图像的编码结果。

```python
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)
```

得到的结果如下：
```[INST] <image>
Is this artwork created by linard, jacques? Please answer yes or no. [/INST]
```
#### 图像和文本的处理
然后我们通过调用`LlavaNextProcessor`的`__call__`方法，得到了`inputs`，这是一个字典，包含了输入的文本和图像的编码结果。
```python
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
print(inputs)
```

得到的结果如下：
```
```
input_ids torch.Size([1, 2369])
attention_mask torch.Size([1, 2369])
pixel_values torch.Size([1, 5, 3, 336, 336])
image_sizes torch.Size([1, 2])
```
##### 图像的处理

对其中的过程进行进一步的拆解，我们发现`LlavaNextProcessor`的`__call__`方法调用了`image_processor`的`__call__`方法，得到了`image_inputs`，这是一个字典，包含了输入的图像的编码结果。
```python
image_inputs = processor.image_processor(image)
```
得到的结果是：
```
pixel_values：(5, 3, 336, 336)
image_sizes：[(808, 1100)]
```

对于`LlavaNextProcessor`的`image_processor`，它是一个`LlavaNextImageProcessor`，我们进一步拆解`LlavaNextImageProcessor`的代码，看看它是如何处理图像的。

### 1.2 LlavaNextImageProcessor

#### 参数

```python
for k, v in image_processor.__dict__.items():
    print(f'{k} = {v}')
```
得到的结果如下：
```
_processor_class = LlavaNextProcessor
aspect_ratio_setting = anyres
image_processor_type = LlavaNextImageProcessor
do_resize = True
size = {'shortest_edge': 336}
image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
resample = 3
do_center_crop = True
crop_size = {'height': 336, 'width': 336}
do_rescale = True
rescale_factor = 0.00392156862745098
do_normalize = True
image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]
do_pad = True
do_convert_rgb = True
```

我们现在来探讨一下，`image_processor`是如何处理图像的。主要是两段代码：

```python
image_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=(size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"])),
                patch_size=crop_size["height"],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )
```

-  `select_best_resolution`
这个函数定义在`transformers/src/transformers/image_processing_utils.py`中。

    主要参数：
    - original_size (tuple)：原始图像的尺寸，格式是 (height, width)。
    - possible_resolutions (list)：候选分辨率列表，每个分辨率也是 (height, width) 格式。

    核心思路：

    代码在选择“最佳分辨率”时，考虑了两个因素：
    - 有效分辨率 (effective_resolution)
        - 指图像在目标分辨率下缩放后，实际能利用的像素数量。
        - 如果目标分辨率比原图大，图像会放大；比原图小，图像会缩小。
        - 有效分辨率就是缩放后实际能填满的像素面积。
    - 浪费分辨率 (wasted_resolution)
        - 指目标分辨率里没有被图像内容填满的像素数量。
        - 计算公式：目标分辨率面积 - 有效分辨率。

    选择策略：
    - 优先 最大化有效分辨率（保证清晰度尽可能高）；
    - 如果有多个分辨率的有效分辨率相同，再选择 浪费最少的。

-  `_resize_for_patching`
通过best resolution来resize图像。resize后的图像大小为(494, 672)。

-  `_pad_for_patching`
通过best resolution来pad图像。pad后的图像大小为(672, 672)。

-  `divide_to_patches`
通过patch size来divide图像获取list。divide后的图像大小为[(336, 336) * 4]。

-  `reize`原始的图片。resize后的图像大小为(336， 336)。然后append到list中。

```python
pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )

```
在这一步，我们要解决的问题是，在LlavaNextProcessor中，如何获取图像的token数量（因为有padding的像素）:

```python
num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
```
对于我们的图像来说：
因为patch图像的大小为(336, 336)，`patch_size`为14，`orig_height` = 808, `orig_width` = 1100, best resolution的`height` = 672, `width` = 672,
所以`patches_height` = 336 / 14 = 24, `patches_width` = 336 / 14 = 24，`scale_height` = 672 / 336 = 2，`scale_width` = 672 / 336 = 2。

```python
base_features = patches_height * patches_width + num_additional_image_tokens
```

对于`unpadded_features`，含义：经过缩放，去掉 padding后，图片真正能分割出的 patch 数量。

对于`newline_features`，含义：相当于 每行的结束标记数。在 LLaVA-NeXT 里，为了让模型更好地理解图像的二维结构（像看“行和列”一样），每一行 patch 结束时会额外插入一个 换行 token。


## 2. LlavaNext Model

处理后的`inputs`会经过LlavaNext Model进行处理，得到输出结果。

```python
outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2.1 Vision Tower

Vision Tower是LlavaNext的视觉模型，它将图像编码为特征向量。

```python
vision_tower = model.vision_tower
```
