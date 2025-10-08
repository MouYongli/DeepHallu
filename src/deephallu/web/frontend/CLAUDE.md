VLM 可视化工具（Next.js） — PRD 草案
一、目标

构建一个基于 Next.js 的网页应用，用于可视化 视觉-语言模型（VLM） 的运行过程与结果，包括：

运行前后统计信息（token 数、推理时长等）

Attention Map（图像特征与文本 token 的注意力关系）

Token Prediction（下一 token 的预测分布、概率与熵）

目标是为研究者与开发者提供一个可交互的、可扩展的 VLM 可视化分析平台。

二、技术与数据
技术栈

框架：Next.js（App Router） + TypeScript

样式：Tailwind CSS

UI 库：shadcn/ui（Card、Tabs、Dropdown、Dialog、Tooltip 等）

状态与数据：React Query 或 SWR（用于拉取 mock 数据）

可视化：Recharts / Canvas（Attention Map 与统计图）

模型支持

当前默认模型：llava-hf/llava-v1.6-mistral-7b-hf

UI 设计需支持未来扩展为多模型。

数据来源
图片列表（Sidebar）

来自 MME 数据集（本地或接口）：

{
  "id": "mme_color_2",
  "name": "000000008277.jpg",
  "path": "/examples/mme/color/000000008277.jpg",
  "category": "Color",
  "dataset": "MME",
  "resolution": "Unknown",
  "format": "JPG",
  "size": "Unknown"
}


每张图片包含两条 questions 与 answers。

拖拽上传

用户可拖拽上传图片（支持多张）。

上传后默认：

question = "Describe what you see in this image."

answer = ""

运行后 Mock 返回数据

Token 列表示例：

[
  {
    "token_id": 12,
    "token": "hi",
    "image/text": "text",
    "input/output": "input",
    "position": 2
  }
]


说明：

Question / Answer / Generated Text 为原始文本；

Token list 来自模型 processor（模板 + tokenizer 处理后）。

三、信息架构与主导航
布局结构

左侧 Sidebar：
图片列表 + 拖拽上传区。
每项显示 name，hover 显示 dataset/category/format 等元信息。

顶部工具栏：

模型选择（下拉，当前仅 1 个但支持扩展）

“运行分析”按钮

右上角两个入口按钮：

Attention Map

Token Prediction

主内容区 Tabs：

Overview（概览）

Attention Map

Token Prediction

四、视图与交互设计
4.1 概览（Overview）
运行前状态

右侧面板：选择图片、模型。

Prompt 区域：

Question 下拉框（从图片自带的两条 question 选择）

可自定义输入 Prompt

Answer 区域：

显示原始 answer（可编辑）

Run Analysis 按钮：开始运行模型推理。

运行后状态

卡片式展示模型输出与统计数据：

Model Response / Generated Text（模型输出文本）

Token 统计卡片：

总 token 数

image tokens

text tokens

input vs output 数量

推理时长（ms/s）

Token 列表（Token Table）

可滚动查看

分组展示（image/text × input/output）

可筛选：

仅 image / 仅 text

仅 input / 仅 output

点击某个 token：

可在其他视图（Attention Map / Prediction）联动高亮。

右上角快捷入口：

进入 Attention Map 或 Token Prediction。

4.2 Attention Map 视图
视觉布局

图像区域

两张原图：

padded embedding（含 padding）

unpadded embedding（去除 padding）

支持 overlay 注意力热力图（Canvas）

文本 token 区域

Input Tokens / Output Tokens 分组

支持流式布局或网格布局

悬停交互

鼠标悬停在任一 output 文本 token 上：

在两张图上显示注意力热力图

在 token 列表中同步高亮

Mock 数据格式
{
  "image_features": {
    "padded": { "grid_h": 24, "grid_w": 24 },
    "unpadded": { "grid_h": 20, "grid_w": 20, "valid_mask": [ ... ] }
  },
  "text_tokens": [
    { "id": 1, "token": "A", "io": "input", "pos": 0 },
    { "id": 2, "token": "cat", "io": "output", "pos": 1 }
  ],
  "attn": [
    {
      "token_id": 2,
      "to_image_padded": [ ... ],
      "to_image_unpadded": [ ... ],
      "to_text": [ ... ]
    }
  ]
}

热力图逻辑

颜色映射：本地计算（0–1 线性）

模式：

阈值裁剪模式（clip by threshold）

最大值聚焦模式（focus on top-k）

4.3 Token Prediction 视图
布局与交互

Token 按 Input / Output 分区显示

悬停在：

任一 output token：显示它之前的“下一 token 分布”

Input 区最后一个 token：显示首个预测分布

弹出内容（气泡或侧边卡片）

Top-5 token（按概率降序）

概率（%）

预测分布熵（entropy）：

单位：nat 或 bit（UI 可切换）

真实标签对比（可选项）

Mock 数据格式
{
  "predictions": [
    {
      "context_pos": 15,
      "topk": [
        { "token": ",", "prob": 0.32 },
        { "token": "the", "prob": 0.21 },
        { "token": "a", "prob": 0.13 },
        { "token": ".", "prob": 0.09 },
        { "token": "of", "prob": 0.06 }
      ],
      "entropy": 1.82
    }
  ]
}

五、组件拆分
组件名	功能描述
Sidebar	图片列表 + 拖拽上传区
ImageMetaCard	展示图片元信息（类别、分辨率、格式等）
PromptPanel	Question 下拉 + 自定义输入 + Answer 编辑
ModelSelector	模型选择下拉菜单
RunStats	Token 数 / 推理时间 / 性能卡片
TokenTable	Token 分组过滤 + 点击联动
AttentionMapView	双图显示 + 文本 token + 注意力热力图
PredictionView	Token 列表 + 悬停弹出 top-5 + 熵
TopBarActions	Attention Map / Token Prediction 快捷入口







请在当前界面风格基础上，重构为一个 Next.js 的 VLM 可视化应用，包含左侧 Sidebar、顶部工具栏，以及三个视图（Overview / Attention Map / Token Prediction）。保持深色主题与卡片式信息层次。

【左侧 Sidebar】
- 显示 MME 图片列表（name 主显，hover 展示 dataset/category/format/resolution/size）。
- 顶部保留拖拽上传区。拖入新图后，为该图片自动创建：
  - question: "Describe what you see in this image."
  - answer: ""（为空）
- 列表项点击后在主区载入图片与其 questions/answers。

【顶部工具栏】
- 左侧：模型选择下拉（仅 1 项：llava-hf/llava-v1.6-mistral-7b-hf，但样式支持多选项）。
- 中间：Run Analysis 按钮（未运行时可点击，运行中显示加载）。
- 右侧：两个按钮
  - “Attention Map”
  - “Token Prediction”
  以 Tab 的形式切换视图；默认落在 “Overview”。

【视图一：Overview】
- 运行前：展示 Prompt 面板（Question 下拉可选两条内置 question，亦可自定义输入；Answer 框可编辑）、图片元信息卡片、模型选择。
- 运行后：显示
  - Generated Text（模型输出，纯文本）
  - 统计卡片：Total Tokens、Image Tokens、Text Tokens、Input Tokens、Output Tokens、Inference Time
  - Token 列表（来自处理后的 token 序列，注意与原始文本区分）：字段包含 token_id / token / image|text / input|output / position；支持筛选（image/text、input/output）与点击高亮联动。
- 右上角保持 “Attention Map”“Token Prediction” 按钮，便于跳转。

【视图二：Attention Map】
- 上方左右并排两张图：
  - 左：resize 后的 embedding（含 padding）
  - 右：固定边长 resize 且 unpad 的 embedding
- 下方为文本 token 区域，按 Input Tokens / Output Tokens 分组，流式排布。
- 悬停任一 Output token：
  - 在两张图上叠加对应注意力热力图（0–1 归一化，提供阈值滑块与聚焦最大值开关）。
  - 在文本 token 中高亮该 token 关注的其他 token（to_text 权重）。
- 支持切换查看对 padded/unpadded 的注意力。

【视图三：Token Prediction】
- 左侧 Input Tokens、右侧 Output Tokens（或上下布局，按屏宽自适应）。
- 悬停：
  - 在任一 Output token 上，展示“预测它之前一步的下一个 token 分布”。
  - 在 Input 最后一个 token 上，展示首个预测分布。
- 弹出气泡/侧边卡片显示：
  - Top-5 tokens（按概率降序）+ 概率
  - 预测分布的 Entropy（标注单位）
  - 可选显示该位置真实 token 以对比（若提供）。
- 与 Overview/Attention Map 保持选中 token 联动。

【数据/Mock 要点】
- Token 列表（示例）：
  [{ "token_id": 12, "token": "hi", "image/text": "text", "input/output": "input", "position": 2 }]
- Attention Map mock：
  - image_features：padded/unpadded 的 grid 尺寸与有效 mask
  - attn：对每个 output token，分别给 to_image_padded / to_image_unpadded / to_text 的归一化向量
- Prediction mock：
  - 对每个 context 位置给出 top5 的 {token, prob} 与 entropy