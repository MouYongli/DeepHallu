# 视觉语言模型中的幻觉问题：数学建模框架

## 1. 背景简介

视觉语言模型旨在处理图像和语言的多模态输入与输出任务，例如图文生成（image captioning）、图文问答（VQA）等。然而，这些模型常常会“幻觉”出图像中不存在的内容。我们希望从概率建模角度形式化该问题。

## 2. 基本定义与建模符号

我们设定如下符号：

- $ \mathcal{I} \in \mathbb{R}^{H \times W \times C} $：图像输入
- $ \mathcal{Q} $：文本输入（如问题）  
- $ \mathcal{A} $：模型输出的答案或生成文本  
- $ \mathcal{M} $：VLM 模型，可表示为条件概率模型  $ \mathcal{M}(\mathcal{A} \mid \mathcal{I}, \mathcal{Q}) $
    - 通常，$ \mathcal{M} $ 可以表示为 $ \mathcal{M}(\mathcal{A} \mid \mathcal{I}, \mathcal{Q}) = p(\mathcal{A} \mid \mathcal{I}, \mathcal{Q}) $，其中 $ p(\mathcal{y}_{t} \mid \mathcal{I}, \mathcal{Q}) $ 是模型输出的概率分布。
- $ \mathcal{E} $：图像中真实存在的实体集合（可由标注或检测器获得）  
- $ \hat{\mathcal{E}}(\mathcal{A}) $：模型输出文本中提及的实体集合（可通过实体识别或规则抽取得到）

