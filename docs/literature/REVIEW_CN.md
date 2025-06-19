# 文献总结

## 目录

- [1. MoLE: Decoding by Mixture of Layer Experts Alleviates Hallucination in Large Vision-Language Models](#1-mole-decoding-by-mixture-of-layer-experts-alleviates-hallucination-in-large-vision-language-models)
- [2. Alleviating Hallucinations in Large Vision-Language Models through Hallucination-Induced Optimization](#2-alleviating-hallucinations-in-large-vision-language-models-through-hallucination-induced-optimization)
- [3. Analyzing and mitigating object hallucination in large vision-language models](#3-analyzing-and-mitigating-object-hallucination-in-large-vision-language-models)
- [4. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](#4-mitigating-object-hallucinations-in-large-vision-language-models-through-visual-contrastive-decoding)
- [5. Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention](#5-mitigating-hallucination-in-large-vision-language-models-via-modular-attribution-and-intervention)




## 1. MoLE: Decoding by Mixture of Layer Experts Alleviates Hallucination in Large Vision-Language Models

- 会议：AAAI-25
- 论文链接：https://ojs.aaai.org/index.php/AAAI/article/view/34056
- 作者：Tian Liang (Zhejiang University), Yuetian Du (Zhejiang University), Jing Huang (Zhejiang University), Ming Kong (Zhejiang University), Luyuan Chen (Beijing Information Science and Technology University), Yadong Li (Ant Group), Siye Chen (Ant Group), Qiang Zhu (Zhejiang University, corresponding author)

- 主题和核心观点
    - 本文提出了一种名为 Mixture of Layer Experts (MoLE) 的无训练解码方法，通过利用大型视觉语言模型（LVLM）中不同层次的专家层协同工作，有效缓解了模型在多模态生成任务中出现的幻觉现象（hallucination），显著提升生成内容的准确性和一致性

- 研究背景与问题描述
    - 大型视觉语言模型（LVLM）在多模态任务（如图像描述、视觉问答）中表现优异，但其生成内容经常出现幻觉，即生成与输入视觉内容或指令不符的内容，严重影响模型的可靠性和实用性。
    - 现有的缓解幻觉方法多基于对比解码（contrastive decoding），通过构造“业余模型”（amateur models）与专家模型对比来过滤错误输出，存在计算开销大且效果受限的问题。
    - 幻觉问题主要源于模型解码过程中的推理和事实信息注入环节，且随着生成序列长度增加，模型对原始提示的遗忘进一步加剧了幻觉。

- 创新点或新方法
    - MoLE 方法：借鉴 Mixture of Experts (MoE) 的思想，提出在单个LVLM模型内部不同层之间进行专家层混合解码。
    - 专家层划分：设计三类专家层——
        - Final Expert（最终专家）：模型最深层，负责综合所有信息生成最终输出。
        - Second Opinion Expert（第二意见专家）：从倒数几层动态选择，基于Jensen-Shannon散度（JSD）选择对关键生成令牌（top-k tokens）意见与最终专家最大差异、对多数令牌一致的层，为关键内容提供备选意见。
        - Prompt Retention Expert（提示保留专家）：选择对输入提示关注度最高的层，随着生成序列长度增长，其输出权重逐渐增加，帮助模型保持对原始提示的忠实。
    - 融合策略：将三类专家的logits直接相加后做softmax，替代传统基于业余模型对比的减法策略，保证协同增强生成准确性且计算开销低。

- 关键公式
    1. 最终专家概率分布

        给定一个prompt $P_T = [p_1, p_2, \cdots, p_T]$ (包括图像特征和文本指令token)和前 $t-1$ 个已经生成的token $x_{<t} = [x_1, x_2, \cdots, x_{t-1}]$。
        将这些输入通过嵌入层得到 
        $$
        H^{(0)}=[h^{(0)}_{p_1}, h^{(0)}_{p_2}, \cdots, h^{(0)}_{p_T}, h^{(0)}_{x_1}, h^{(0)}_{x_2}, \cdots, h^{(0)}_{x_{t-1}}]
        $$
        然后通过 $N$ 层Transformer层得到 
        $$
        H^{(N)}=[h^{(N)}_{p_1}, h^{(N)}_{p_2}, \cdots, h^{(N)}_{p_T}, h^{(N)}_{x_1}, h^{(N)}_{x_2}, \cdots, h^{(N)}_{x_{t-1}}]
        $$
        最后通过分类头得到最终专家概率分布为：
        $$
        p(x_t \mid P_T; x_{<t}) = \mathrm{SoftMax}\big(\phi_{\mathrm{X}}(h^{(N)}_{t-1})\big)
        $$
        其中，$ h^{(N)}_{t-1} $ 是第 $N$ 层对第 $t-1$ 个token, 即$x_{t-1}$， 的输出；$\phi_{\mathrm{X}}$ 是分类头； $\mathrm{X}$ 是输出空间，即词汇表。

    2. 第二意见专家层选取依据（Jensen-Shannon散度）

        先通过计算第 $N$ 层（最终专家）和第 $j$ 层的logits分布的Jensen-Shannon散度：
        $$
        d(q_N, q_j) = \mathrm{JSD}\big(q_N \parallel q_j\big)
        $$
        其中，$q_N$ 是第 $N$ 层（最终专家）的logits分布，$q_j$ 是第 $j$ 层的logits分布。

        **__保证SOE层对关键token与大多数token的分歧与一致性__**：
        关键token的Top-3选取基于Final Expert层的概率排名，选概率最大的3个token。计算各候选层在这3个token上的概率分布差异（JSD）。JSD越大，代表候选层与Final Expert在关键token上的观点越分歧。Second Opinion Expert会选出对关键token分歧最大且对大多数token一致的层。
        $$
        M_{j}^{topk} = \arg\max_{j} d(q_N^{topk}, q_j^{topk})
        $$
        对多数token，寻找与Final Expert一致性最高的层：
        $$
        M_{j}^{majority} = \arg\min_{j} d(q_N^{majority}, q_j^{majority})
        $$
    
        $$
        M_{j}^{SOE} = \begin{cases}
            M_{j}^{topk}, & \text{if } M_{j}^{topk} = M_{j}^{majority} \\
            -1, & \text{if } M_{j}^{topk} \neq M_{j}^{majority}
        \end{cases}
        $$

        

    3. 提示保留专家权重随时间变化
        $$
        q_{PR} = \left(1 - e^{-\frac{t}{\lambda}}\right) \cdot q_{PR_t}
        $$
        随生成时间 $t$ 增加，提示保留专家层的权重逐渐增强，$\lambda$ 控制增长速率。

    4. 最终解码概率融合
        $$
        p_{\mathrm{MoLE}} = \mathrm{SoftMax}\big(q_F + q_{SO} + q_{PR}\big)
        $$
        三个专家层的logits相加后进行归一化，得到最终预测概率。

- 方法详解
    - 模型结构：典型LVLM包括嵌入层、N个Transformer层、以及分类头。生成时，每个时间步将视觉特征和文本指令与已生成令牌编码后，经过所有层输出下一个令牌概率。
    - Final Expert：使用最深层的输出预测，是生成的主要依据。
    - Second Opinion Expert：通过计算各层与最终层logits分布的JSD，选择一个在关键令牌上有较大分歧但多数令牌分布接近最终层的层作为第二意见专家，提供多样化视角辅助决策。
    - Prompt Retention Expert：通过计算每层对输入提示部分的注意力分数和，选出关注提示最高的层。其输出在生成初期权重较小，随着序列增长权重逐步加大，防止模型遗忘原始输入。
    - Gating机制：
        - 第二意见专家通过top-k令牌JSD最大化及多数令牌JSD最小化的筛选机制动态确定。
        - 提示保留专家输出加权随时间指数增长。
    - 解码融合：三专家层logits相加后归一化，得到最终预测概率，无需额外训练，且只需一次前向传播，效率高。

- 实验
    - 基线方法比较：与Beam Search、Greedy Search、OPERA、VCD、DoLA等多种先进无训练解码方法对比。
    - 模型及数据：在MiniGPT-4、LLaVA-1.5和Shikra三种主流LVLM模型上测试，数据使用COCO图像和多模态幻觉检测基准数据集。
    - 评价指标：
        - POPE (Polling-based Object Probing Evaluation) ：通过问答方式检测是否错误识别图像中不存在物体。
        - CHAIR (Caption Hallucination Assessment with Image Relevance)：评估图像描述中提及不存在物体的频率，含句子级和图像级指标。
    - 主要结果：
        - MoLE在POPE各采样模式（随机、流行、对抗性）中均显著超越所有基线，提升准确率和精度，MiniGPT-4模型准确率提升约8.7%。
        - 在CHAIR指标上，MoLE降低幻觉率明显，如MiniGPT-4模型CHAIRI下降约21%。
        - 消融实验显示，三种专家层均有效减少幻觉，且引入的门控机制进一步提升性能。
        - 动态选层机制优于随机或静态选层，Prompt Retention专家权重随序列增长的设计合理有效。

## 2. Alleviating Hallucinations in Large Vision-Language Models through Hallucination-Induced Optimization

- 会议：NeurIPS 2024

- 作者：
    - Xinyu Lyu*	1. Southwestern University of Finance and Economics, Chengdu, China	xinyulyu68@gmail.com
    - Beitao Chen*	2. Shenzhen Institute for Advanced Study, UESTC	chenbeitao@gmail.com
    - Lianli Gao†	2. Shenzhen Institute for Advanced Study, UESTC	lianli.gao@uestc.edu.cn
    - Jingkuan Song	2. Shenzhen Institute for Advanced Study, UESTC	jingkuan.song@gmail.com
    - Heng Tao Shen	3. Center for Future Media, UESTC 4. Tongji University	shenhengtao@hotmail.com

- 主题和核心观点
    - 本文提出了一种名为“Hallucination-Induced Optimization (HIO)”的新优化策略，通过理论化的偏好模型增强幻觉与目标Token的对比，从而有效缓解大规模视觉语言模型（LVLMs）的幻觉问题，并在多项基准上超越了现有SOTA方法。

- 研究背景与问题描述
    - 大规模视觉语言模型（LVLMs）在多模态任务中表现出色，但在生成过程中经常出现幻觉，即生成与输入视觉内容或指令不符的内容，严重影响模型的可靠性和实用性。
    - 现有的缓解幻觉方法多基于对比解码（contrastive decoding），通过构造“业余模型”（amateur models）与专家模型对比来过滤错误输出，存在计算开销大且效果受限的问题。
    - 幻觉问题主要源于模型解码过程中的推理和事实信息注入环节，且随着生成序列长度增加，模型对原始提示的遗忘进一步加剧了幻觉。

- 创新点或新方法
    - HIO 方法：通过理论化的偏好模型增强幻觉与目标Token的对比，从而有效缓解大规模视觉语言模型（LVLMs）的幻觉问题，并在多项基准上超越了现有SOTA方法。

- 关键公式
    1. 偏好模型
    $$
    p(x_t \mid P_T; x_{<t}) = \mathrm{SoftMax}\big(\phi(h^{(N)}_{t-1})\big)
    $$
    其中，$ h^{(N)}_{t-1} $ 是第 $N$ 层的输出，$\phi$ 是分类头。

    2. 偏好模型



## 3. Analyzing and mitigating object hallucination in large vision-language models

- 会议：ICLR 2024

- 作者：
    - 

- 主题和核心观点
    - 本文针对大型视觉语言模型（LVLMs）在生成图像描述时出现的“对象幻觉”（即描述中包含图像中实际不存在的对象）问题，提出了一种轻量级、通用的后处理修正方法——LVLM Hallucination Revisor (LURE)，基于对幻觉产生的关键因素（共现性、不确定性和对象位置）的统计分析，显著降低了LVLM生成描述中的对象幻觉。

- 研究背景与问题描述


## 4. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding

## 5. Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention

- 会议：NeurIPS 2024 Workshop

- 作者：
    - 

- 主题和核心观点

- 研究背景与问题描述

- 创新点或新方法