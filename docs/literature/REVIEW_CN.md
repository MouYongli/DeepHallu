# 文献总结

## 目录

- [1. MoLE: Decoding by Mixture of Layer Experts Alleviates Hallucination in Large Vision-Language Models](#1-mole-decoding-by-mixture-of-layer-experts-alleviates-hallucination-in-large-vision-language-models)
- [2. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](#2-mitigating-object-hallucinations-in-large-vision-language-models-through-visual-contrastive-decoding)




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

## 2. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding

- 会议：CVPR 2024

- 论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.pdf

- 作者：Sicong Leng (Alibaba Group & NTU), Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing	

- 主题和核心观点
    - 本文提出了一种名为Visual Contrastive Decoding (VCD)的训练free的方法，有效缓解大型视觉语言模型（LVLM）中的对象幻觉问题。VCD通过对比原始与失真视觉输入产生的输出概率分布，减少模型对统计偏差和语言先验的过度依赖，从而显著提升生成文本与视觉内容的一致性。实验验证表明，VCD在多个LVLM模型和基准数据集上均表现出明显优势，并增强了模型的视觉感知能力。

- 研究背景与问题描述
    - 研究背景
        - 大型视觉语言模型（LVLMs）在融合视觉理解与自然语言生成方面取得重大进展，支持诸多实际应用场景。
    - 问题描述
        - 这些模型存在对象幻觉（object hallucination）问题，即生成的文本中包含图像中不存在的对象，造成内容与视觉输入不符。该问题主要源于训练数据的统计偏差及语言模型的先验偏好，影响模型的可靠性，特别是在医疗、自动驾驶等关键领域。
    - 挑战
        - 现有方法多依赖额外数据、复杂微调或外部模型，成本高且难以推广，急需简单高效的解决方案。

- 创新点或新方法
    - 视觉不确定性分析：首次系统分析视觉输入中不确定性（通过加噪声模拟失真）如何放大统计偏差和语言先验，导致幻觉加剧。
    - Visual Contrastive Decoding (VCD)：提出一种无需额外训练的解码策略，通过对比原始与失真视觉输入的输出分布，校正模型过度依赖语言先验和统计偏差的倾向。

    - 核心公式: 
        $$
        p_{vcd}(y \mid v, v', x) = \mathrm{SoftMax}\big[(1 + \alpha) \cdot \text{logit}_{\theta}(y \mid v, x) - \alpha \cdot \text{logit}_{\theta}(y \mid v', x)\big]
        $$
        其中，$v$ 为原始视觉输入，$v'$ 为失真视觉输入，$x$ 为文本提示，$\alpha$ 控制对比强度。该方法在保证语言合理性的基础上，剔除与失真输入相关的幻觉倾向。

- 方法详解
    - 视觉不确定性分析：
        - 在大型视觉语言模型（LVLMs）中，视觉输入 $v$ 对生成文本的准确性至关重要。视觉不确定性指的是视觉输入信号的不清晰或模糊，比如图像质量下降、遮挡、噪声等情况。当视觉输入变得不确定时，模型对视觉特征的编码能力下降，容易依赖语言模型中的语言先验（即对词语出现概率的先验假设）和训练数据中的统计偏差，从而增加“对象幻觉”（hallucination）的发生率。
        - 论文采用了高斯噪声添加（Gaussian noise masking）这一简单而有效的方式来模拟视觉不确定性。通过在原始视觉输入图像上逐步加入高斯噪声，形成一系列从清晰到高度失真的图像序列。数学建模使用了**扩散过程（Diffusion Process）**的形式：
            $$
            q(v_t \mid v_{t-1}) = \mathcal{N}(v_t; \sqrt{1 - \gamma} \cdot v_{t-1}, \gamma \cdot I)
            $$
            其中，$v_t$ 是第 $t$ 步的失真图像，$v_{t-1}$ 是第 $t-1$ 步的图像，$\gamma$ 是高斯噪声方差，$I$ 是单位矩阵。

- 关键实验、数据与案例
    - 数据集和指标
        - POPE: 面向对象存在性的问答评估，衡量幻觉准确性（Accuracy, Precision, Recall, F1）。
        - MME: 多维度多模态模型评估，涵盖对象与属性级幻觉。
        - LLaVA-Bench: 多样图像与问题的开放式生成评测。

    - 实验结果
        - VCD在POPE各设置（随机、流行、对抗）中均明显优于常规解码，最高提升7.4点F1。
        - MME对象与属性幻觉显著减少，总分提升超过30分（如LLaVA-1.5）。
        - LLaVA-Bench中案例显示VCD有效剔除“surfboard”等幻觉物体，输出更加符合视觉内容。
        - GPT-4V辅助的开放式生成评价显示VCD提升生成文本的准确度与细节丰富度。

    - 结论验证
        - 视觉不确定性加剧幻觉问题，VCD有效缓解，且对模型的整体视觉理解能力有积极影响。

- 总结与展望
    - 本文深入揭示视觉不确定性对LVLM幻觉的影响，创新提出训练免费且高效的Visual Contrastive Decoding策略，实现了显著的幻觉缓解和视觉感知提升。   
    - 局限与未来方向
        - 目前采用的高斯噪声作为视觉扰动较为基础，未来可探索更细粒度如对象模糊的失真方法。
        - 研究仅限于图像与文本LVLM，未来计划拓展至视频理解领域。
        - VCD框架具备拓展性，可适配更多LVLM变体和更复杂采样策略。
