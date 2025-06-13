# VLM Attention Modification Framework

本框架提供了对大视觉语言模型注意力机制进行修改和分析的工具，特别针对你提到的研究方向：

## 🎯 核心功能

### 1. 注意力机制修改
- **Vision Tokens**: 稀疏化 + 对称（双向）注意力
- **Text Tokens**: 因果（单向）注意力 
- **Cross-modal**: 可配置的跨模态注意力模式
- **Embedding Fusion**: 多种融合策略

### 2. 幻觉检测与评估
- CHAIR Score (图像描述幻觉)
- POPE (对象探测幻觉)
- 自定义幻觉率指标

## 📁 文件结构

```
src/deephallu/models/
├── attention_hooks.py      # 注意力修改核心功能
├── base.py                # VLM基础类
└── __init__.py

src/deephallu/evaluation/
├── metrics.py             # 幻觉评估指标
└── __init__.py

scripts/
└── attention_modification_experiment.py  # 完整实验脚本

examples/
└── attention_hooks_demo.py              # 简单演示脚本
```

## 🚀 快速开始

### 1. 运行简单演示

```bash
cd Projects/DeepHallu
python examples/attention_hooks_demo.py
```

这个脚本会演示：
- 注意力修改的基本过程
- 不同embedding融合策略
- 稀疏注意力的效果分析

### 2. 运行完整实验

```bash
# 使用mock模型进行演示
python scripts/attention_modification_experiment.py --model mock

# 使用真实模型（需要下载）
python scripts/attention_modification_experiment.py --model deepseek-vl2 --batch-size 2

# 自定义参数
python scripts/attention_modification_experiment.py \
    --model mock \
    --batch-size 4 \
    --vision-seq-len 49 \
    --text-seq-len 128 \
    --vision-sparsity 0.1 \
    --output-dir ./my_results
```

### 3. 参数说明

- `--model`: 模型选择 (`mock`, `deepseek-vl2`, `qwen2.5-vl`)
- `--batch-size`: 批次大小
- `--vision-seq-len`: 视觉token序列长度
- `--text-seq-len`: 文本token序列长度  
- `--vision-sparsity`: 视觉注意力稀疏度比例
- `--output-dir`: 结果输出目录

## 🔧 核心API使用

### 1. 注意力修改器

```python
from deephallu.models.attention_hooks import AttentionModifier

# 创建注意力修改器
modifier = AttentionModifier(
    vision_sparsity_ratio=0.1,        # 保留10%的vision attention weights
    vision_attention_type="symmetric", # 视觉tokens使用对称注意力
    text_attention_type="causal",     # 文本tokens使用因果注意力
    cross_attention_type="bidirectional"  # 跨模态双向注意力
)

# 注册hooks到模型
modifier.register_hooks(
    model=your_vlm_model,
    vision_seq_len=49,
    text_seq_len=128,
    target_layers=['attention_layer_11']  # 指定要修改的层
)

# 运行模型（注意力会被自动修改）
output = your_vlm_model(inputs)

# 分析注意力变化
analysis = modifier.get_attention_analysis()

# 清理hooks
modifier.remove_hooks()
```

### 2. Embedding融合

```python
from deephallu.models.attention_hooks import EmbeddingFuser

# 创建融合器
fuser = EmbeddingFuser(
    fusion_method="attention",  # 可选: "concat", "add", "gated", "attention"
    hidden_dim=768
)

# 融合embeddings
fused_embeddings = fuser.fuse_embeddings(
    vision_embeddings,  # [batch, vision_seq_len, hidden_dim]
    text_embeddings     # [batch, text_seq_len, hidden_dim]
)
```

### 3. 幻觉评估

```python
from deephallu.evaluation.metrics import CHAIRScore, POPEEvaluator

# CHAIR评估（图像描述）
chair = CHAIRScore()
chair_results = chair.compute(
    predictions=generated_captions,
    references=reference_captions
)

# POPE评估（对象存在性）
pope = POPEEvaluator()
pope_results = pope.compute(
    predictions=model_yes_no_answers,
    references=ground_truth_labels
)
```

## 📊 实验结果分析

实验完成后，你会获得：

### 1. 注意力分析
```python
{
    'layer_name': {
        'attention_entropy': 5.2341,      # 原始注意力熵
        'modified_entropy': 4.8923,      # 修改后注意力熵
        'attention_sparsity': 0.1234,    # 注意力稀疏度
        'attention_change': 0.5678       # 注意力变化程度
    }
}
```

### 2. 幻觉评估结果
```python
{
    'chair': {
        'CHAIR_S': 0.25,      # 句子级幻觉率
        'CHAIR_I': 0.15,      # 实例级幻觉率
        'CHAIR_avg': 0.20     # 平均CHAIR分数
    },
    'pope': {
        'accuracy': 0.85,           # 准确率
        'hallucination_rate': 0.12, # 幻觉率
        'f1_score': 0.83           # F1分数
    }
}
```

### 3. 可视化输出
- `attention_comparison.png`: 注意力模式对比图
- `experiment_results.json`: 详细实验数据

## 🔬 研究应用

### 1. 注意力机制研究
- 分析vision和text tokens的不同注意力模式
- 研究稀疏性对模型性能的影响
- 探索cross-modal attention的最优配置

### 2. 幻觉缓解研究  
- 比较不同注意力策略对幻觉的影响
- 评估attention modification的有效性
- 开发新的幻觉检测方法

### 3. 模型可解释性
- 可视化注意力权重变化
- 分析模型关注的图像区域
- 理解vision-text交互机制

## 🛠️ 自定义扩展

### 1. 添加新的注意力模式
```python
class CustomAttentionModifier(AttentionModifier):
    def apply_custom_attention(self, attention_weights):
        # 实现你的自定义注意力逻辑
        return modified_attention
```

### 2. 添加新的评估指标
```python
class CustomHallucinationMetric(BaseMetric):
    def compute(self, predictions, references, **kwargs):
        # 实现你的评估逻辑
        return {'custom_score': score}
```

### 3. 集成真实VLM模型
```python
# 在setup_model()中添加你的模型
elif self.model_name == "your_model":
    from your_model_package import YourVLM
    self.model = YourVLM.from_pretrained("model_path")
```

## ⚠️ 注意事项

1. **内存使用**: 注意力权重存储会占用额外内存
2. **Hook清理**: 实验完成后记得调用`remove_hooks()`
3. **模型兼容性**: 确保目标模型的attention层命名符合预期
4. **GPU需求**: 建议使用GPU加速大型模型实验

## 📝 实验建议

1. **逐步测试**: 先在小模型上验证方法，再扩展到大模型
2. **参数调优**: 尝试不同的sparsity ratio和attention类型
3. **对比实验**: 设置多组对照实验比较效果
4. **结果记录**: 详细记录实验参数和结果，便于复现

## 🔗 相关资源

- [DeepSeek-VL2 论文](https://arxiv.org/abs/2310.04269)
- [Qwen2.5-VL 文档](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [CHAIR 评估方法](https://arxiv.org/abs/1711.07131)
- [POPE 评估基准](https://arxiv.org/abs/2305.14552)

这个框架为你的VLM幻觉研究提供了完整的工具链，支持从注意力机制修改到幻觉评估的整个研究流程。🚀 