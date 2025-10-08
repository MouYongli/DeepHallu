import json
import os
os.environ["HF_HOME"] = "/DATA2/HuggingFace"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import os.path as osp
import argparse
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from deephallu.data.mme import MMEDataset

HERE = osp.dirname(osp.abspath(__file__))
RESULTS_DIR = osp.join(HERE, "..", "..", "..", "results")

def calculate_entropy(probs:torch.Tensor, log_base='e'):
    """计算概率分布的熵
    log_base: 对数底数，'e'表示自然对数(nats)，'2'表示以2为底(bits)
    """
    if log_base == 'e':
        return - (probs * torch.log(probs + 1e-10)).sum(dim=-1)
    elif log_base == '2':
        return - (probs * torch.log2(probs + 1e-10)).sum(dim=-1)
    else:
        raise ValueError(f"Invalid log base: {log_base}")
    

def analyze_scores_steps(
    scores: Tuple[torch.Tensor, ...],
    processor,
    top_k: int = 5
) -> List[List[Dict]]:
    """
    分析每个生成步骤的概率分布熵和top-k tokens
    Args:
        scores: 来自model.generate()的scores输出，tuple of tensors (step, (batch_size, vocab_size))
        processor: LlavaNextProcessor实例，用于解码token ids
        top_k: 保留top-k个最高概率的tokens
        
    Returns:
        List[List[Dict]]: 每个步骤的分析结果
        第一个维度是batch_size，第二个维度是每个batch的分析结果，包含:
            - step: 步骤索引
            - entropy: 该步骤的熵值
            - top_k_tokens: top-k tokens的列表
            - top_k_probs: top-k tokens对应的概率
            - top_k_token_ids: top-k tokens对应的token ids
    """
    batch_size = scores[0].shape[0]
    results = [[] for _ in range(batch_size)]
    for step_idx, logits in enumerate(scores):
        logits = logits.detach().cpu()
        # logits shape: (batch_size, vocab_size)
        for i in range(batch_size):
            logits_single = logits[i]  # shape: (vocab_size,)
            # 计算概率分布
            probs = F.softmax(logits_single, dim=-1)
            # 计算熵 H(p) = -Σ p(x) * log(p(x))
            # 使用loge计算，单位为nats，使用log2计算，单位为bits
            entropy = calculate_entropy(probs)
            # 获取top-k tokens
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            # 解码token ids到文本
            top_k_token_ids = top_k_indices.tolist()
            top_k_tokens = [
                processor.decode([token_id], skip_special_tokens=False) 
                for token_id in top_k_token_ids
            ]
            # 保存结果
            step_result = {
                'step': step_idx,
                'entropy': entropy.item(),
                'top_k_tokens': top_k_tokens,
                'top_k_probs': top_k_probs.tolist(),
                'top_k_token_ids': top_k_token_ids
            }
            results[i].append(step_result)
    return results        

def main(args):
    if args.model == "llava-next":
        processor = LlavaNextProcessor.from_pretrained(args.model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name, 
            attn_implementation="eager"
        ).to("cuda")
        # 确保模型配置启用 attention 输出
        model.config.output_attentions = True
        model.language_model.config.output_attentions = True
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    if args.dataset == "mme":
        dataset = MMEDataset()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if osp.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(dataset)} samples...")
    results = []
    step_details_flat = []  # 展开后的step信息
    for idx, data in enumerate(tqdm(dataset, desc="Processing")):
        try:
            image, id, image_name, category, question, answer = data
            print(f"i")
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
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            
            # 将inputs移动到模型设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1000, 
                    output_attentions=True, 
                    output_scores=True, 
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences
                scores = outputs.scores

            scores_results = analyze_scores_steps(scores, processor, args.top_k)
            generated_text = processor.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            avg_entropy = sum(result['entropy'] for result in scores_results[0]) / len(scores_results[0])
            results.append({
                "sample_id": id,
                "category": category,
                "question": question,
                "answer": answer,
                "generated_text": generated_text,
                "avg_entropy": avg_entropy
            })

            # 将每个step展开为一行
            for step_info in scores_results[0]:
                step_row = {
                    "sample_id": id,
                    "category": category,
                    "step": step_info['step'],
                    "entropy": step_info['entropy'],
                }
                # 添加top-k tokens和概率
                for k in range(args.top_k):
                    step_row[f'top{k+1}_token'] = step_info['top_k_tokens'][k]
                    step_row[f'top{k+1}_prob'] = step_info['top_k_probs'][k]
                    step_row[f'top{k+1}_token_id'] = step_info['top_k_token_ids'][k]
                
                step_details_flat.append(step_row)
            
            # 清理GPU内存
            del outputs, inputs
            torch.cuda.empty_cache()

            print(f"Processed data {idx} (id: {id if id else 'unknown'})")
        except Exception as e:
            print(f"\nError processing data {idx} (id: {id if id else 'unknown'}): {str(e)}")
            continue
    
    # 保存基本结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(osp.join(args.output_dir, "results.csv"), index=False)   
    # 保存详细的step信息
    step_details_df = pd.DataFrame(step_details_flat)
    step_details_df.to_csv(osp.join(args.output_dir, "step_details.csv"), index=False)
    print(f"\nProcessing completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-next", choices=["llava-next"])
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--dataset", type=str, default="mme", choices=["mme"])
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--save_all_scores", action="store_true",
                        help="Save all scores, otherwise save only top k scores")
    parser.add_argument("--top_k", type=int, default=5, help="Top k tokens to save")
    parser.add_argument("--save_all_attentions", action="store_true",
                        help="Save all attentions, otherwise only the attentions of generation steps")
    args = parser.parse_args()
    main(args)

    