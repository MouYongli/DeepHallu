"""
LlavaNext Score and Attention Analysis
这个文件详细展示了LlavaNext的Output中Score和Attention的分析流程
"""

import os
os.environ["HF_HOME"] = "/DATA2/HuggingFace"
import os.path as osp
import argparse
import json
import requests
import pandas as pd
import re

HERE = osp.dirname(osp.abspath(__file__))
RESULTS_DIR = osp.join(HERE, "..", "..", "..", "results")

def llm_judge(question: str, answer: str, generated_text: str, model: str = "gpt-oss:120b"):
    """
    使用LLM判断generated_text是否符合answer
    Args:
        question: 问题
        answer: 答案
        generated_text: 生成文本
        model: 模型
    Returns:
        bool: 是否符合
    """
    url = "http://ollama.warhol.informatik.rwth-aachen.de/api/chat"
    judge_prompt = f"""Please evaluate if the generated answer is correct.

    Question: {question}
    Ground Truth: {answer}
    Generated Answer: {generated_text}

    Please output in the following format:
    - Judgment: <judgment>Correct/Incorrect/Unknown</judgment>
    - Reasoning: <reasoning>Your analysis</reasoning>
    - Suggestions: <suggestions>How to improve, optional</suggestions>
    """
    payload = {
        "model": model,
        "messages": [
            { "role": "user", "content": judge_prompt }
        ],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()["message"]["content"]
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("原始响应内容:")
        print(response.text)
        return None

def extract_judgment(response_text: str) -> int:
    """从LLM的回答中提取判断结果
    Args:
        response_text: LLM的回答文本
    Returns:
        int: 提取的判断结果 (1/0/-1)表示正确/错误/未知
    """
    judgment_pattern = r'<judgment>(.*?)</judgment>'
    match = re.search(judgment_pattern, response_text)
    if match:
        judgment_text = match.group(1).strip()
    else:
        judgment_text = response_text
    if 'Correct' in judgment_text:
        return 1
    elif 'Incorrect' in judgment_text:
        return 0
    else:
        return -1

def main(args):
    results_csv_path = osp.join(args.results_dir, "results.csv")
    results = pd.read_csv(results_csv_path)
    for idx, row in results.iterrows():
        print(f"Judging sample {idx+1}/{len(results)}")
        sample_id = row["sample_id"]
        category = row["category"]
        question = row["question"]
        answer = row["answer"]
        generated_text = row["generated_text"]
        avg_entropy = row["avg_entropy"]
        judgment = llm_judge(question, answer, generated_text)
        if judgment is None:
            judgment = -1
        else:
            judgment = extract_judgment(judgment)
        results.loc[idx, "answer_code"] = 1 if answer.lower() == "yes" else 0
        if (judgment == 1 and answer.lower() == "yes") or (judgment == 0 and answer.lower() == "no"):
            results.loc[idx, "generated_text_code"] = 1
        elif (judgment == 1 and answer.lower() == "no") or (judgment == 0 and answer.lower() == "yes"):
            results.loc[idx, "generated_text_code"] = 0
        else:
            results.loc[idx, "generated_text_code"] = -1
        results.loc[idx, "judgment"] = judgment
    results.to_csv(results_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--model", type=str, default="gpt-oss:120b")
    args = parser.parse_args()
    main(args)