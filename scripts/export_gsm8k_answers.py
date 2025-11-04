#!/usr/bin/env python3
"""
导出 GSM8K 答案（学生模型）

功能：
- 逐样本生成回答，使用项目统一的答案提取（extract_answer_unified，支持 ####）
- 可选导出下一步token分布的top-k（用于后续蒸馏对比近似KL/JS/余弦）
- 结果保存为 JSONL，便于后续合并对比

用法示例：
  学生模型（RL检查点）
    python scripts/export_gsm8k_answers.py \
      --student_model_path checkpoints/rl_model/checkpoint-1000 \
      --config config/training_config.yaml \
      --eval_samples 200 \
      --out results/student_gsm8k.jsonl

注意：教师模型请使用 scripts/export_teacher_gsm8k_answers.py
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

# 加入项目根目录
sys.path.append(str(Path(__file__).parent.parent))

from models.student_model import StudentModel
from utils.math_utils import extract_answer_unified


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def softmax_topk_from_logits(logits: torch.Tensor, top_k: int = 50) -> Tuple[List[int], List[float]]:
    """从最后一步logits计算top-k概率分布（返回ids与probs列表）。
    仅取最后一步（prompt最后token）以降低体积，供近似蒸馏对比使用。
    """
    if logits is None:
        return [], []
    last_logits = logits[-1]  # (vocab_size,)
    probs = torch.softmax(last_logits.float(), dim=-1)
    topk = min(top_k, probs.shape[-1])
    values, indices = torch.topk(probs, k=topk, dim=-1)
    return indices.tolist(), values.tolist()


def main():
    parser = argparse.ArgumentParser(description="导出GSM8K答案（学生模型）")
    parser.add_argument("--student_model_path", type=str, required=True,
                        help="学生模型（或RL检查点）路径")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "test"],
                        help="GSM8K分片")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="评估样本数（默认200）")
    parser.add_argument("--max_length", type=int, default=512,
                        help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度")
    parser.add_argument("--topk_dist", type=int, default=50,
                        help="导出下一步分布的top-k大小（0表示不导出）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--out", type=str, required=True,
                        help="输出JSONL文件路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别")
    args = parser.parse_args()

    setup_logging(args.log_level)
    torch.manual_seed(args.seed)

    config = load_config(args.config)

    # 加载数据
    ds = load_dataset("gsm8k", "main")
    split = ds[args.eval_split]
    n = min(args.eval_samples, len(split))
    eval_ds = split.select(range(n))
    logging.info(f"数据集: GSM8K/{args.eval_split}, 样本数={n}")

    # 加载学生模型
    logging.info("加载学生模型...")
    student = StudentModel(
        model_name=args.student_model_path,
        lora_config=config["lora"],
        device=config["device"]["device_map"],
        torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
        use_lora=True
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_ok = 0
    num_failed = 0
    failed_indices = []

    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(eval_ds, desc="导出进度", ncols=100)):
            q = sample["question"]
            gt = sample["answer"]
            prompt = f"Question: {q}\nAnswer: "

            # 生成
            resp = ""
            generation_error = None
            try:
                resp = student.generate(prompt, max_length=args.max_length,
                                        temperature=args.temperature, do_sample=True)
                if not isinstance(resp, str):
                    resp = str(resp) if resp else ""
            except Exception as e:
                generation_error = str(e)
                logging.warning(f"样本{idx+1} 生成失败: {e}")
                resp = ""
                # 如果是CUDA错误，清理缓存
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    torch.cuda.empty_cache()

            # 提取答案（统一实现，优先 ####）
            gt_text, gt_num = extract_answer_unified(gt)
            pred_text, pred_num = extract_answer_unified(resp) if resp else ("", None)

            # 可选：导出下一步分布（在prompt上，避免响应长度差异影响体积）
            top_ids: List[int] = []
            top_probs: List[float] = []
            logits_error = None
            if args.topk_dist and args.topk_dist > 0 and not generation_error:
                try:
                    logits = student.get_logits(prompt)  # (seq_len, vocab)
                    if logits is not None and logits.ndim >= 2:
                        top_ids, top_probs = softmax_topk_from_logits(logits, args.topk_dist)
                except Exception as e:
                    logits_error = str(e)
                    logging.debug(f"样本{idx+1} 提取top-k失败: {e}")

            record = {
                "index": idx,
                "question": q,
                "prompt": prompt,
                "ground_truth": gt,
                "ground_truth_text": gt_text if gt_text else "N/A",
                "ground_truth_num": gt_num if gt_num is not None else "N/A",
                "response": resp if resp else "",
                "answer_text": pred_text if pred_text else "N/A",
                "answer_num": pred_num if pred_num is not None else "N/A",
                "top_ids": top_ids,
                "top_probs": top_probs,
                "error": generation_error if generation_error else (logits_error if logits_error else None)
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 实时落盘，避免长时间无输出的观感
            f.flush()
            os.fsync(f.fileno())
            
            if generation_error:
                num_failed += 1
                failed_indices.append(idx)
            else:
                num_ok += 1

    logging.info(f"导出完成: {out_path}")
    logging.info(f"  成功: {num_ok} 条")
    if num_failed > 0:
        logging.warning(f"  失败: {num_failed} 条")
        logging.warning(f"  失败样本索引: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")


if __name__ == "__main__":
    main()


