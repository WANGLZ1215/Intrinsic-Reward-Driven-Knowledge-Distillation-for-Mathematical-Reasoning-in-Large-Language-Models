#!/usr/bin/env python3
"""
独立测试：Qwen2.5-32B-Instruct 在 GSM8K 数据集上的表现

完全独立版本 - 不依赖项目中的其他模块
可单独上传到VAST AI运行

功能：
- 直接加载模型和tokenizer
- 测试GSM8K数据集
- 提取答案并计算准确率
- 导出JSONL格式（与学生模型输出一致，包含top_ids/top_probs）
- 保存结果到JSONL文件，便于知识蒸馏对比

用法:
python test_qwen32b_gsm8k.py --eval_samples 200 --out teacher_gsm8k.jsonl
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import threading
import hashlib
from collections import OrderedDict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from tqdm import tqdm


# ✅ 修复1: 设置CUDA_LAUNCH_BLOCKING
if "CUDA_LAUNCH_BLOCKING" not in os.environ:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def setup_logging(level: str = "INFO") -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        logging.info("✅ CUDA_LAUNCH_BLOCKING=1 已启用")


def extract_answer_unified(text: str) -> Tuple[str, Optional[float]]:
    """
    统一的答案提取函数 - 支持多种格式
    这是项目中唯一的答案提取实现，其他模块应调用此函数
    
    支持格式：
    - #### (GSM8K标准格式) - 最高优先级
    - \\boxed{} (LaTeX格式)
    - "answer:" 或 "answer："
    - "The answer is"
    - 兜底：最后一个数字
    
    Args:
        text: 输入文本（包含推理过程和答案）
        
    Returns:
        (答案文本, 答案数字)
        - 答案文本: 提取的原始答案字符串
        - 答案数字: 转换为浮点数，如果无法提取则为None
    """
    if not text:
        return "", None
    
    # 清理文本中的特殊字符
    text_clean = re.sub(r'[^\w\s\.,!?;:()\[\]{}"\'-]', '', text)
    
    # 第一优先级：GSM8K标准的 #### 格式
    matches = re.findall(r"####\s*([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)", text_clean)
    if matches:
        # 使用最后一个匹配（避免示例中的干扰）
        last_match = matches[-1]
        # 检查这个匹配是否在示例之后（避免提取示例中的答案）
        match_pos = text_clean.rfind(f"#### {last_match}")
        example_pos = max(text_clean.rfind("Example"), text_clean.rfind("样例"), text_clean.rfind("example"))
        
        if example_pos == -1 or match_pos > example_pos:
            answer_text = last_match
        else:
            # 如果最后一个匹配在示例中，尝试倒数第二个
            if len(matches) > 1:
                answer_text = matches[-2]
            else:
                answer_text = None
        
        if answer_text:
            # 清理数字格式并转换
            num_clean = answer_text.replace('$', '').replace(',', '').strip()
            if num_clean.endswith('%'):
                num_clean = num_clean[:-1]
            try:
                return answer_text, float(num_clean)
            except ValueError:
                pass
    
    # 第二优先级：\boxed{} 格式
    match = re.search(r"\\boxed\{([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)\}", text_clean)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace('$', '').replace(',', '').strip()
        if num_clean.endswith('%'):
            num_clean = num_clean[:-1]
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 第三优先级："answer:" 或 "answer：" 格式
    match = re.search(r"answer[:：]?\s*([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)", text_clean, re.IGNORECASE)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace('$', '').replace(',', '').strip()
        if num_clean.endswith('%'):
            num_clean = num_clean[:-1]
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 第四优先级："The answer is" 格式
    match = re.search(r'The answer is\s*[\$]?([-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace(',', '').strip()
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 兜底方案：提取最后一个数字
    numbers = re.findall(r'[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        answer_text = numbers[-1]
        num_clean = answer_text.replace(',', '')
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    return "", None


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


class SimpleTeacherModel:
    """简化版教师模型包装器 - 独立运行版本"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                 cache_size: int = 1000, device_map: str = "auto",
                 torch_dtype: str = "bfloat16"):
        """初始化教师模型"""
        self.model_name = model_name
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self._tokenizer_lock = threading.Lock()
        
        # 转换torch_dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
        
        # 加载tokenizer
        logging.info(f"加载tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        logging.info(f"加载模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype_val,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 检查vocab_size
        tokenizer_vocab_size = len(self.tokenizer)
        model_emb_size = self.model.get_input_embeddings().weight.size(0)
        logging.info(f"📊 Vocab大小检查:")
        logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
        logging.info(f"   model embedding size: {model_emb_size}")
        
        if model_emb_size != tokenizer_vocab_size:
            logging.warning(f"⚠️  Vocab大小不匹配（{model_emb_size} vs {tokenizer_vocab_size}）")
            logging.info("   采用方式1：不resize，直接使用模型原始权重")
        else:
            logging.info("✅ Vocab大小已匹配")
        
        self.vocab_size = model_emb_size
        # 获取模型所在的device（对于device_map="auto"的情况）
        self.device = next(self.model.parameters()).device
        logging.info(f"✅ 模型加载完成，设备: {self.device}")
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """更新缓存"""
        self.cache[key] = value.clone().detach().cpu()
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
    
    def get_logits(self, text: str, use_cache: bool = True) -> torch.Tensor:
        """获取文本的logits"""
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # 计算logits
        with self._tokenizer_lock:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # 确保input_ids在有效范围内
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    if input_ids.numel() > 0:
                        if input_ids.max().item() >= self.vocab_size or input_ids.min().item() < 0:
                            logging.warning(f"⚠️ input_ids超出范围，自动clamp")
                            inputs['input_ids'] = torch.clamp(input_ids, 0, self.vocab_size - 1)
                
                # 将inputs移动到模型所在device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # 去掉batch维度
                
                if use_cache and cache_key is not None:
                    self._update_cache(cache_key, logits)
                
                return logits
    
    def generate_response(self, prompt: str, max_length: int = 512,
                         temperature: float = 0.7, do_sample: bool = True) -> str:
        """生成响应"""
        with self._tokenizer_lock:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # 确保input_ids在有效范围内
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    if input_ids.numel() > 0:
                        if input_ids.max().item() >= self.vocab_size or input_ids.min().item() < 0:
                            logging.warning(f"⚠️ input_ids超出范围，自动clamp")
                            inputs['input_ids'] = torch.clamp(input_ids, 0, self.vocab_size - 1)
                
                # 设置有效的pad_token_id和eos_token_id
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_token_id
                
                # 确保token ID在有效范围内
                pad_token_id = min(pad_token_id, self.vocab_size - 1)
                eos_token_id = min(eos_token_id, self.vocab_size - 1)
                
                # 计算max_new_tokens
                current_len = inputs['input_ids'].shape[1]
                max_allowed_new_tokens = min(max_length, 2048 - current_len - 10)
                
                if max_allowed_new_tokens <= 0:
                    logging.error(f"❌ 无法生成新token: 当前长度 {current_len} >= 最大长度 {2048}")
                    return ""
                
                # LogitsProcessor来mask超出范围的token
                class TokenRangeLogitsProcessor:
                    def __init__(self, max_valid_token_id: int):
                        self.max_valid_token_id = max_valid_token_id
                        self.vocab_end_idx = max_valid_token_id + 1
                    
                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                        if scores.shape[-1] > self.vocab_end_idx:
                            scores[..., self.vocab_end_idx:] = float('-inf')
                        return scores
                
                logits_processor = LogitsProcessorList([
                    TokenRangeLogitsProcessor(self.vocab_size - 1)
                ])
                
                # 将inputs移动到模型所在device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_allowed_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    repetition_penalty=1.1,
                    logits_processor=logits_processor,
                    use_cache=True
                )
                
                # 解码
                input_length = inputs['input_ids'].shape[1]
                if len(outputs[0]) > input_length:
                    generated_text = self.tokenizer.decode(
                        outputs[0][input_length:],
                        skip_special_tokens=True
                    )
                else:
                    generated_text = ""
                
                return generated_text


def main():
    parser = argparse.ArgumentParser(description="测试Qwen2.5-32B-Instruct在GSM8K上的表现")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                       help="模型名称")
    parser.add_argument("--eval_samples", type=int, default=200,
                       help="评估样本数")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "test"],
                       help="数据集分片")
    parser.add_argument("--max_length", type=int, default=512,
                       help="生成最大长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--topk_dist", type=int, default=50,
                       help="导出下一步分布的top-k大小（0表示不导出）")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="设备映射")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="数据类型")
    parser.add_argument("--out", type=str, default="teacher_gsm8k.jsonl",
                       help="输出JSONL文件路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="日志级别")
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    torch.manual_seed(42)
    
    # 创建输出目录
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    logging.info(f"加载数据集: GSM8K/{args.eval_split}")
    try:
        ds = load_dataset("gsm8k", "main")
        split = ds[args.eval_split]
        n = min(args.eval_samples, len(split))
        eval_ds = split.select(range(n))
        logging.info(f"✅ 数据集加载成功: {n} 个样本")
    except Exception as e:
        logging.error(f"❌ 数据集加载失败: {e}")
        return
    
    # 加载教师模型
    logging.info("加载教师模型...")
    try:
        teacher = SimpleTeacherModel(
            model_name=args.model_name,
            cache_size=1000,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype
        )
        logging.info("✅ 教师模型加载完成")
    except Exception as e:
        logging.error(f"❌ 教师模型加载失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 测试并导出
    logging.info("=" * 80)
    logging.info("开始测试")
    logging.info("=" * 80)
    
    num_ok = 0
    num_failed = 0
    num_correct = 0
    failed_indices = []
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(eval_ds, desc="测试进度", ncols=100)):
            q = sample["question"]
            gt = sample["answer"]
            prompt = f"Question: {q}\nAnswer: "
            
            # 生成
            resp = ""
            generation_error = None
            try:
                resp = teacher.generate_response(
                    prompt, 
                    max_length=args.max_length,
                    temperature=args.temperature, 
                    do_sample=True
                )
                if not isinstance(resp, str):
                    resp = str(resp) if resp else ""
            except Exception as e:
                generation_error = str(e)
                logging.warning(f"样本{idx+1} 生成失败: {e}")
                resp = ""
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    torch.cuda.empty_cache()
            
            # 提取答案（统一实现，优先 ####）
            gt_text, gt_num = extract_answer_unified(gt)
            pred_text, pred_num = extract_answer_unified(resp) if resp else ("", None)
            
            # 判断正确性
            is_correct = False
            if pred_text and gt_text:
                if pred_num is not None and gt_num is not None:
                    if abs(gt_num) < 1e-10:
                        is_correct = abs(pred_num - gt_num) < 1e-6
                    else:
                        relative_error = abs(pred_num - gt_num) / abs(gt_num)
                        is_correct = relative_error < 1e-6
                else:
                    is_correct = pred_text.strip().lower() == gt_text.strip().lower()
            
            if is_correct:
                num_correct += 1
            
            # 可选：导出下一步分布（在prompt上，避免响应长度差异影响体积）
            top_ids: List[int] = []
            top_probs: List[float] = []
            logits_error = None
            if args.topk_dist and args.topk_dist > 0 and not generation_error:
                try:
                    logits = teacher.get_logits(prompt)
                    if logits is not None and logits.ndim >= 2:
                        top_ids, top_probs = softmax_topk_from_logits(logits, args.topk_dist)
                except Exception as e:
                    logits_error = str(e)
                    logging.debug(f"样本{idx+1} 提取top-k失败: {e}")
            
            # 保存为JSONL格式（与学生模型输出一致）
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
            f.flush()
            os.fsync(f.fileno())
            
            if generation_error:
                num_failed += 1
                failed_indices.append(idx)
            else:
                num_ok += 1
    
    # 打印总结
    accuracy = num_correct / n if n > 0 else 0.0
    logging.info("=" * 80)
    logging.info("测试完成")
    logging.info("=" * 80)
    logging.info(f"总样本数: {n}")
    logging.info(f"成功生成: {num_ok}")
    logging.info(f"生成失败: {num_failed}")
    logging.info(f"正确答案: {num_correct}")
    logging.info(f"准确率: {accuracy:.4f} ({num_correct}/{n})")
    logging.info(f"结果已保存到: {out_path.absolute()}")
    if num_failed > 0:
        logging.warning(f"失败样本索引: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
