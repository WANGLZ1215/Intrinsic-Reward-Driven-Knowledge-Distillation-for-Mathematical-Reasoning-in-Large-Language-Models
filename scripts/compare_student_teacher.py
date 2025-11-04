#!/usr/bin/env python3
"""
对比学生/教师在 GSM8K 的结果，计算：
- 各自准确率（基于 extract_answer_unified 的数值/文本对比）
- 近似蒸馏指标（基于prompt处的top-k分布）：KL、反向KL、JS、余弦相似度

输入：两份由 export_gsm8k_answers.py 生成的 JSONL（student / teacher）
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import math


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(levelname)s - %(message)s')


def load_jsonl(path: str) -> Dict[int, Dict]:
    data: Dict[int, Dict] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[int(obj["index"])]=obj
    return data


def accuracy(data: Dict[int, Dict]) -> float:
    total = len(data)
    correct = 0
    for v in data.values():
        gt_num = v.get("ground_truth_num")
        ans_num = v.get("answer_num")
        if isinstance(gt_num, (int,float)) and isinstance(ans_num,(int,float)):
            # 数值比较（容差）
            tol = 1e-6
            if abs(float(ans_num) - float(gt_num)) < tol:
                correct += 1
        else:
            # 文本回退
            gt_text = str(v.get("ground_truth_text",""))
            ans_text = str(v.get("answer_text",""))
            if gt_text and ans_text and gt_text.strip().lower()==ans_text.strip().lower():
                correct += 1
    return correct/total if total>0 else 0.0


def sparse_vec(top_ids: List[int], top_probs: List[float]) -> Dict[int, float]:
    return {int(i): float(p) for i,p in zip(top_ids, top_probs)}


def kl_divergence(p: Dict[int,float], q: Dict[int,float], eps: float=1e-12) -> float:
    # KL(P||Q)
    keys = set(p.keys()) | set(q.keys())
    s = 0.0
    for k in keys:
        pk = max(eps, p.get(k, 0.0))
        qk = max(eps, q.get(k, 0.0))
        s += pk * math.log(pk / qk)
    return float(s)


def js_divergence(p: Dict[int,float], q: Dict[int,float], eps: float=1e-12) -> float:
    m: Dict[int,float] = {}
    keys = set(p.keys()) | set(q.keys())
    for k in keys:
        m[k] = 0.5*(p.get(k,0.0)+q.get(k,0.0))
    return 0.5*kl_divergence(p,m,eps) + 0.5*kl_divergence(q,m,eps)


def cosine_similarity(p: Dict[int,float], q: Dict[int,float], eps: float=1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    dot = sum(p.get(k,0.0)*q.get(k,0.0) for k in keys)
    np = math.sqrt(sum((v*v) for v in p.values()))
    nq = math.sqrt(sum((v*v) for v in q.values()))
    if np<eps or nq<eps:
        return 0.0
    return float(dot/(np*nq))


def main():
    parser = argparse.ArgumentParser(description="对比学生/教师的GSM8K结果并计算蒸馏指标")
    parser.add_argument("--student_jsonl", required=True, help="学生JSONL路径")
    parser.add_argument("--teacher_jsonl", required=True, help="教师JSONL路径")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    stu = load_jsonl(args.student_jsonl)
    tea = load_jsonl(args.teacher_jsonl)

    common_idx = sorted(set(stu.keys()) & set(tea.keys()))
    logging.info(f"样本对齐：{len(common_idx)} 条")
    
    # 过滤掉有错误的样本（用于准确率计算）
    valid_idx = [i for i in common_idx 
                 if not stu[i].get("error") and not tea[i].get("error")]
    
    logging.info(f"有效样本（无错误）: {len(valid_idx)}/{len(common_idx)}")

    acc_stu = accuracy({i:stu[i] for i in valid_idx}) if valid_idx else 0.0
    acc_tea = accuracy({i:tea[i] for i in valid_idx}) if valid_idx else 0.0
    logging.info(f"学生准确率: {acc_stu:.4f}")
    logging.info(f"教师准确率: {acc_tea:.4f}")

    # 蒸馏指标（基于prompt处top-k分布，只计算有效样本）
    kls, r_kls, jss, coss = [], [], [], []
    for i in valid_idx:
        p = sparse_vec(stu[i].get("top_ids",[]), stu[i].get("top_probs",[]))
        q = sparse_vec(tea[i].get("top_ids",[]), tea[i].get("top_probs",[]))
        if not p or not q:
            continue
        kls.append(kl_divergence(p,q))
        r_kls.append(kl_divergence(q,p))
        jss.append(js_divergence(p,q))
        coss.append(cosine_similarity(p,q))

    def avg(x: List[float]) -> float:
        return float(sum(x)/len(x)) if x else float('nan')

    logging.info("蒸馏近似指标（prompt分布top-k）")
    logging.info(f"  KL(student||teacher): {avg(kls):.6f} (n={len(kls)})")
    logging.info(f"  KL(teacher||student): {avg(r_kls):.6f}")
    logging.info(f"  JS: {avg(jss):.6f}")
    logging.info(f"  Cosine: {avg(coss):.6f}")


if __name__ == "__main__":
    main()


