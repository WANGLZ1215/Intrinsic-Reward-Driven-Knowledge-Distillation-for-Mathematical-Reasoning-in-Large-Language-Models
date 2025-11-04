"""
Reasoning Quality Evaluation Module
Function: Evaluate mathematical reasoning quality, including step coverage, logical consistency, etc.
"""

import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from collections import Counter
import sympy as sp
from utils.math_utils import extract_answer_unified  # 导入统一的答案提取函数


class ReasoningEvaluator:
    """Reasoning Quality Evaluator"""
    
    def __init__(self):
        """初始化评估器"""
        # 推理步骤模式
        self.step_patterns = [
            r'Step \d+:',      # "Step 1:", "Step 2:", etc.
            r'\d+\.',          # "1.", "2.", etc.
            r'First,',         # "First,", "Second,", etc.
            r'Then,',          # "Then,", "Next,", etc.
            r'Therefore,',     # "Therefore,", "Thus,", etc.
            r'So,',            # "So,", "Hence,", etc.
            r'Finally,',       # "Finally,", "In conclusion,", etc.
        ]
        
        # 数学操作符
        self.math_operators = ['+', '-', '*', '/', '=', '^', '**']
        
        # 逻辑连接词
        self.logical_connectors = ['therefore', 'thus', 'so', 'hence', 'because', 'since']
        
        logging.info("Reasoning quality evaluator initialized")
    
    def extract_reasoning_steps(self, response: str) -> List[str]:
        """
        提取推理步骤
        
        Args:
            response: 模型响应文本
            
        Returns:
            推理步骤列表
        """
        steps = []
        
        # 按行分割
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 检查是否匹配步骤模式
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.step_patterns):
                steps.append(line)
            elif line and self._is_math_expression(line):
                # 如果包含数学表达式，也认为是推理步骤
                steps.append(line)
        
        return steps
    
    def _is_math_expression(self, text: str) -> bool:
        """
        判断文本是否包含数学表达式
        
        Args:
            text: 输入文本
            
        Returns:
            是否包含数学表达式
        """
        # 检查是否包含数学操作符
        has_operator = any(op in text for op in self.math_operators)
        
        # 检查是否包含数字
        has_numbers = bool(re.search(r'\d+', text))
        
        # 检查是否是数学关键词
        math_keywords = ['calculate', 'compute', 'solve', 'find', 'result', 'answer']
        has_math_keyword = any(keyword in text.lower() for keyword in math_keywords)
        
        return has_operator and (has_numbers or has_math_keyword)
    
    def evaluate_step_coverage(self, student_steps: List[str], 
                             teacher_steps: List[str]) -> Dict[str, float]:
        """
        评估步骤覆盖率
        
        Args:
            student_steps: 学生推理步骤
            teacher_steps: 教师推理步骤
            
        Returns:
            覆盖率评估结果
        """
        if not teacher_steps:
            return {
                "step_coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # 提取关键数学表达式
        student_expressions = self._extract_math_expressions(student_steps)
        teacher_expressions = self._extract_math_expressions(teacher_steps)
        
        # 计算交集
        common_expressions = set(student_expressions) & set(teacher_expressions)
        
        # 计算指标
        precision = len(common_expressions) / max(1, len(student_expressions))
        recall = len(common_expressions) / len(teacher_expressions)
        f1_score = 2 * precision * recall / max(1, precision + recall)
        
        return {
            "step_coverage": recall,  # 使用recall作为覆盖率
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "student_steps": len(student_steps),
            "teacher_steps": len(teacher_steps),
            "common_steps": len(common_expressions)
        }
    
    def _extract_math_expressions(self, steps: List[str]) -> List[str]:
        """提取数学表达式"""
        expressions = []
        
        for step in steps:
            # 查找数学表达式模式
            patterns = [
                r'\d+\s*[+\-*/=]\s*\d+',  # 基本运算
                r'\d+\s*=\s*\d+',          # 等式
                r'[a-zA-Z]\s*=\s*\d+',     # 变量赋值
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, step)
                expressions.extend(matches)
        
        return expressions
    
    def evaluate_logical_consistency(self, response: str) -> Dict[str, float]:
        """
        评估逻辑一致性
        
        Args:
            response: 模型响应
            
        Returns:
            逻辑一致性评估结果
        """
        # 提取数字序列
        numbers = self._extract_numbers(response)
        
        # 检查数字的合理性
        number_consistency = self._check_number_consistency(numbers)
        
        # 检查逻辑连接词的使用
        logical_flow = self._check_logical_flow(response)
        
        # 检查数学表达式的有效性
        expression_validity = self._check_expression_validity(response)
        
        # 综合评分
        overall_consistency = (number_consistency + logical_flow + expression_validity) / 3
        
        return {
            "overall_consistency": overall_consistency,
            "number_consistency": number_consistency,
            "logical_flow": logical_flow,
            "expression_validity": expression_validity,
            "extracted_numbers": numbers
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取文本中的数字"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]
    
    def _check_number_consistency(self, numbers: List[float]) -> float:
        """
        检查数字的一致性（基于数字序列的平滑度）
        
        移除了启发式的负数/大数过滤，因为：
        - 题目本身可能涉及负数（亏损、温度、债务等）
        - 题目可能涉及大数（人口、距离、金额等）
        """
        if not numbers:
            return 0.0
        
        if len(numbers) < 2:
            return 1.0
        
        # 计算数字序列的变化平滑度
        # 检查相邻数字之间的变化是否合理（不要有突然的巨大跳跃）
        score = 1.0
        changes = []
        
        for i in range(len(numbers) - 1):
            if numbers[i] != 0:
                # 计算相对变化
                relative_change = abs((numbers[i+1] - numbers[i]) / numbers[i])
                changes.append(relative_change)
        
        if changes:
            # 如果有过多的极端变化（超过1000倍），可能有问题
            extreme_changes = sum(1 for change in changes if change > 1000)
            if extreme_changes > len(changes) * 0.5:
                score -= 0.3
        
        return max(0.0, score)
    
    def _check_logical_flow(self, text: str) -> float:
        """检查逻辑流程"""
        text_lower = text.lower()
        
        # 检查逻辑连接词的使用
        connector_count = sum(1 for connector in self.logical_connectors 
                             if connector in text_lower)
        
        # 检查是否有明确的推理结构
        structure_score = 0.0
        
        if 'step' in text_lower or 'first' in text_lower:
            structure_score += 0.3
        if 'then' in text_lower or 'next' in text_lower:
            structure_score += 0.3
        if 'therefore' in text_lower or 'so' in text_lower or 'thus' in text_lower:
            structure_score += 0.4
        
        # 归一化分数
        normalized_connector_score = min(1.0, connector_count / 3.0)
        
        return min(1.0, structure_score + normalized_connector_score * 0.3)
    
    def _check_expression_validity(self, text: str) -> float:
        """
        检查数学表达式的有效性
        
        改进：使用更宽松的标准，不依赖eval（eval对复杂推理文本无效）
        只检查是否有合理的数学结构
        """
        # 提取数学表达式
        expressions = self._extract_math_expressions([text])
        
        if not expressions:
            # 没有明确的数学表达式不代表推理无效
            # 检查是否至少有数字和推理词汇
            has_numbers = bool(re.search(r'\d+', text))
            has_reasoning = any(word in text.lower() for word in 
                              ['calculate', 'compute', 'total', 'sum', 'multiply', 'divide', 'add', 'subtract'])
            if has_numbers and has_reasoning:
                return 0.7  # 有数字和推理词汇，给较高分
            elif has_numbers:
                return 0.5  # 只有数字，给中等分
            else:
                return 0.3  # 缺乏数学内容
        
        # 检查表达式的结构合理性
        valid_count = 0
        for expr in expressions:
            # 检查是否有基本的数学结构（数字 + 运算符 + 数字）
            if re.search(r'\d+\s*[+\-*/]\s*\d+', expr):
                valid_count += 1
            # 检查等式结构（左边 = 右边，都有数字）
            elif '=' in expr:
                parts = expr.split('=')
                if len(parts) == 2 and all(re.search(r'\d+', part) for part in parts):
                    valid_count += 1
        
        if len(expressions) == 0:
            return 0.5
        
        return max(0.3, valid_count / len(expressions))  # 至少给0.3分
    
    def extract_final_answer(self, text: str) -> Optional[float]:
        """
        提取文本中的最终答案
        
        注意：此方法现在调用 utils.math_utils.extract_answer_unified 统一实现
        支持多种格式：####, \\boxed{}, "answer:", "The answer is"
        
        Args:
            text: 响应文本
            
        Returns:
            提取的数字答案，如果无法提取则返回None
        """
        _, answer_num = extract_answer_unified(text)
        return answer_num
    
    def evaluate_answer_correctness(self, student_response: str, 
                                   ground_truth_answer: float,
                                   tolerance: float = 1e-4) -> Dict[str, Union[float, bool]]:
        """
        评估最终答案的正确性
        
        Args:
            student_response: 学生响应
            ground_truth_answer: 正确答案（数值）
            tolerance: 容差范围
            
        Returns:
            答案正确性评估结果
        """
        student_answer = self.extract_final_answer(student_response)
        
        if student_answer is None:
            # 无法提取答案，视为错误
            return {
                "is_correct": False,
                "correctness_score": 0.0,
                "student_answer": None,
                "ground_truth": ground_truth_answer,
                "error": "Unable to extract answer"
            }
        
        # 计算相对误差
        if abs(ground_truth_answer) < 1e-10:
            # 真值接近0，使用绝对误差
            is_correct = abs(student_answer - ground_truth_answer) < tolerance
            relative_error = abs(student_answer - ground_truth_answer)
        else:
            # 使用相对误差
            relative_error = abs(student_answer - ground_truth_answer) / abs(ground_truth_answer)
            is_correct = relative_error < tolerance
        
        # 计算连续的正确性分数（即使不完全正确，也给予部分分数）
        if is_correct:
            correctness_score = 1.0
        else:
            # 基于误差给予部分分数
            if relative_error < 0.01:  # 1%以内的误差
                correctness_score = 0.9
            elif relative_error < 0.05:  # 5%以内的误差
                correctness_score = 0.7
            elif relative_error < 0.1:  # 10%以内的误差
                correctness_score = 0.5
            elif relative_error < 0.5:  # 50%以内的误差
                correctness_score = 0.2
            else:
                correctness_score = 0.0
        
        return {
            "is_correct": is_correct,
            "correctness_score": correctness_score,
            "student_answer": student_answer,
            "ground_truth": ground_truth_answer,
            "relative_error": relative_error
        }
    
    def compute_kl_divergence(self, student_logits: torch.Tensor, 
                            teacher_logits: torch.Tensor) -> float:
        """
        计算KL散度
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            
        Returns:
            KL散度
        """
        # 🔍 安全检查：确保logits不为空且维度正确
        if student_logits is None or teacher_logits is None:
            return 0.0
        
        # 确保是tensor类型
        if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
            return 0.0
        
        # 检查tensor是否为空
        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            return 0.0
        
        # 🔍 修复索引越界：确保维度正确
        try:
            # 检查维度（应该是 [batch, seq_len, vocab_size]）
            if len(student_logits.shape) < 2 or len(teacher_logits.shape) < 2:
                return 0.0
            
            # 如果只有2维，添加batch维度
            if len(student_logits.shape) == 2:
                student_logits = student_logits.unsqueeze(0)
            if len(teacher_logits.shape) == 2:
                teacher_logits = teacher_logits.unsqueeze(0)
            
            # 确保维度匹配（取最小序列长度，避免索引越界）
            student_seq_len = student_logits.shape[1]
            teacher_seq_len = teacher_logits.shape[1]
            
            if student_seq_len == 0 or teacher_seq_len == 0:
                return 0.0
            
            min_len = min(student_seq_len, teacher_seq_len)
            
            # 🔍 安全切片：确保索引在有效范围内
            if min_len > 0:
                student_logits = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
            else:
                return 0.0
            
            # 计算概率分布
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # 计算KL散度
            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='batchmean'
            )
            
            return kl_div.item() if not torch.isnan(kl_div) else 0.0
            
        except (IndexError, RuntimeError, ValueError) as e:
            # 捕获索引越界或其他运行时错误
            logging.warning(f"计算KL散度时出错（可能由于tensor维度不匹配）: {e}")
            return 0.0
    
    def evaluate_reasoning_quality(self, student_response: str, 
                                 teacher_response: str,
                                 ground_truth_answer: Optional[float] = None,
                                 student_logits: Optional[torch.Tensor] = None,
                                 teacher_logits: Optional[torch.Tensor] = None) -> Dict[str, Union[float, Dict]]:
        """
        综合评估推理质量
        
        改进：增加了最终答案正确性作为独立维度，并给予更高权重
        
        权重分配：
        - 答案正确性：50% （最重要，特别是在GSM8K等数学任务中）
        - 步骤覆盖率：20% （推理过程的完整性）
        - 逻辑一致性：20% （推理过程的合理性）
        - KL散度：10% （与教师模型的一致性）
        
        Args:
            student_response: 学生响应
            teacher_response: 教师响应
            ground_truth_answer: 正确答案（可选，强烈推荐提供）
            student_logits: 学生模型logits（可选）
            teacher_logits: 教师模型logits（可选）
            
        Returns:
            推理质量评估结果
        """
        # 提取推理步骤
        student_steps = self.extract_reasoning_steps(student_response)
        teacher_steps = self.extract_reasoning_steps(teacher_response)
        
        # 评估步骤覆盖率
        step_coverage_results = self.evaluate_step_coverage(student_steps, teacher_steps)
        
        # 评估逻辑一致性
        logical_consistency_results = self.evaluate_logical_consistency(student_response)
        
        # 评估最终答案正确性
        answer_correctness_results = None
        if ground_truth_answer is not None:
            answer_correctness_results = self.evaluate_answer_correctness(
                student_response, ground_truth_answer
            )
        
        # 计算KL散度（如果有logits）
        kl_divergence = 0.0
        if student_logits is not None and teacher_logits is not None:
            kl_divergence = self.compute_kl_divergence(student_logits, teacher_logits)
        
        # 综合评分
        if answer_correctness_results is not None:
            # 有正确答案时，使用新的权重分配
            overall_score = (
                answer_correctness_results["correctness_score"] * 0.5 +  # 答案正确性 50%
                step_coverage_results["step_coverage"] * 0.2 +           # 步骤覆盖率 20%
                logical_consistency_results["overall_consistency"] * 0.2 + # 逻辑一致性 20%
                (1.0 / (1.0 + kl_divergence)) * 0.1                       # KL散度 10%
            )
        else:
            # 没有正确答案时，使用旧的权重（但调整为更合理的分配）
            overall_score = (
                step_coverage_results["step_coverage"] * 0.35 +
                logical_consistency_results["overall_consistency"] * 0.35 +
                (1.0 / (1.0 + kl_divergence)) * 0.3
            )
        
        result = {
            "overall_score": overall_score,
            "step_coverage": step_coverage_results,
            "logical_consistency": logical_consistency_results,
            "kl_divergence": kl_divergence,
            "student_steps_count": len(student_steps),
            "teacher_steps_count": len(teacher_steps)
        }
        
        # 如果有答案正确性结果，添加到返回值中
        if answer_correctness_results is not None:
            result["answer_correctness"] = answer_correctness_results
        
        return result


class BatchReasoningEvaluator:
    """Batch Reasoning Evaluator"""
    
    def __init__(self, evaluator: ReasoningEvaluator):
        """
        初始化批量评估器
        
        Args:
            evaluator: 推理评估器
        """
        self.evaluator = evaluator
    
    def evaluate_batch(self, student_responses: List[str],
                      teacher_responses: List[str],
                      ground_truth_answers: Optional[List[float]] = None,
                      student_logits_list: Optional[List[torch.Tensor]] = None,
                      teacher_logits_list: Optional[List[torch.Tensor]] = None) -> Dict[str, List]:
        """
        批量评估推理质量
        
        Args:
            student_responses: 学生响应列表
            teacher_responses: 教师响应列表
            ground_truth_answers: 正确答案列表（可选，推荐提供）
            student_logits_list: 学生logits列表（可选）
            teacher_logits_list: 教师logits列表（可选）
            
        Returns:
            批量评估结果
        """
        batch_results = []
        
        for i, (student_resp, teacher_resp) in enumerate(zip(student_responses, teacher_responses)):
            student_logits = student_logits_list[i] if student_logits_list else None
            teacher_logits = teacher_logits_list[i] if teacher_logits_list else None
            ground_truth = ground_truth_answers[i] if ground_truth_answers else None
            
            result = self.evaluator.evaluate_reasoning_quality(
                student_resp, teacher_resp, ground_truth, student_logits, teacher_logits
            )
            
            batch_results.append(result)
        
        # 计算批量统计
        batch_stats = self._compute_batch_statistics(batch_results)
        
        return {
            "individual_results": batch_results,
            "batch_statistics": batch_stats
        }
    
    def _compute_batch_statistics(self, batch_results: List[Dict]) -> Dict[str, float]:
        """计算批量统计信息"""
        if not batch_results:
            return {}
        
        # 提取各项指标
        overall_scores = [result["overall_score"] for result in batch_results]
        step_coverage_scores = [result["step_coverage"]["step_coverage"] for result in batch_results]
        logical_consistency_scores = [result["logical_consistency"]["overall_consistency"] for result in batch_results]
        kl_divergences = [result["kl_divergence"] for result in batch_results]
        
        stats = {
            "mean_overall_score": np.mean(overall_scores),
            "std_overall_score": np.std(overall_scores),
            "mean_step_coverage": np.mean(step_coverage_scores),
            "mean_logical_consistency": np.mean(logical_consistency_scores),
            "mean_kl_divergence": np.mean(kl_divergences),
            "batch_size": len(batch_results)
        }
        
        # 如果有答案正确性数据，也计算其统计信息
        if "answer_correctness" in batch_results[0]:
            correctness_scores = [
                result["answer_correctness"]["correctness_score"] 
                for result in batch_results 
                if "answer_correctness" in result
            ]
            is_correct_list = [
                result["answer_correctness"]["is_correct"] 
                for result in batch_results 
                if "answer_correctness" in result
            ]
            
            if correctness_scores:
                stats["mean_answer_correctness"] = np.mean(correctness_scores)
                stats["std_answer_correctness"] = np.std(correctness_scores)
                stats["accuracy"] = np.mean(is_correct_list)  # 完全正确的比例
                stats["correct_count"] = sum(is_correct_list)
                stats["total_count"] = len(is_correct_list)
        
        return stats


def create_reasoning_evaluator() -> ReasoningEvaluator:
    """创建推理评估器的便捷函数"""
    return ReasoningEvaluator()






