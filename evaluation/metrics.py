"""
评估指标模块
功能：定义和计算各种评估指标
"""

import torch
import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple
import logging
from collections import defaultdict


class MathAccuracyMetrics:
    """数学准确率指标"""
    
    @staticmethod
    def exact_match_accuracy(predictions: List[str], 
                           ground_truths: List[str]) -> float:
        """
        计算完全匹配准确率
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实答案列表
            
        Returns:
            完全匹配准确率
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测和真实答案数量不匹配")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.strip().lower() == gt.strip().lower():
                correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def numerical_accuracy(predictions: List[str], 
                          ground_truths: List[str], 
                          tolerance: float = 1e-6) -> float:
        """
        计算数值准确率
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实答案列表
            tolerance: 数值容差
            
        Returns:
            数值准确率
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测和真实答案数量不匹配")
        
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_num = MathAccuracyMetrics._extract_number(pred)
            gt_num = MathAccuracyMetrics._extract_number(gt)
            
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) < tolerance:
                    correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """从文本中提取数字"""
        # 查找最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None


class ReasoningQualityMetrics:
    """推理质量指标"""
    
    @staticmethod
    def step_coverage_ratio(student_steps: List[str], 
                           teacher_steps: List[str]) -> float:
        """
        计算步骤覆盖率
        
        Args:
            student_steps: 学生推理步骤
            teacher_steps: 教师推理步骤
            
        Returns:
            步骤覆盖率
        """
        if not teacher_steps:
            return 0.0
        
        # 提取关键信息
        student_keywords = ReasoningQualityMetrics._extract_keywords(student_steps)
        teacher_keywords = ReasoningQualityMetrics._extract_keywords(teacher_steps)
        
        # 计算交集
        common_keywords = set(student_keywords) & set(teacher_keywords)
        
        return len(common_keywords) / len(teacher_keywords)
    
    @staticmethod
    def logical_consistency_score(response: str) -> float:
        """
        计算逻辑一致性得分
        
        Args:
            response: 模型响应
            
        Returns:
            逻辑一致性得分
        """
        # 提取数字序列
        numbers = re.findall(r'-?\d+\.?\d*', response)
        numbers = [float(n) for n in numbers if n]
        
        if len(numbers) < 2:
            return 1.0
        
        # 检查数字的合理性
        score = 1.0
        
        # 检查是否有异常值
        for i in range(len(numbers) - 1):
            if numbers[i] > 10000 or numbers[i] < -1000:
                score -= 0.1
        
        # 检查递增/递减逻辑
        if len(numbers) >= 3:
            diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                score += 0.1  # 有明确趋势
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def _extract_keywords(steps: List[str]) -> List[str]:
        """提取关键词"""
        keywords = []
        for step in steps:
            # 提取数字
            numbers = re.findall(r'\d+', step)
            keywords.extend(numbers)
            
            # 提取操作符
            operators = re.findall(r'[+\-*/=]', step)
            keywords.extend(operators)
        
        return keywords


class DistillationMetrics:
    """知识蒸馏指标"""
    
    @staticmethod
    def kl_divergence(student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor) -> float:
        """
        计算KL散度
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            
        Returns:
            KL散度
        """
        # 计算概率分布
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        # 计算KL散度
        kl_div = torch.sum(teacher_probs * torch.log(teacher_probs / (student_probs + 1e-8)), dim=-1)
        
        return kl_div.mean().item()
    
    @staticmethod
    def cosine_similarity(student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor) -> float:
        """
        计算余弦相似度
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            
        Returns:
            余弦相似度
        """
        # 展平logits
        student_flat = student_logits.view(-1)
        teacher_flat = teacher_logits.view(-1)
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            student_flat.unsqueeze(0), 
            teacher_flat.unsqueeze(0)
        )
        
        return cos_sim.item()
    
    @staticmethod
    def js_divergence(student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor) -> float:
        """
        计算Jensen-Shannon散度
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            
        Returns:
            JS散度
        """
        # 计算概率分布
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        # 计算平均分布
        mean_probs = (student_probs + teacher_probs) / 2
        
        # 计算JS散度
        kl_student = torch.sum(student_probs * torch.log(student_probs / (mean_probs + 1e-8)), dim=-1)
        kl_teacher = torch.sum(teacher_probs * torch.log(teacher_probs / (mean_probs + 1e-8)), dim=-1)
        
        js_div = (kl_student + kl_teacher) / 2
        
        return js_div.mean().item()


class TrainingMetrics:
    """训练指标"""
    
    @staticmethod
    def convergence_rate(loss_history: List[float], 
                        window_size: int = 100) -> float:
        """
        计算收敛率
        
        Args:
            loss_history: 损失历史
            window_size: 窗口大小
            
        Returns:
            收敛率
        """
        if len(loss_history) < window_size * 2:
            return 0.0
        
        # 计算最近窗口的平均损失
        recent_loss = np.mean(loss_history[-window_size:])
        
        # 计算早期窗口的平均损失
        early_loss = np.mean(loss_history[:window_size])
        
        # 计算收敛率
        convergence_rate = (early_loss - recent_loss) / early_loss
        
        return max(0.0, convergence_rate)
    
    @staticmethod
    def training_stability(loss_history: List[float], 
                          window_size: int = 50) -> float:
        """
        计算训练稳定性
        
        Args:
            loss_history: 损失历史
            window_size: 窗口大小
            
        Returns:
            稳定性得分（越高越稳定）
        """
        if len(loss_history) < window_size:
            return 0.0
        
        # 计算最近窗口的标准差
        recent_losses = loss_history[-window_size:]
        stability = 1.0 / (1.0 + np.std(recent_losses))
        
        return stability
    
    @staticmethod
    def learning_efficiency(reward_history: List[float], 
                           step_history: List[int]) -> float:
        """
        计算学习效率
        
        Args:
            reward_history: 奖励历史
            step_history: 步数历史
            
        Returns:
            学习效率
        """
        if len(reward_history) < 2 or len(step_history) < 2:
            return 0.0
        
        # 计算奖励改善率
        initial_reward = np.mean(reward_history[:10]) if len(reward_history) >= 10 else reward_history[0]
        final_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else reward_history[-1]
        
        reward_improvement = final_reward - initial_reward
        
        # 计算步数
        total_steps = step_history[-1] - step_history[0]
        
        # 学习效率 = 奖励改善 / 步数
        efficiency = reward_improvement / max(1, total_steps)
        
        return efficiency


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        """初始化综合评估器"""
        self.math_metrics = MathAccuracyMetrics()
        self.reasoning_metrics = ReasoningQualityMetrics()
        self.distillation_metrics = DistillationMetrics()
        self.training_metrics = TrainingMetrics()
    
    def evaluate_comprehensive(self, 
                             predictions: List[str],
                             ground_truths: List[str],
                             student_logits: Optional[torch.Tensor] = None,
                             teacher_logits: Optional[torch.Tensor] = None,
                             training_history: Optional[Dict] = None) -> Dict[str, float]:
        """
        综合评估
        
        Args:
            predictions: 预测结果
            ground_truths: 真实答案
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            training_history: 训练历史
            
        Returns:
            综合评估结果
        """
        results = {}
        
        # 数学准确率
        results["exact_match_accuracy"] = self.math_metrics.exact_match_accuracy(predictions, ground_truths)
        results["numerical_accuracy"] = self.math_metrics.numerical_accuracy(predictions, ground_truths)
        
        # 推理质量（如果有logits）
        if student_logits is not None and teacher_logits is not None:
            results["kl_divergence"] = self.distillation_metrics.kl_divergence(student_logits, teacher_logits)
            results["cosine_similarity"] = self.distillation_metrics.cosine_similarity(student_logits, teacher_logits)
            results["js_divergence"] = self.distillation_metrics.js_divergence(student_logits, teacher_logits)
        
        # 训练指标（如果有训练历史）
        if training_history is not None:
            if "loss_history" in training_history:
                results["convergence_rate"] = self.training_metrics.convergence_rate(training_history["loss_history"])
                results["training_stability"] = self.training_metrics.training_stability(training_history["loss_history"])
            
            if "reward_history" in training_history and "step_history" in training_history:
                results["learning_efficiency"] = self.training_metrics.learning_efficiency(
                    training_history["reward_history"], 
                    training_history["step_history"]
                )
        
        # 综合得分
        accuracy_weight = 0.4
        distillation_weight = 0.3
        training_weight = 0.3
        
        overall_score = 0.0
        total_weight = 0.0
        
        # 准确率部分
        overall_score += results["numerical_accuracy"] * accuracy_weight
        total_weight += accuracy_weight
        
        # 蒸馏部分
        if "cosine_similarity" in results:
            overall_score += results["cosine_similarity"] * distillation_weight
            total_weight += distillation_weight
        
        # 训练部分
        if "training_stability" in results:
            overall_score += results["training_stability"] * training_weight
            total_weight += training_weight
        
        if total_weight > 0:
            results["overall_score"] = overall_score / total_weight
        else:
            results["overall_score"] = results["numerical_accuracy"]
        
        return results


def create_comprehensive_evaluator() -> ComprehensiveEvaluator:
    """创建综合评估器的便捷函数"""
    return ComprehensiveEvaluator()






