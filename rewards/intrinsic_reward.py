"""
Intrinsic Reward Computation Module
Function: Compute intrinsic rewards based on paper theory, implement inverse soft Bellman operator
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path


class IntrinsicRewardComputer:
    """Intrinsic Reward Computer"""
    
    def __init__(self, temperature: float = 1.0, 
                 normalization_method: str = "mean_std",
                 update_rate: float = 0.01):
        """
        初始化内在奖励计算器
        
        Args:
            temperature: 温度参数α
            normalization_method: 归一化方法
            update_rate: 统计信息更新率
        """
        self.temperature = temperature
        self.normalization_method = normalization_method
        self.update_rate = update_rate
        
        # 统计信息
        self.intrinsic_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -float('inf'),
            "max": float('inf'),
            "count": 0
        }
        
        logging.info(f"Intrinsic reward computer initialized: temperature={temperature}, normalization={normalization_method}")
    
    def compute_intrinsic_reward(self, teacher_logits: torch.Tensor, 
                               student_tokens: torch.Tensor,
                               question_length: int = 0) -> torch.Tensor:
        """
        基于论文Equation (10)计算内在奖励
        
        Args:
            teacher_logits: 教师模型logits [batch_size, full_seq_len, vocab_size] 完整序列logits
            student_tokens: 学生模型生成的token [batch_size, response_len] 学生生成部分
            question_length: 问题部分的长度，用于提取学生生成部分对应的logits
            
        Returns:
            内在奖励 [batch_size, response_len]
        """
        alpha = max(self.temperature, 1e-8)  # 数值稳定性
        
        # 确保 student_tokens 与 teacher_logits 在同一设备上
        student_tokens = student_tokens.to(teacher_logits.device)
        
        batch_size, response_len = student_tokens.shape
        full_seq_len = teacher_logits.shape[1]
        
        # 确保question_length不超过teacher_logits的长度
        if question_length >= full_seq_len:
            question_length = 0
        
        # 计算学生生成部分在完整序列中的起始位置
        start_pos = question_length
        end_pos = min(start_pos + response_len, full_seq_len)
        
        # 提取教师模型对学生生成部分的logits
        if end_pos > start_pos and start_pos < full_seq_len:
            student_logits = teacher_logits[:, start_pos:end_pos, :]
            actual_response_len = end_pos - start_pos
        else:
            # 如果无法提取，返回零奖励
            logging.warning(f"无法提取学生生成部分的logits: start_pos={start_pos}, end_pos={end_pos}, full_seq_len={full_seq_len}")
            return torch.zeros_like(student_tokens, dtype=torch.float32)
        
        # 计算价值函数 V_Q̂(s_h) = α log(∑ exp(Q̂(s_h, a_h)/α))
        value_function = alpha * torch.logsumexp(student_logits / alpha, dim=-1)
        
        # 初始化奖励张量
        intrinsic_rewards = torch.zeros_like(student_tokens, dtype=torch.float32)
        
        # 计算逐token内在奖励
        for h in range(actual_response_len - 1):  # 最后一个token没有下一状态
            # 当前状态的Q值
            current_q = student_logits[:, h, :]  # [batch_size, vocab_size]
            
            # 学生模型选择的token的Q值
            selected_token_q = current_q.gather(dim=-1, index=student_tokens[:, h:h+1]).squeeze(-1)
            
            # 下一状态的价值函数
            next_value = value_function[:, h + 1]
            
            # 内在奖励 = Q值 - 下一状态价值
            intrinsic_rewards[:, h] = selected_token_q - next_value
        
        # 最后一个token的奖励（只有Q值，没有下一状态）
        if actual_response_len > 0:
            last_q = student_logits[:, actual_response_len - 1, :]
            last_selected_q = last_q.gather(dim=-1, index=student_tokens[:, actual_response_len - 1:actual_response_len]).squeeze(-1)
            intrinsic_rewards[:, actual_response_len - 1] = last_selected_q
        
        # 奖励裁剪以提高训练稳定性
        intrinsic_rewards = torch.clamp(intrinsic_rewards, -10.0, 10.0)
        
        return intrinsic_rewards
    
    def compute_outcome_reward(self, teacher_model, question: str, 
                             student_response: str) -> float:
        """
        基于论文Equation (12)计算完整响应的结果奖励
        
        Args:
            teacher_model: 教师模型
            question: 输入问题
            student_response: 学生模型的完整响应
            
        Returns:
            结果奖励值
        """
        alpha = self.temperature
        
        # 构建完整序列
        full_sequence = question + student_response
        
        with torch.no_grad():
            # 获取教师模型logits
            inputs = teacher_model.tokenizer(
                full_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(teacher_model.model.device) for k, v in inputs.items()}
            outputs = teacher_model.model(**inputs)
            logits = outputs.logits
            
            # 计算完整响应的log概率
            response_tokens = teacher_model.tokenizer.encode(student_response, add_special_tokens=False)
            
            total_log_prob = 0.0
            for i, token_id in enumerate(response_tokens):
                if i < logits.shape[1]:
                    token_log_prob = F.log_softmax(logits[0, i], dim=-1)[token_id]
                    total_log_prob += token_log_prob.item()
            
            # 结果奖励 = α * log(π̂(τ|s₁))
            outcome_reward = alpha * total_log_prob
            
            return outcome_reward
    
    def compute_token_level_rewards(self, teacher_logits: torch.Tensor,
                                  student_tokens: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算token级别的奖励（更稳定的方法）
        
        Args:
            teacher_logits: 教师模型logits
            student_tokens: 学生模型tokens
            attention_mask: 注意力掩码
            
        Returns:
            token级别奖励
        """
        # 计算内在奖励
        intrinsic_rewards = self.compute_intrinsic_reward(teacher_logits, student_tokens)
        
        # 如果有注意力掩码，将padding位置的奖励设为0
        if attention_mask is not None:
            intrinsic_rewards = intrinsic_rewards * attention_mask.float()
        
        return intrinsic_rewards
    
    def compute_trajectory_reward(self, token_rewards: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将token级别奖励累加为trajectory级别奖励
        
        Args:
            token_rewards: token级别奖励
            attention_mask: 注意力掩码
            
        Returns:
            trajectory级别奖励
        """
        if attention_mask is not None:
            # 只对有效token求和
            trajectory_rewards = (token_rewards * attention_mask.float()).sum(dim=-1)
        else:
            trajectory_rewards = token_rewards.sum(dim=-1)
        
        return trajectory_rewards
    
    def update_statistics(self, rewards: torch.Tensor):
        """
        更新奖励统计信息
        
        Args:
            rewards: 奖励张量
        """
        if rewards.numel() == 0:
            return
        
        # 展平奖励
        flat_rewards = rewards.flatten()
        
        # 计算当前批次统计
        batch_mean = torch.mean(flat_rewards).item()
        batch_std = torch.std(flat_rewards).item()
        batch_min = torch.min(flat_rewards).item()
        batch_max = torch.max(flat_rewards).item()
        batch_count = flat_rewards.numel()
        
        # 更新全局统计（指数移动平均）
        alpha = self.update_rate
        
        if self.intrinsic_stats["count"] == 0:
            # 第一次更新
            self.intrinsic_stats["mean"] = batch_mean
            self.intrinsic_stats["std"] = batch_std
            self.intrinsic_stats["min"] = batch_min
            self.intrinsic_stats["max"] = batch_max
        else:
            # 指数移动平均
            self.intrinsic_stats["mean"] = (1 - alpha) * self.intrinsic_stats["mean"] + alpha * batch_mean
            self.intrinsic_stats["std"] = (1 - alpha) * self.intrinsic_stats["std"] + alpha * batch_std
            self.intrinsic_stats["min"] = min(self.intrinsic_stats["min"], batch_min)
            self.intrinsic_stats["max"] = max(self.intrinsic_stats["max"], batch_max)
        
        self.intrinsic_stats["count"] += batch_count
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return self.intrinsic_stats.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.intrinsic_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": -float('inf'),
            "max": float('inf'),
            "count": 0
        }


class IntrinsicRewardBatchProcessor:
    """Intrinsic Reward Batch Processor"""
    
    def __init__(self, intrinsic_computer: IntrinsicRewardComputer):
        """
        初始化批处理器
        
        Args:
            intrinsic_computer: 内在奖励计算器
        """
        self.intrinsic_computer = intrinsic_computer
    
    def process_batch(self, teacher_logits_list: List[torch.Tensor],
                     student_tokens_list: List[torch.Tensor],
                     attention_masks: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        批量处理内在奖励计算
        
        Args:
            teacher_logits_list: 教师logits列表
            student_tokens_list: 学生tokens列表
            attention_masks: 注意力掩码列表
            
        Returns:
            奖励列表
        """
        rewards_list = []
        
        for i, (teacher_logits, student_tokens) in enumerate(zip(teacher_logits_list, student_tokens_list)):
            attention_mask = attention_masks[i] if attention_masks else None
            
            # 计算token级别奖励
            token_rewards = self.intrinsic_computer.compute_token_level_rewards(
                teacher_logits.unsqueeze(0), 
                student_tokens.unsqueeze(0),
                attention_mask.unsqueeze(0) if attention_mask is not None else None
            )
            
            rewards_list.append(token_rewards.squeeze(0))
        
        return rewards_list


def create_intrinsic_reward_computer(config: Dict) -> IntrinsicRewardComputer:
    """
    创建内在奖励计算器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        内在奖励计算器实例
    """
    return IntrinsicRewardComputer(
        temperature=config["intrinsic_reward"]["temperature"],
        normalization_method=config["intrinsic_reward"]["normalization_method"],
        update_rate=config["intrinsic_reward"]["update_rate"]
    )


