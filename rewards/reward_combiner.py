"""
Reward Combination Module
Function: Combine intrinsic and external rewards, implement multi-signal fusion
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from collections import deque


class RewardCombiner:
    """Reward Combiner"""
    
    def __init__(self, lambda_intrinsic: float = 0.7,
                 lambda_correctness: float = 0.3,
                 lambda_reasoning: float = 0.0,
                 lambda_format: float = 0.0,
                 use_adaptive_weights: bool = True,
                 adaptation_rate: float = 0.01):
        """
        初始化奖励组合器
        
        Args:
            lambda_intrinsic: 内在奖励权重
            lambda_correctness: 答案正确性权重
            lambda_reasoning: 推理过程权重
            lambda_format: 格式约束权重
            use_adaptive_weights: 是否使用自适应权重
            adaptation_rate: 权重适应率
        """
        self.lambda_intrinsic = lambda_intrinsic
        self.lambda_correctness = lambda_correctness
        self.lambda_reasoning = lambda_reasoning
        self.lambda_format = lambda_format
        self.use_adaptive_weights = use_adaptive_weights
        self.adaptation_rate = adaptation_rate
        
        # 统计信息（先初始化，避免save_weights等方法访问未初始化的属性）
        self.combination_stats = {
            "total_combinations": 0,
            "intrinsic_mean": 0.0,
            "correctness_mean": 0.0,
            "combined_mean": 0.0,
            "weight_history": []
        }
        
        # 自适应权重
        if use_adaptive_weights:
            self.adaptive_weights = {
                "intrinsic": lambda_intrinsic,
                "correctness": lambda_correctness,
                "reasoning": lambda_reasoning,
                "format": lambda_format
            }
            
            # 权重性能跟踪
            self.weight_performance = {key: 0.0 for key in self.adaptive_weights.keys()}
            self.performance_window = deque(maxlen=100)
    
    def save_weights(self, filepath: str):
        """保存自适应权重"""
        if self.use_adaptive_weights:
            import json
            weight_data = {
                "adaptive_weights": self.adaptive_weights,
                "weight_performance": self.weight_performance,
                "combination_stats": self.combination_stats
            }
            with open(filepath, 'w') as f:
                json.dump(weight_data, f, indent=2)
            logging.info(f"自适应权重已保存到: {filepath}")
    
    def load_weights(self, filepath: str):
        """加载自适应权重"""
        if self.use_adaptive_weights:
            import json
            try:
                with open(filepath, 'r') as f:
                    weight_data = json.load(f)
                
                self.adaptive_weights = weight_data["adaptive_weights"]
                self.weight_performance = weight_data["weight_performance"]
                self.combination_stats = weight_data["combination_stats"]
                
                logging.info(f"自适应权重已从 {filepath} 加载")
            except Exception as e:
                logging.warning(f"加载自适应权重失败: {e}")
        
        # 统计信息已在前面初始化，这里不需要重复初始化
        
        logging.info(f"Reward combiner initialized: λ_intrinsic={self.lambda_intrinsic}, λ_correctness={self.lambda_correctness}")
    
    def combine_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor,
                       reasoning_rewards: Optional[torch.Tensor] = None,
                       format_rewards: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        组合多种奖励信号
        
        Args:
            intrinsic_rewards: 内在奖励
            correctness_rewards: 答案正确性奖励
            reasoning_rewards: 推理过程奖励（可选）
            format_rewards: 格式约束奖励（可选）
            
        Returns:
            组合后的奖励
        """
        # 获取当前权重
        if self.use_adaptive_weights:
            weights = self.adaptive_weights.copy()
        else:
            weights = {
                "intrinsic": self.lambda_intrinsic,
                "correctness": self.lambda_correctness,
                "reasoning": self.lambda_reasoning,
                "format": self.lambda_format
            }
        
        # 组合奖励
        combined_rewards = (
            weights["intrinsic"] * intrinsic_rewards +
            weights["correctness"] * correctness_rewards
        )
        
        if reasoning_rewards is not None:
            combined_rewards += weights["reasoning"] * reasoning_rewards
        
        if format_rewards is not None:
            combined_rewards += weights["format"] * format_rewards
        
        # 更新统计信息
        self._update_statistics(intrinsic_rewards, correctness_rewards, combined_rewards, weights)
        
        # 自适应权重更新
        if self.use_adaptive_weights:
            self._adapt_weights(intrinsic_rewards, correctness_rewards, combined_rewards)
        
        return combined_rewards
    
    def combine_batch_rewards(self, batch_rewards: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        批量组合奖励
        
        Args:
            batch_rewards: 批量奖励字典
            
        Returns:
            组合后的奖励列表
        """
        batch_size = len(batch_rewards["intrinsic"])
        combined_batch = []
        
        for i in range(batch_size):
            intrinsic = batch_rewards["intrinsic"][i]
            correctness = batch_rewards["correctness"][i]
            reasoning = batch_rewards.get("reasoning", [None] * batch_size)[i]
            format_reward = batch_rewards.get("format", [None] * batch_size)[i]
            
            combined = self.combine_rewards(intrinsic, correctness, reasoning, format_reward)
            combined_batch.append(combined)
        
        return combined_batch
    
    def _update_statistics(self, intrinsic_rewards: torch.Tensor,
                          correctness_rewards: torch.Tensor,
                          combined_rewards: torch.Tensor,
                          weights: Dict[str, float]):
        """更新统计信息"""
        # 确保combination_stats已初始化
        if not hasattr(self, 'combination_stats'):
            self.combination_stats = {
                "total_combinations": 0,
                "intrinsic_mean": 0.0,
                "correctness_mean": 0.0,
                "combined_mean": 0.0,
                "weight_history": []
            }
        
        self.combination_stats["total_combinations"] += 1
        
        # 计算均值
        intrinsic_mean = torch.mean(intrinsic_rewards).item()
        correctness_mean = torch.mean(correctness_rewards).item()
        combined_mean = torch.mean(combined_rewards).item()
        
        # 更新统计（指数移动平均）
        alpha = 0.01
        self.combination_stats["intrinsic_mean"] = (
            (1 - alpha) * self.combination_stats["intrinsic_mean"] + 
            alpha * intrinsic_mean
        )
        self.combination_stats["correctness_mean"] = (
            (1 - alpha) * self.combination_stats["correctness_mean"] + 
            alpha * correctness_mean
        )
        self.combination_stats["combined_mean"] = (
            (1 - alpha) * self.combination_stats["combined_mean"] + 
            alpha * combined_mean
        )
        
        # 记录权重历史
        self.combination_stats["weight_history"].append(weights.copy())
        if len(self.combination_stats["weight_history"]) > 1000:
            self.combination_stats["weight_history"] = self.combination_stats["weight_history"][-500:]
    
    def _adapt_weights(self, intrinsic_rewards: torch.Tensor,
                      correctness_rewards: torch.Tensor,
                      combined_rewards: torch.Tensor):
        """自适应调整权重"""
        # 计算各奖励信号的贡献度
        intrinsic_contribution = self._calculate_contribution(intrinsic_rewards, combined_rewards)
        correctness_contribution = self._calculate_contribution(correctness_rewards, combined_rewards)
        
        # 更新性能分数
        self.weight_performance["intrinsic"] = intrinsic_contribution
        self.weight_performance["correctness"] = correctness_contribution
        
        # 记录权重历史
        self.combination_stats["weight_history"].append(self.adaptive_weights.copy())
        
        # 自适应调整权重
        self._update_adaptive_weights()
        
        # 记录权重变化
        logging.info(f"自适应权重更新: 内在={self.adaptive_weights['intrinsic']:.4f}, "
                    f"正确性={self.adaptive_weights['correctness']:.4f}")
    
    def _calculate_contribution(self, individual_rewards: torch.Tensor,
                              combined_rewards: torch.Tensor) -> float:
        """计算个体奖励的贡献度"""
        # 基于相关性计算贡献度
        correlation = torch.corrcoef(torch.stack([
            individual_rewards.flatten(),
            combined_rewards.flatten()
        ]))[0, 1].item()
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _update_adaptive_weights(self):
        """更新自适应权重"""
        # 基于性能分数调整权重
        total_performance = sum(self.weight_performance.values())
        
        if total_performance > 0:
            for key in self.adaptive_weights:
                # 计算新的权重比例
                # 避免除零错误
                if total_performance < 1e-8:
                    new_ratio = 1.0 / len(self.weight_performance)  # 平均分配
                else:
                    new_ratio = self.weight_performance[key] / total_performance
                
                # 平滑更新权重
                self.adaptive_weights[key] = (
                    (1 - self.adaptation_rate) * self.adaptive_weights[key] +
                    self.adaptation_rate * new_ratio
                )
            
            # 归一化权重，避免除零错误
            total_weight = sum(self.adaptive_weights.values())
            if total_weight > 1e-8:
                for key in self.adaptive_weights:
                    self.adaptive_weights[key] /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        if self.use_adaptive_weights:
            return self.adaptive_weights.copy()
        else:
            return {
                "intrinsic": self.lambda_intrinsic,
                "correctness": self.lambda_correctness,
                "reasoning": self.lambda_reasoning,
                "format": self.lambda_format
            }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        # 确保combination_stats已初始化
        if not hasattr(self, 'combination_stats'):
            self.combination_stats = {
                "total_combinations": 0,
                "intrinsic_mean": 0.0,
                "correctness_mean": 0.0,
                "combined_mean": 0.0,
                "weight_history": []
            }
        
        stats = self.combination_stats.copy()
        stats["current_weights"] = self.get_current_weights()
        
        if self.use_adaptive_weights:
            stats["adaptive_weights"] = self.adaptive_weights.copy()
            stats["weight_performance"] = self.weight_performance.copy()
            
            # 计算权重变化统计
            weight_history = self.combination_stats["weight_history"]
            if len(weight_history) > 1:
                recent_weights = weight_history[-10:]  # 取最近10个
                stats["weight_variance"] = {
                    "intrinsic": float(np.var([w["intrinsic"] for w in recent_weights])),
                    "correctness": float(np.var([w["correctness"] for w in recent_weights]))
                }
                
                # 计算权重趋势（需要至少5个历史记录才能计算）
                if len(recent_weights) >= 5:
                    first_half = recent_weights[:len(recent_weights)//2] if len(recent_weights) >= 10 else []
                    second_half = recent_weights[-len(recent_weights)//2:] if len(recent_weights) >= 10 else recent_weights[-5:]
                    
                    if len(first_half) > 0 and len(second_half) > 0:
                        stats["weight_trend"] = {
                            "intrinsic": float(np.mean([w["intrinsic"] for w in second_half]) - 
                                               np.mean([w["intrinsic"] for w in first_half])),
                            "correctness": float(np.mean([w["correctness"] for w in second_half]) - 
                                                np.mean([w["correctness"] for w in first_half]))
                        }
                    else:
                        # 如果记录太少，使用简单的差值
                        stats["weight_trend"] = {
                            "intrinsic": float(recent_weights[-1]["intrinsic"] - recent_weights[0]["intrinsic"]),
                            "correctness": float(recent_weights[-1]["correctness"] - recent_weights[0]["correctness"])
                        }
                else:
                    # 记录太少，无法计算趋势
                    stats["weight_trend"] = {
                        "intrinsic": 0.0,
                        "correctness": 0.0
                    }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.combination_stats = {
            "total_combinations": 0,
            "intrinsic_mean": 0.0,
            "correctness_mean": 0.0,
            "combined_mean": 0.0,
            "weight_history": []
        }
        
        if self.use_adaptive_weights:
            self.weight_performance = {key: 0.0 for key in self.adaptive_weights.keys()}
            self.performance_window.clear()


class RewardBalancer:
    """奖励平衡器"""
    
    def __init__(self, target_balance: float = 0.5,
                 balance_threshold: float = 0.1,
                 adjustment_rate: float = 0.01):
        """
        初始化奖励平衡器
        
        Args:
            target_balance: 目标平衡比例
            balance_threshold: 平衡阈值
            adjustment_rate: 调整率
        """
        self.target_balance = target_balance
        self.balance_threshold = balance_threshold
        self.adjustment_rate = adjustment_rate
        
        # 平衡统计
        self.balance_stats = {
            "intrinsic_ratio": 0.5,
            "correctness_ratio": 0.5,
            "balance_score": 0.0
        }
    
    def balance_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        平衡两种奖励信号
        
        Args:
            intrinsic_rewards: 内在奖励
            correctness_rewards: 答案正确性奖励
            
        Returns:
            平衡后的奖励对
        """
        # 计算当前比例
        intrinsic_mean = torch.mean(torch.abs(intrinsic_rewards)).item()
        correctness_mean = torch.mean(torch.abs(correctness_rewards)).item()
        
        total_mean = intrinsic_mean + correctness_mean
        
        # 避免除零错误
        if abs(total_mean) > 1e-8:
            current_intrinsic_ratio = intrinsic_mean / total_mean
            current_correctness_ratio = correctness_mean / total_mean
        else:
            current_intrinsic_ratio = 0.5  # 默认平均分配
            current_correctness_ratio = 0.5
        
        # 计算平衡分数
        balance_score = 1.0 - abs(current_intrinsic_ratio - self.target_balance)
        self.balance_stats["balance_score"] = balance_score
        
        # 如果不平衡，进行调整
        if balance_score < (1.0 - self.balance_threshold):
            intrinsic_rewards, correctness_rewards = self._adjust_rewards(
                intrinsic_rewards, correctness_rewards,
                current_intrinsic_ratio, current_correctness_ratio
            )
        
        # 更新统计
        self.balance_stats["intrinsic_ratio"] = current_intrinsic_ratio
        self.balance_stats["correctness_ratio"] = current_correctness_ratio
        
        return intrinsic_rewards, correctness_rewards
    
    def _adjust_rewards(self, intrinsic_rewards: torch.Tensor,
                       correctness_rewards: torch.Tensor,
                       current_intrinsic_ratio: float,
                       current_correctness_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """调整奖励以改善平衡"""
        # 计算调整因子
        if current_intrinsic_ratio > self.target_balance:
            # 内在奖励过强，需要减弱
            intrinsic_factor = 1.0 - self.adjustment_rate
            correctness_factor = 1.0 + self.adjustment_rate
        else:
            # 答案奖励过强，需要减弱
            intrinsic_factor = 1.0 + self.adjustment_rate
            correctness_factor = 1.0 - self.adjustment_rate
        
        # 应用调整
        adjusted_intrinsic = intrinsic_rewards * intrinsic_factor
        adjusted_correctness = correctness_rewards * correctness_factor
        
        return adjusted_intrinsic, adjusted_correctness
    
    def get_balance_statistics(self) -> Dict[str, float]:
        """获取平衡统计信息"""
        return self.balance_stats.copy()


def create_reward_combiner(config: Dict) -> RewardCombiner:
    """
    创建奖励组合器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        奖励组合器实例
    """
    return RewardCombiner(
        lambda_intrinsic=config["reward_combination"]["lambda_intrinsic"],
        lambda_correctness=config["reward_combination"]["lambda_correctness"],
        lambda_reasoning=config.get("lambda_reasoning", 0.0),
        lambda_format=config.get("lambda_format", 0.0),
        use_adaptive_weights=config.get("use_adaptive_weights", False),
        adaptation_rate=config.get("adaptation_rate", 0.01)
    )





