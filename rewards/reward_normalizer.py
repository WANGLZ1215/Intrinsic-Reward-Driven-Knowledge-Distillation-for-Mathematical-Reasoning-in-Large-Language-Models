"""
奖励归一化模块
功能：对内在奖励和外部奖励进行归一化处理，确保训练稳定性
"""

import torch
import numpy as np
from typing import Optional, Dict


class RewardNormalizer:
    """奖励归一化器"""
    
    def __init__(self, 
                 method: str = "mean_std",
                 clip_min: float = -5.0,
                 clip_max: float = 5.0,
                 epsilon: float = 1e-8,
                 momentum: float = 0.99):
        """
        初始化奖励归一化器
        
        Args:
            method: 归一化方法 ('mean_std', 'min_max', 'z_score', 'robust')
            clip_min: 裁剪最小值
            clip_max: 裁剪最大值
            epsilon: 数值稳定性参数
            momentum: 移动平均动量
        """
        self.method = method
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.momentum = momentum
        
        # 统计信息（使用移动平均）
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_min = 0.0
        self.running_max = 1.0
        
        # 是否已初始化
        self.initialized = False
    
    def normalize_intrinsic_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化内在奖励
        
        Args:
            rewards: 奖励张量 [batch_size] 或 [batch_size, seq_len]
            
        Returns:
            归一化后的奖励
        """
        if self.method == "mean_std":
            return self._normalize_mean_std(rewards)
        elif self.method == "min_max":
            return self._normalize_min_max(rewards)
        elif self.method == "z_score":
            return self._normalize_z_score(rewards)
        elif self.method == "robust":
            return self._normalize_robust(rewards)
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")
    
    def normalize_external_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化外部奖励
        
        Args:
            rewards: 奖励张量
            
        Returns:
            归一化后的奖励
        """
        # 外部奖励通常已经在合理范围内（如0-1），只需简单裁剪
        return torch.clamp(rewards, self.clip_min, self.clip_max)
    
    def _normalize_mean_std(self, rewards: torch.Tensor) -> torch.Tensor:
        """均值-标准差归一化"""
        # 计算当前批次的统计信息
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        
        # 更新移动平均
        if not self.initialized:
            self.running_mean = batch_mean
            self.running_std = max(batch_std, self.epsilon)
            self.initialized = True
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * batch_std
        
        # 归一化
        normalized = (rewards - self.running_mean) / (self.running_std + self.epsilon)
        
        # 裁剪
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def _normalize_min_max(self, rewards: torch.Tensor) -> torch.Tensor:
        """最小-最大归一化"""
        # 计算当前批次的最小最大值
        batch_min = rewards.min().item()
        batch_max = rewards.max().item()
        
        # 更新移动平均
        if not self.initialized:
            self.running_min = batch_min
            self.running_max = batch_max
            self.initialized = True
        else:
            self.running_min = self.momentum * self.running_min + (1 - self.momentum) * batch_min
            self.running_max = self.momentum * self.running_max + (1 - self.momentum) * batch_max
        
        # 归一化到 [0, 1]
        range_val = self.running_max - self.running_min
        if abs(range_val) < self.epsilon:
            normalized = torch.zeros_like(rewards)
        else:
            normalized = (rewards - self.running_min) / (range_val + self.epsilon)
        
        # 缩放到 [clip_min, clip_max]
        normalized = normalized * (self.clip_max - self.clip_min) + self.clip_min
        
        return normalized
    
    def _normalize_z_score(self, rewards: torch.Tensor) -> torch.Tensor:
        """Z-score标准化（与mean_std类似，但不使用移动平均）"""
        mean = rewards.mean()
        std = rewards.std()
        
        normalized = (rewards - mean) / (std + self.epsilon)
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def _normalize_robust(self, rewards: torch.Tensor) -> torch.Tensor:
        """鲁棒归一化（使用中位数和四分位距）"""
        # 转换为numpy进行计算
        rewards_np = rewards.detach().cpu().numpy().flatten()
        
        median = np.median(rewards_np)
        q75 = np.percentile(rewards_np, 75)
        q25 = np.percentile(rewards_np, 25)
        iqr = q75 - q25
        
        # 归一化
        if iqr < self.epsilon:
            normalized = torch.zeros_like(rewards)
        else:
            normalized = (rewards - median) / (iqr + self.epsilon)
        
        # 裁剪
        normalized = torch.clamp(normalized, self.clip_min, self.clip_max)
        
        return normalized
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取当前统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "running_min": self.running_min,
            "running_max": self.running_max,
            "initialized": self.initialized
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.running_mean = 0.0
        self.running_std = 1.0
        self.running_min = 0.0
        self.running_max = 1.0
        self.initialized = False
    
    def load_stats(self, stats: Dict[str, float]):
        """
        加载统计信息
        
        Args:
            stats: 统计信息字典
        """
        self.running_mean = stats.get("running_mean", 0.0)
        self.running_std = stats.get("running_std", 1.0)
        self.running_min = stats.get("running_min", 0.0)
        self.running_max = stats.get("running_max", 1.0)
        self.initialized = stats.get("initialized", False)

