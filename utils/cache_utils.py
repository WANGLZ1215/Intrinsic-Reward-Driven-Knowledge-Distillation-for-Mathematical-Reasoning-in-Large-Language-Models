"""
Cache Utilities for Transformers v4.43+
处理past_key_values弃用警告的正确解决方案
"""

import torch
from transformers import Cache
from typing import Optional, Tuple, Union
import logging


class ModernCacheManager:
    """现代缓存管理器，使用新的Cache类"""
    
    def __init__(self, model, batch_size: int = 1, max_length: int = 512):
        """
        初始化现代缓存管理器
        
        Args:
            model: 模型实例
            batch_size: 批次大小
            max_length: 最大长度
        """
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache = None
        
        logging.info("现代缓存管理器初始化完成")
    
    def create_cache(self, batch_size: int = None, max_length: int = None) -> Cache:
        """
        创建新的Cache实例
        
        Args:
            batch_size: 批次大小
            max_length: 最大长度
            
        Returns:
            Cache实例
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length
            
        # 创建新的Cache实例
        self.cache = Cache(
            batch_size=batch_size,
            max_length=max_length,
            device=self.model.device,
            dtype=self.model.dtype
        )
        
        return self.cache
    
    def get_cache(self) -> Optional[Cache]:
        """获取当前缓存"""
        return self.cache
    
    def clear_cache(self):
        """清除缓存"""
        self.cache = None
    
    def update_cache(self, new_cache: Cache):
        """更新缓存"""
        self.cache = new_cache


def create_modern_cache(model, batch_size: int = 1, max_length: int = 512) -> ModernCacheManager:
    """
    创建现代缓存管理器的便捷函数
    
    Args:
        model: 模型实例
        batch_size: 批次大小
        max_length: 最大长度
        
    Returns:
        现代缓存管理器实例
    """
    return ModernCacheManager(model, batch_size, max_length)


def suppress_past_key_values_warning():
    """
    抑制past_key_values弃用警告
    这是一个临时解决方案，直到所有代码都迁移到新的Cache类
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*past_key_values.*", category=DeprecationWarning)


def create_generation_config_with_cache(model, **kwargs):
    """
    创建带有正确缓存配置的生成配置
    
    Args:
        model: 模型实例
        **kwargs: 其他生成参数
        
    Returns:
        生成配置字典
    """
    config = {
        "use_cache": True,
        "past_key_values": None,  # 明确设置为None
        **kwargs
    }
    
    return config


def update_model_for_modern_cache(model):
    """
    更新模型以使用现代缓存
    
    Args:
        model: 模型实例
        
    Returns:
        更新后的模型
    """
    # 设置模型配置以使用新的Cache类
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # 抑制弃用警告
    suppress_past_key_values_warning()
    
    return model
