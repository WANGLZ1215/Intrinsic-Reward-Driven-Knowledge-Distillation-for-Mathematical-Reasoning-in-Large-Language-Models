"""
缓存管理模块
功能：管理教师模型logits缓存，提高训练效率
"""

import torch
import pickle
import hashlib
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import time
import json


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_cache_size: int = 10000, 
                 eviction_policy: str = "LRU",
                 cache_file: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            max_cache_size: 最大缓存大小
            eviction_policy: 淘汰策略 (LRU, LFU, FIFO)
            FIFO (First-In, First-Out)—— 先进先出
            LRU (Least Recently Used) —— 最近最少使用
            LFU (Least Frequently Used) —— 最少使用次数
            cache_file: 缓存文件路径
        """
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy
        self.cache_file = cache_file or "./cache/teacher_cache.pkl"
        
        # 初始化缓存
        self.cache = OrderedDict()
        self.access_count = {}  # 访问计数（用于LFU）
        self.access_time = {}   # 访问时间（用于LRU）
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        
        # 加载现有缓存
        self._load_cache()
        
        logging.info(f"缓存管理器初始化完成: size={max_cache_size}, policy={eviction_policy}")
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_access_info(self, key: str):
        """更新访问信息"""
        current_time = time.time()
        self.access_time[key] = current_time
        
        if key in self.access_count:
            self.access_count[key] += 1
        else:
            self.access_count[key] = 1
    
    def _evict_item(self):
        """根据策略淘汰项目"""
        if not self.cache:
            return
        
        if self.eviction_policy == "LRU":
            # 移除最久未使用的项目
            oldest_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            self._remove_item(oldest_key)
            
        elif self.eviction_policy == "LFU":
            # 移除最少使用的项目
            least_frequent_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self._remove_item(least_frequent_key)
            
        elif self.eviction_policy == "FIFO":
            # 移除最早加入的项目
            oldest_key = next(iter(self.cache))
            self._remove_item(oldest_key)
    
    def _remove_item(self, key: str):
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
            del self.access_count[key]
            del self.access_time[key]
            self.cache_evictions += 1
    
    def _load_cache(self):
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get("cache", {}))
                self.access_count = cache_data.get("access_count", {})
                self.access_time = cache_data.get("access_time", {})
                self.cache_hits = cache_data.get("cache_hits", 0)
                self.cache_misses = cache_data.get("cache_misses", 0)
                self.cache_evictions = cache_data.get("cache_evictions", 0)
                
                logging.info(f"缓存已加载: {len(self.cache)} 项")
                
            except Exception as e:
                logging.warning(f"缓存加载失败: {e}")
                self.cache = OrderedDict()
    
    def _save_cache(self):
        """保存缓存"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                "cache": dict(self.cache),
                "access_count": self.access_count,
                "access_time": self.access_time,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_evictions": self.cache_evictions
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logging.info(f"缓存已保存: {len(self.cache)} 项")
            
        except Exception as e:
            logging.error(f"缓存保存失败: {e}")
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """
        获取缓存的logits
        
        Args:
            text: 输入文本
            
        Returns:
            缓存的logits或None
        """
        key = self._get_cache_key(text)
        
        if key in self.cache:
            self.cache_hits += 1
            self._update_access_info(key)
            
            # 移动到末尾（LRU）
            self.cache.move_to_end(key)
            
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def put(self, text: str, logits: torch.Tensor):
        """
        存储logits到缓存
        
        Args:
            text: 输入文本
            logits: 要缓存的logits
        """
        key = self._get_cache_key(text)
        
        # 如果缓存已满，先淘汰一些项目
        while len(self.cache) >= self.max_cache_size:
            self._evict_item()
        
        # 存储到缓存
        self.cache[key] = logits.clone().detach().cpu()  # 移动到CPU以节省GPU内存
        self._update_access_info(key)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        logging.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_evictions": self.cache_evictions,
            "eviction_policy": self.eviction_policy
        }
    
    def save_stats(self, filepath: str):
        """保存统计信息"""
        stats = self.get_stats()
        stats["timestamp"] = time.time()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def cleanup(self):
        """清理资源"""
        self._save_cache()
        logging.info("缓存管理器清理完成")
    
    def save_cache(self, filepath: str):
        """公开方法：保存缓存到指定路径"""
        # 临时保存原路径
        original_cache_file = self.cache_file
        # 使用新路径
        self.cache_file = filepath
        # 保存缓存
        self._save_cache()
        # 恢复原路径
        self.cache_file = original_cache_file
    
    def load_cache(self, filepath: str):
        """公开方法：从指定路径加载缓存"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get("cache", {}))
                self.access_count = cache_data.get("access_count", {})
                self.access_time = cache_data.get("access_time", {})
                self.cache_hits = cache_data.get("cache_hits", 0)
                self.cache_misses = cache_data.get("cache_misses", 0)
                self.cache_evictions = cache_data.get("cache_evictions", 0)
                
                logging.info(f"缓存已从 {filepath} 加载: {len(self.cache)} 项")
                
            except Exception as e:
                logging.warning(f"从 {filepath} 加载缓存失败: {e}")
                self.cache = OrderedDict()
        else:
            logging.warning(f"缓存文件不存在: {filepath}")


class BatchCacheManager:
    """批处理缓存管理器"""
    
    def __init__(self, cache_manager: CacheManager, batch_size: int = 8):
        """
        初始化批处理缓存管理器
        
        Args:
            cache_manager: 基础缓存管理器
            batch_size: 批处理大小
        """
        self.cache_manager = cache_manager
        self.batch_size = batch_size
    
    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[torch.Tensor]], List[str]]:
        """
        批量获取缓存
        
        Args:
            texts: 文本列表
            
        Returns:
            (缓存的logits列表, 未缓存的文本列表)
        """
        cached_logits = []
        uncached_texts = []
        
        for text in texts:
            logits = self.cache_manager.get(text)
            if logits is not None:
                cached_logits.append(logits.to('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                cached_logits.append(None)
                uncached_texts.append(text)
        
        return cached_logits, uncached_texts
    
    def put_batch(self, texts: List[str], logits_list: List[torch.Tensor]):
        """
        批量存储缓存
        
        Args:
            texts: 文本列表
            logits_list: logits列表
        """
        for text, logits in zip(texts, logits_list):
            self.cache_manager.put(text, logits)


class CacheMonitor:
    """缓存监控器"""
    
    def __init__(self, cache_manager: CacheManager, log_interval: int = 100):
        """
        初始化监控器
        
        Args:
            cache_manager: 缓存管理器
            log_interval: 日志间隔
        """
        self.cache_manager = cache_manager
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.request_count = 0
    
    def log_request(self):
        """记录请求"""
        self.request_count += 1
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            stats = self.cache_manager.get_stats()
            logging.info(f"缓存统计: {stats}")
            self.last_log_time = current_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        stats = self.cache_manager.get_stats()
        current_time = time.time()
        
        return {
            **stats,
            "request_count": self.request_count,
            "uptime": current_time - self.last_log_time,
            "requests_per_second": self.request_count / max(1, current_time - self.last_log_time)
        }


def create_cache_manager(config: Dict) -> CacheManager:
    """
    创建缓存管理器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        缓存管理器实例
    """
    return CacheManager(
        max_cache_size=config["teacher_model"]["cache_size"],
        eviction_policy=config["teacher_model"]["cache_policy"],
        cache_file=config.get("cache_file", "./cache/teacher_cache.pkl")
    )


