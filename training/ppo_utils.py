"""
PPO工具函数
功能：提供PPO训练相关的工具函数和辅助类，支持并行处理
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import partial


class PPOMetrics:
    """PPO指标计算器"""
    
    @staticmethod
    def compute_kl_divergence(old_log_probs: torch.Tensor, 
                            new_log_probs: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算KL散度
        
        Args:
            old_log_probs: 旧策略的对数概率
            new_log_probs: 新策略的对数概率
            attention_mask: 注意力掩码
            
        Returns:
            KL散度
        """
        kl_div = old_log_probs - new_log_probs
        
        if attention_mask is not None:
            kl_div = kl_div * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # 避免除零错误
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            kl_div = kl_div.sum(dim=-1) / mask_sum
        else:
            kl_div = kl_div.mean(dim=-1)
        
        return kl_div
    
    @staticmethod
    def compute_entropy(log_probs: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算熵
        
        Args:
            log_probs: 对数概率
            attention_mask: 注意力掩码
            
        Returns:
            熵
        """
        entropy = -log_probs
        
        if attention_mask is not None:
            entropy = entropy * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # 避免除零错误
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            entropy = entropy.sum(dim=-1) / mask_sum
        else:
            entropy = entropy.mean(dim=-1)
        
        return entropy
    
    @staticmethod
    def compute_advantages(rewards: torch.Tensor,
                          values: torch.Tensor,
                          gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> torch.Tensor:
        """
        计算GAE优势函数
        
        Args:
            rewards: 奖励
            values: 价值函数
            gamma: 折扣因子
            lambda_gae: GAE参数
            
        Returns:
            优势函数
        """
        batch_size, seq_len = rewards.shape
        
        # 计算TD误差
        td_errors = torch.zeros_like(rewards)
        for t in range(seq_len - 1):
            td_errors[:, t] = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        
        # 计算GAE优势
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size)
        
        for t in reversed(range(seq_len)):
            delta = td_errors[:, t]
            gae = delta + gamma * lambda_gae * gae
            advantages[:, t] = gae
        
        return advantages


class PPOLossCalculator:
    """PPO损失计算器"""
    
    def __init__(self, clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.1,
                 entropy_coef: float = 0.01,
                 kl_coef: float = 0.05):
        """
        初始化损失计算器
        
        Args:
            clip_ratio: 裁剪比例
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数
            kl_coef: KL散度系数
        """
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
    
    def compute_policy_loss(self, advantages: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           new_log_probs: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算策略损失
        
        Args:
            advantages: 优势函数
            old_log_probs: 旧策略对数概率
            new_log_probs: 新策略对数概率
            attention_mask: 注意力掩码
            
        Returns:
            策略损失
        """
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 裁剪概率比
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # 计算损失
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        
        if attention_mask is not None:
            policy_loss = policy_loss * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # 避免除零错误
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            policy_loss = policy_loss.sum(dim=-1) / mask_sum
        else:
            policy_loss = policy_loss.mean(dim=-1)
        
        return policy_loss.mean()
    
    def compute_value_loss(self, returns: torch.Tensor,
                          values: torch.Tensor,
                          old_values: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算价值损失
        
        Args:
            returns: 回报
            values: 当前价值函数
            old_values: 旧价值函数
            attention_mask: 注意力掩码
            
        Returns:
            价值损失
        """
        # 计算价值损失
        value_loss = F.mse_loss(values, returns, reduction='none')
        
        if attention_mask is not None:
            value_loss = value_loss * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # 避免除零错误
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            value_loss = value_loss.sum(dim=-1) / mask_sum
        else:
            value_loss = value_loss.mean(dim=-1)
        
        return value_loss.mean()
    
    def compute_entropy_loss(self, log_probs: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算熵损失
        
        Args:
            log_probs: 对数概率
            attention_mask: 注意力掩码
            
        Returns:
            熵损失
        """
        entropy = -log_probs
        
        if attention_mask is not None:
            entropy = entropy * attention_mask.float()
            mask_sum = attention_mask.float().sum(dim=-1)
            # 避免除零错误
            mask_sum = torch.clamp(mask_sum, min=1e-8)
            entropy = entropy.sum(dim=-1) / mask_sum
        else:
            entropy = entropy.mean(dim=-1)
        
        return -entropy.mean()  # 负熵，鼓励探索
    
    def compute_total_loss(self, policy_loss: torch.Tensor,
                          value_loss: torch.Tensor,
                          entropy_loss: torch.Tensor,
                          kl_div: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy_loss: 熵损失
            kl_div: KL散度（可选）
            
        Returns:
            总损失
        """
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        if kl_div is not None:
            total_loss += self.kl_coef * kl_div.mean()
        
        return total_loss


class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, buffer_size: int = 10000):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.attention_masks = []
        
        self.ptr = 0
        self.size = 0
    
    def add(self, observation: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            value: torch.Tensor,
            log_prob: torch.Tensor,
            advantage: torch.Tensor,
            return_val: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None):
        """
        添加经验到缓冲区
        
        Args:
            observation: 观察
            action: 动作
            reward: 奖励
            value: 价值
            log_prob: 对数概率
            advantage: 优势
            return_val: 回报
            attention_mask: 注意力掩码
        """
        if self.size < self.buffer_size:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.advantages.append(advantage)
            self.returns.append(return_val)
            self.attention_masks.append(attention_mask)
            self.size += 1
        else:
            self.observations[self.ptr] = observation
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.advantages[self.ptr] = advantage
            self.returns[self.ptr] = return_val
            self.attention_masks[self.ptr] = attention_mask
            self.ptr = (self.ptr + 1) % self.buffer_size
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        从缓冲区采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次数据
        """
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        batch = {
            "observations": torch.stack([self.observations[i] for i in indices]),
            "actions": torch.stack([self.actions[i] for i in indices]),
            "rewards": torch.stack([self.rewards[i] for i in indices]),
            "values": torch.stack([self.values[i] for i in indices]),
            "log_probs": torch.stack([self.log_probs[i] for i in indices]),
            "advantages": torch.stack([self.advantages[i] for i in indices]),
            "returns": torch.stack([self.returns[i] for i in indices]),
        }
        
        if self.attention_masks[0] is not None:
            batch["attention_masks"] = torch.stack([self.attention_masks[i] for i in indices])
        
        return batch
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.size >= self.buffer_size
    
    def get_size(self) -> int:
        """获取缓冲区大小"""
        return self.size


class PPOScheduler:
    """PPO学习率调度器"""
    
    def __init__(self, initial_lr: float = 1e-5,
                 final_lr: float = 1e-6,
                 total_steps: int = 1000,
                 warmup_steps: int = 100):
        """
        初始化调度器
        
        Args:
            initial_lr: 初始学习率
            final_lr: 最终学习率
            total_steps: 总步数
            warmup_steps: 预热步数
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        """
        获取当前步数的学习率
        
        Args:
            step: 当前步数
            
        Returns:
            学习率
        """
        if step < self.warmup_steps:
            # 预热阶段
            return self.initial_lr * (step / self.warmup_steps)
        else:
            # 衰减阶段
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.initial_lr * (1 - progress) + self.final_lr * progress


def create_ppo_optimizer(model: torch.nn.Module, 
                        learning_rate: float = 1e-5,
                        weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    创建PPO优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        
    Returns:
        优化器
    """
    # 分离可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    return optimizer


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    计算梯度范数
    
    Args:
        model: 模型
        
    Returns:
        梯度范数
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    return total_norm


def clip_grad_norm(model: torch.nn.Module, max_norm: float = 1.0):
    """
    裁剪梯度范数
    
    Args:
        model: 模型
        max_norm: 最大范数
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class ParallelRewardProcessor:
    """并行奖励处理器"""
    
    def __init__(self, num_workers: int = 4, use_threads: bool = True):
        """
        初始化并行处理器
        
        Args:
            num_workers: 工作进程/线程数
            use_threads: 是否使用线程池（True）还是进程池（False）
        """
        self.num_workers = num_workers
        self.use_threads = use_threads
        self.executor = None
        
    def __enter__(self):
        """上下文管理器入口"""
        if self.use_threads:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def compute_rewards_parallel(self, questions: List[str], 
                               student_responses: List[str],
                               reward_func,
                               **kwargs) -> List[float]:
        """
        并行计算奖励
        
        Args:
            questions: 问题列表
            student_responses: 学生响应列表
            reward_func: 奖励计算函数
            **kwargs: 传递给奖励函数的额外参数
            
        Returns:
            奖励列表
        """
        if not self.executor:
            raise RuntimeError("ParallelRewardProcessor must be used as context manager")
        
        # 创建任务
        tasks = []
        for question, response in zip(questions, student_responses):
            task = self.executor.submit(reward_func, question, response, **kwargs)
            tasks.append(task)
        
        # 收集结果
        rewards = []
        for task in as_completed(tasks):
            try:
                reward = task.result()
                rewards.append(reward)
            except Exception as e:
                logging.error(f"奖励计算失败: {e}")
                rewards.append(0.0)
        
        return rewards


class ParallelModelInference:
    """并行模型推理器"""
    
    def __init__(self, model, batch_size: int = 8, num_workers: int = 4):
        """
        初始化并行推理器
        
        Args:
            model: 模型实例
            batch_size: 批次大小
            num_workers: 工作线程数
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._lock = threading.Lock()
    
    def generate_batch_parallel(self, prompts: List[str], 
                              max_length: int = 256,
                              temperature: float = 0.7,
                              **kwargs) -> List[str]:
        """
        并行生成批次文本
        
        Args:
            prompts: 提示文本列表
            max_length: 最大长度
            temperature: 温度参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本列表
        """
        # 将输入分批
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交批次任务
            futures = []
            for batch in batches:
                future = executor.submit(self._generate_single_batch, batch, max_length, temperature, **kwargs)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"批次生成失败: {e}")
                    # 添加空结果作为占位符
                    results.extend([""] * len(batch))
        
        return results
    
    def _generate_single_batch(self, batch_prompts: List[str], 
                             max_length: int, temperature: float, **kwargs) -> List[str]:
        """
        生成单个批次
        
        Args:
            batch_prompts: 批次提示
            max_length: 最大长度
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            批次结果
        """
        with self._lock:  # 确保模型访问的线程安全
            try:
                return self.model.generate(
                    batch_prompts,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs
                )
            except Exception as e:
                logging.error(f"单批次生成失败: {e}")
                return [""] * len(batch_prompts)
    
    def get_logits_batch_parallel(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        并行获取批次logits
        
        Args:
            sequences: 输入序列列表
            
        Returns:
            logits张量列表
        """
        # 将输入分批
        batches = [sequences[i:i + self.batch_size] for i in range(0, len(sequences), self.batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交批次任务
            futures = []
            for batch in batches:
                future = executor.submit(self._get_logits_single_batch, batch)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"批次logits获取失败: {e}")
                    # 添加None作为占位符
                    results.extend([None] * len(batch))
        
        return results
    
    def _get_logits_single_batch(self, batch_sequences: List[str]) -> List[torch.Tensor]:
        """
        获取单个批次的logits
        
        Args:
            batch_sequences: 批次序列
            
        Returns:
            批次logits列表
        """
        with self._lock:  # 确保模型访问的线程安全
            try:
                # 教师模型的get_logits方法返回单个张量，需要拆分
                batch_logits = self.model.get_logits(batch_sequences)
                # 如果返回的是单个张量，需要按批次拆分
                if isinstance(batch_logits, torch.Tensor) and batch_logits.dim() == 3:
                    # 按批次大小拆分
                    batch_size = len(batch_sequences)
                    return [batch_logits[i:i+1] for i in range(batch_size)]
                else:
                    # 如果已经是列表，直接返回
                    return batch_logits if isinstance(batch_logits, list) else [batch_logits]
            except Exception as e:
                logging.error(f"单批次logits获取失败: {e}")
                return [None] * len(batch_sequences)


class AsyncCacheManager:
    """异步缓存管理器"""
    
    def __init__(self, cache_manager, max_queue_size: int = 1000):
        """
        初始化异步缓存管理器
        
        Args:
            cache_manager: 原始缓存管理器
            max_queue_size: 最大队列大小
        """
        self.cache_manager = cache_manager
        self.max_queue_size = max_queue_size
        self._queue = []
        self._lock = threading.Lock()
        self._worker_thread = None
        self._stop_event = threading.Event()
        
    def start_async_worker(self):
        """启动异步工作线程"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._async_worker, daemon=True)
            self._worker_thread.start()
    
    def stop_async_worker(self):
        """停止异步工作线程"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join()
    
    def _async_worker(self):
        """异步工作线程"""
        while not self._stop_event.is_set():
            # 获取待处理的项目
            items_to_process = []
            with self._lock:
                if self._queue:
                    # 批量处理，提高效率
                    batch_size = min(10, len(self._queue))
                    items_to_process = self._queue[:batch_size]
                    self._queue = self._queue[batch_size:]
            
            # 在锁外处理，避免长时间持有锁
            for key, value in items_to_process:
                try:
                    self.cache_manager.put(key, value)
                except Exception as e:
                    logging.error(f"异步缓存更新失败: {e}")
            
            # 短暂休眠避免过度占用CPU
            threading.Event().wait(0.01)
    
    def put_async(self, key: str, value: torch.Tensor):
        """
        异步放入缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            if len(self._queue) < self.max_queue_size:
                self._queue.append((key, value))
            else:
                # 队列满时，移除最旧的项目
                self._queue.pop(0)
                self._queue.append((key, value))
        
        # 确保工作线程在运行
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.start_async_worker()
    
    def get(self, key: str):
        """获取缓存值"""
        return self.cache_manager.get(key)


class ParallelDataLoader:
    """并行数据加载器"""
    
    def __init__(self, dataset, batch_size: int = 8, num_workers: int = 4, shuffle: bool = True):
        """
        初始化并行数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            num_workers: 工作进程数
            shuffle: 是否打乱数据
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """迭代器"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # 分批处理
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield batch_data
    
    def __len__(self):
        """返回批次数量"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_parallel_processor(config: Dict) -> ParallelRewardProcessor:
    """
    创建并行处理器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        并行处理器实例
    """
    num_workers = config.get("parallel", {}).get("num_workers", 4)
    use_threads = config.get("parallel", {}).get("use_threads", True)
    
    return ParallelRewardProcessor(
        num_workers=num_workers,
        use_threads=use_threads
    )


def create_parallel_inference(model, config: Dict) -> ParallelModelInference:
    """
    创建并行推理器的便捷函数
    
    Args:
        model: 模型实例
        config: 配置字典
        
    Returns:
        并行推理器实例
    """
    batch_size = config.get("parallel", {}).get("inference_batch_size", 8)
    num_workers = config.get("parallel", {}).get("num_workers", 4)
    
    return ParallelModelInference(
        model=model,
        batch_size=batch_size,
        num_workers=num_workers
    )


def create_async_cache_manager(cache_manager, config: Dict) -> AsyncCacheManager:
    """
    创建异步缓存管理器的便捷函数
    
    Args:
        cache_manager: 原始缓存管理器
        config: 配置字典
        
    Returns:
        异步缓存管理器实例
    """
    max_queue_size = config.get("parallel", {}).get("cache_queue_size", 1000)
    
    return AsyncCacheManager(
        cache_manager=cache_manager,
        max_queue_size=max_queue_size
    )


