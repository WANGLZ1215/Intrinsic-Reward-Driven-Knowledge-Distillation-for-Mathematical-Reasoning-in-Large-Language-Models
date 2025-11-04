"""
强化学习训练器
功能：实现基于内在奖励的PPO训练
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import yaml
import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import wandb
from collections import deque
from tqdm import tqdm
import time

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from models.cache_manager import CacheManager
from rewards.intrinsic_reward import IntrinsicRewardComputer
from rewards.reward_normalizer import RewardNormalizer
from rewards.reward_combiner import RewardCombiner
from data.gsm8k_processor import GSM8KProcessor
from utils.math_utils import extract_final_answer, is_answer_correct
from training.ppo_utils import (
    ParallelRewardProcessor, ParallelModelInference, 
    AsyncCacheManager, ParallelDataLoader,
    create_parallel_processor, create_parallel_inference, create_async_cache_manager,
    compute_grad_norm  # 用于检查梯度
)
import functools
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


def handle_errors(func):
    """错误处理装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取logger实例
            if hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(__name__)
            
            logger.error(f"❌ 函数 {func.__name__} 执行失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 根据错误类型决定是否重新抛出
            if isinstance(e, (ValueError, TypeError, KeyError)):
                raise
            else:
                logger.error(f"❌ 未知错误类型，继续执行")
                return None
    
    return wrapper


def validate_data_batch(batch: Dict) -> bool:
    """验证数据批次"""
    required_keys = ["input_ids", "attention_mask"]
    
    for key in required_keys:
        if key not in batch:
            print(f"❌ 缺少必需的批次键: {key}")
            return False
        
        if not isinstance(batch[key], torch.Tensor):
            print(f"❌ 批次键 {key} 不是张量")
            return False
        
        if batch[key].numel() == 0:
            print(f"❌ 批次键 {key} 为空")
            return False
    
    # 检查批次大小一致性
    batch_size = batch["input_ids"].shape[0]
    for key in required_keys:
        if batch[key].shape[0] != batch_size:
            print(f"❌ 批次大小不一致: {key}")
            return False
    
    return True


def validate_config(config: Dict) -> Dict:
    """验证和补充配置"""
    # 默认配置
    default_config = {
        "model": {
            "teacher_model_name": "Qwen/Qwen2.5-32B-Instruct",
            "student_model_name": "Qwen/Qwen2.5-7B-Instruct",
            "cache_size": 10000,
            "cache_policy": "LRU",
            "use_lora": True
        },
        "device": {
            "device_map": "auto",
            "torch_dtype": "bfloat16"
        },
        "reward": {
            "temperature": 1.0,
            "normalization": "mean_std",
            "lambda_intrinsic": 0.7,
            "lambda_correctness": 0.3,
            "update_rate": 0.01,
            "clip_min": -5.0,
            "clip_max": 5.0,
            "use_adaptive_weights": True,
            "adaptation_rate": 0.01,
            "reasoning_weight": 0.0,
            "format_weight": 0.0
        },
        "ppo": {
            "learning_rate": 1e-5,
            "batch_size": 8,
            "mini_batch_size": 4,
            "ppo_epochs": 4,
            "clip_ratio": 0.2,
            "value_loss_coef": 0.1,
            "entropy_coef": 0.01,
            "kl_coef": 0.05,
            "gamma": 0.99,
            "lambda_gae": 0.95,
            "max_grad_norm": 1.0,
            "max_length": 512,
            "temperature": 0.7,
            "do_sample": True,
            "output_dir": "./checkpoints/rl_model"
        },
        "training": {
            "max_steps": 1000,
            "save_steps": 50,
            "eval_steps": 100,
            "logging_steps": 10
        },
        "parallel": {
            "enabled": True,
            "num_workers": 4,
            "use_threads": True,
            "inference_batch_size": 16,
            "cache_queue_size": 1000,
            "use_parallel_data_loader": True,
            "data_loader_workers": 4
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "intrinsic-reward-distillation",
            "use_tensorboard": True,
            "tensorboard_log_dir": "./logs"
        }
    }
    
    # 递归合并配置
    def merge_config(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_config(base[key], value)
            else:
                base[key] = value
        return base
    
    return merge_config(default_config, config)


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化RL训练器
        
        Args:
            config: 训练配置
        """
        self.config = validate_config(config)  # 验证和补充配置
        
        # 抑制past_key_values警告
        suppress_past_key_values_warning()
        
        self.teacher_model = None
        self.student_model = None
        self.ppo_model = None  # PPO模型（带ValueHead）
        self.ppo_trainer = None
        self.cache_manager = None
        
        # 奖励计算组件
        self.intrinsic_computer = None
        self.reward_normalizer = None
        self.reward_combiner = None
        
        # 数据处理器
        self.data_processor = None
        
        # 并行处理组件
        self.parallel_processor = None
        self.parallel_inference_student = None
        self.parallel_inference_teacher = None
        self.async_cache_manager = None
        self.parallel_data_loader = None
        
        # 训练统计
        self.training_stats = {
            "step": 0,
            "total_rewards": [],
            "intrinsic_rewards": [],
            "correctness_rewards": [],
            "combined_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "kl_divergences": []
        }
        
        # 内存管理 - 根据GPU显存自动调整清理频率
        # 对于H200 140GB等大显存GPU，可以减少清理频率以提升速度
        # 注意：初始化时CUDA可能未就绪，先设置默认值
        self._memory_cleanup_interval = 3
        self._force_cleanup_every_n_steps = 2
        self._last_cleanup_step = 0
        self._vram_detected = False  # 标记是否已检测VRAM
        
        # 性能优化
        self._use_mixed_precision = self.config.get("device", {}).get("use_mixed_precision", True)
        self._gradient_accumulation_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 1)
        self._gradient_accumulation_count = 0
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 缓存并行配置检查结果
        # 注意：默认禁用并行处理，避免tokenizer线程安全问题
        self._use_parallel = self.config.get("parallel", {}).get("enabled", False)
    
    def _create_progress_bar(self, iterable, desc: str, total: int = None, unit: str = "sample"):
        """创建标准化的进度条"""
        return tqdm(
            iterable,
            total=total or len(iterable) if hasattr(iterable, '__len__') else None,
            desc=desc,
            unit=unit,
            ncols=80,
            leave=False
        )
    
    def _cleanup_memory(self, step: int, force: bool = False):
        """清理内存"""
        should_cleanup = force or (step - self._last_cleanup_step >= self._memory_cleanup_interval)
        
        if should_cleanup:
            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理Python垃圾回收
            gc.collect()
            
            # 清理训练统计（保留最近的数据）
            for key in ["total_rewards", "intrinsic_rewards", "correctness_rewards", 
                       "combined_rewards", "policy_losses", "value_losses", "kl_divergences"]:
                if len(self.training_stats[key]) > 500:
                    self.training_stats[key] = self.training_stats[key][-500:]
            
            # 记录缓存统计信息
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                self.logger.info(f"📊 缓存统计(CacheManager): 命中率={cache_stats['hit_rate']:.3f}, "
                               f"大小={cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
            
            # ✅ 修复：Teacher模型内部也有缓存，记录其统计信息
            if self.teacher_model and hasattr(self.teacher_model, 'get_cache_stats'):
                teacher_cache_stats = self.teacher_model.get_cache_stats()
                self.logger.info(f"📊 缓存统计(Teacher内部): 命中率={teacher_cache_stats['hit_rate']:.3f}, "
                               f"大小={teacher_cache_stats['cache_size']}/{teacher_cache_stats['max_cache_size']}")
            
            self._last_cleanup_step = step
            if not force:  # 强制清理时不打印日志，避免日志过多
                self.logger.info(f"🧹 内存清理完成 (step {step})")
    
    def cleanup_resources(self):
        """清理所有资源"""
        try:
            # 清理模型
            if self.teacher_model:
                del self.teacher_model
                self.teacher_model = None
            
            if self.student_model:
                del self.student_model
                self.student_model = None
            
            if self.ppo_trainer:
                del self.ppo_trainer
                self.ppo_trainer = None
            
            # 清理缓存
            if self.cache_manager:
                self.cache_manager.clear()  # 使用正确的方法名
                del self.cache_manager
                self.cache_manager = None
            
            # 清理并行组件
            if self.parallel_processor:
                del self.parallel_processor
                self.parallel_processor = None
            
            if self.parallel_inference_student:
                del self.parallel_inference_student
                self.parallel_inference_student = None
            
            if self.parallel_inference_teacher:
                del self.parallel_inference_teacher
                self.parallel_inference_teacher = None
            
            if self.async_cache_manager:
                del self.async_cache_manager
                self.async_cache_manager = None
            
            if self.parallel_data_loader:
                del self.parallel_data_loader
                self.parallel_data_loader = None
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("🧹 所有资源清理完成")
            
        except Exception as e:
            self.logger.error(f"❌ 资源清理失败: {e}")
            # 记录详细的错误信息
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        # 初始化wandb（如果启用）
        if self.config.get("logging", {}).get("use_wandb", False):
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                config=self.config
            )
    
    def setup_models(self):
        """设置教师和学生模型"""
        # 注意：不使用 @handle_errors，因为需要确保失败时抛出异常，而不是返回 None
        try:
            self.logger.info("🚀 开始设置模型...")
            
            # 检查GPU数量并决定模型分配策略
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"📊 检测到 {num_gpus} 个GPU设备")
            
            # 🎯 检测GPU显存大小并调整清理频率（H200优化）
            if num_gpus >= 1 and not self._vram_detected:
                try:
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.logger.info(f"📊 GPU 0显存: {total_vram_gb:.1f}GB")
                    if total_vram_gb >= 120:  # H200 140GB或类似大显存GPU
                        self._memory_cleanup_interval = 5  # 🔥 紧急修复：从50降到5，更频繁清理
                        self._force_cleanup_every_n_steps = 3  # 🔥 紧急修复：从50降到3
                        self.logger.info("⚠️ 检测到H200等大显存GPU，但使用保守清理策略（每5步清理）避免OOM")
                    else:  # A100 80GB或更小显存
                        self._memory_cleanup_interval = 3
                        self._force_cleanup_every_n_steps = 2
                        self.logger.info(f"📊 使用保守清理策略（每3步清理）")
                    self._vram_detected = True
                except Exception as e:
                    self.logger.warning(f"⚠️ 检测GPU显存失败，使用默认策略: {e}")
            
            # 根据GPU数量选择最优分配策略
            if num_gpus >= 4:
                # 4卡或更多：Teacher跨GPU 0,1自动分布，Student在GPU 2，GPU 3备用/缓存
                # 使用max_memory限制Teacher模型只在GPU 0和1上分配
                # 这样可以确保Student模型可以安全使用GPU 2
                import os
                # 设置max_memory限制，只允许在GPU 0和1上分配模型
                max_memory = {
                    0: "75GB",  # GPU 0: 预留5GB系统显存
                    1: "75GB",  # GPU 1: 预留5GB系统显存
                }
                teacher_device_map = "auto"  # 配合max_memory使用，自动分配到GPU 0和1
                # 临时设置max_memory环境变量（如果HuggingFace支持）
                # 注意：实际使用时需要在from_pretrained中传入max_memory参数
                student_device_map = "cuda:2"  # Student模型（7B）放在GPU 2
                self.logger.info("✅ 4卡配置：Teacher→GPU 0,1（自动平衡分布），Student→GPU 2，GPU 3备用/缓存")
                self.logger.info("   显存分配：Teacher(32B)约70GB，Student(7B+PPO)约40-50GB，剩余~200GB安全余量")
                # 存储max_memory供后续使用
                self._teacher_max_memory = max_memory
            elif num_gpus >= 2:
                # 2卡：Teacher单卡GPU 1，Student单卡GPU 0
                # 🎯 注意：模型并行（跨GPU）会增加通信开销和延迟
                # 单卡140GB H200足够装下Teacher 32B（64GB）+ Student 7B（14GB）
                teacher_device_map = "cuda:1"  # Teacher 32B单卡GPU 1
                student_device_map = "cuda:0"  # Student 7B + PPO单卡GPU 0
                self.logger.info("✅ 2卡配置：Student+PPO→GPU 0, Teacher→GPU 1")
                self.logger.info("   显存：GPU 0约50GB，GPU 1约70GB，都充足")
                self.logger.info("   💡 如遇OOM，可能是batch_size太大，而非单卡容量不足")
            else:
                # 如果只有一个GPU，使用auto
                teacher_device_map = self.config["device"]["device_map"]
                student_device_map = self.config["device"]["device_map"]
                self.logger.warning("⚠️ 只有一个GPU，使用auto分配")
            
            # 加载教师模型
            self.logger.info("📚 加载教师模型...")
            with tqdm(total=3, desc="教师模型加载", ncols=80) as pbar:
                from models.teacher_model import TeacherModel
                # 准备Teacher模型初始化参数
                teacher_kwargs = {
                    "model_name": self.config["model"]["teacher_model_name"],
                    "cache_size": self.config["model"]["cache_size"],
                    "cache_policy": self.config["model"]["cache_policy"],
                    "device": teacher_device_map,  # 使用显式分配的GPU
                    "torch_dtype": getattr(torch, self.config["device"]["torch_dtype"])
                }
                # 如果是4卡配置，传递max_memory限制
                if num_gpus >= 4 and hasattr(self, '_teacher_max_memory'):
                    teacher_kwargs["max_memory"] = self._teacher_max_memory
                
                self.teacher_model = TeacherModel(**teacher_kwargs)
                pbar.update(1)
                pbar.set_postfix({"status": "教师模型加载完成"})
                
                # 检查Teacher模型实际分布
                if hasattr(self.teacher_model.model, 'hf_device_map'):
                    device_map = self.teacher_model.model.hf_device_map
                    self.logger.info(f"📊 Teacher模型设备分布: {device_map}")
                    
                    # ⚠️ 检查设备分配是否平衡（仅对4卡配置）
                    if num_gpus >= 4 and isinstance(device_map, dict):
                        gpu_0_layers = sum(1 for v in device_map.values() if v == 0 or (isinstance(v, (list, tuple)) and 0 in v))
                        gpu_1_layers = sum(1 for v in device_map.values() if v == 1 or (isinstance(v, (list, tuple)) and 1 in v))
                        total_layers = gpu_0_layers + gpu_1_layers
                        if total_layers > 0:
                            balance_ratio = min(gpu_0_layers, gpu_1_layers) / max(gpu_0_layers, gpu_1_layers)
                            if balance_ratio < 0.7:  # 如果分配不平衡度超过30%
                                self.logger.warning(f"⚠️ Teacher模型设备分配不平衡：GPU 0有{gpu_0_layers}层，GPU 1有{gpu_1_layers}层（平衡度{balance_ratio:.2%}）")
                                self.logger.warning("   建议检查显存使用，可能需要手动调整device_map")
                
                # 加载学生模型
                self.logger.info("🎓 加载学生模型...")
                from models.student_model import StudentModel
                self.student_model = StudentModel(
                    model_name=self.config["model"]["student_model_name"],
                    lora_config=self.config["lora"],
                    device=student_device_map,  # 使用显式分配的GPU
                    torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                    use_lora=True
                )
                pbar.update(1)
                pbar.set_postfix({"status": "学生模型加载完成"})
                
                # 🔥 关键验证：确保teacher和student使用相同tokenizer或相同大小
                teacher_tok_size = len(self.teacher_model.tokenizer)
                student_tok_size = len(self.student_model.tokenizer)
                if teacher_tok_size != student_tok_size:
                    self.logger.warning(f"⚠️ Teacher tokenizer大小 ({teacher_tok_size}) != Student tokenizer大小 ({student_tok_size})")
                    self.logger.warning(f"   这可能导致vocab_size不匹配问题，已启用的'限域+clamp'策略会保护")
                else:
                    self.logger.info(f"✅ Teacher和Student tokenizer大小一致: {teacher_tok_size}")
                
                # 🔥 关键验证：检查真实embedding大小
                try:
                    teacher_input_emb = self.teacher_model.model.get_input_embeddings().weight.size(0)
                    student_input_emb = self.student_model.model.get_input_embeddings().weight.size(0)
                    self.logger.info(f"📊 真实embedding大小:")
                    self.logger.info(f"   Teacher input_embeddings: {teacher_input_emb}")
                    self.logger.info(f"   Student input_embeddings: {student_input_emb}")
                    self.logger.info(f"   Teacher tokenizer: {teacher_tok_size}")
                    self.logger.info(f"   Student tokenizer: {student_tok_size}")
                    
                    if teacher_input_emb != teacher_tok_size:
                        self.logger.warning(f"⚠️ Teacher embedding ({teacher_input_emb}) != tokenizer ({teacher_tok_size})")
                    if student_input_emb != student_tok_size:
                        self.logger.warning(f"⚠️ Student embedding ({student_input_emb}) != tokenizer ({student_tok_size})")
                except Exception as e:
                    self.logger.warning(f"⚠️ 无法检查embedding大小: {e}")
                
                # 检查Student模型实际分布
                if hasattr(self.student_model.model, 'hf_device_map'):
                    self.logger.info(f"📊 Student模型设备分布: {self.student_model.model.hf_device_map}")
                
                # 设置PPO模型
                self.logger.info("⚙️ 设置PPO模型...")
                self.ppo_model = self.student_model.setup_for_ppo()
                pbar.update(1)
                pbar.set_postfix({"status": "PPO模型设置完成"})
            
            # 更新模型以使用现代缓存
            self.teacher_model.model = update_model_for_modern_cache(self.teacher_model.model)
            self.student_model.model = update_model_for_modern_cache(self.student_model.model)
            
            self.logger.info("✅ 模型设置完成")
            self.logger.info("模型设置完成")
            
        except Exception as e:
            self.logger.error(f"模型设置失败: {e}")
            raise
    
    def setup_components(self):
        """设置奖励计算组件"""
        try:
            print("🔧 开始设置组件...")
            
            with tqdm(total=6, desc="组件设置", ncols=80) as pbar:
                # 缓存管理器
                print("💾 设置缓存管理器...")
                from models.cache_manager import CacheManager
                self.cache_manager = CacheManager(
                    max_cache_size=self.config["model"]["cache_size"],
                    eviction_policy=self.config["model"]["cache_policy"]
                )
                pbar.update(1)
                pbar.set_postfix({"status": "缓存管理器设置完成"})
                
                # 内在奖励计算器
                print("🧠 设置内在奖励计算器...")
                self.intrinsic_computer = IntrinsicRewardComputer(
                    temperature=self.config["reward"]["temperature"],
                    normalization_method=self.config["reward"]["normalization"],
                    update_rate=self.config["reward"].get("update_rate", 0.01)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "内在奖励计算器设置完成"})
                
                # 奖励归一化器
                print("📊 设置奖励归一化器...")
                self.reward_normalizer = RewardNormalizer(
                    method=self.config["reward"]["normalization"],
                    clip_min=self.config["reward"].get("clip_min", -5.0),
                    clip_max=self.config["reward"].get("clip_max", 5.0)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "奖励归一化器设置完成"})
                
                # 奖励组合器
                print("🔗 设置奖励组合器...")
                self.reward_combiner = RewardCombiner(
                    lambda_intrinsic=self.config["reward"]["lambda_intrinsic"],
                    lambda_correctness=self.config["reward"]["lambda_correctness"],
                    lambda_reasoning=self.config["reward"].get("reasoning_weight", 0.0),
                    lambda_format=self.config["reward"].get("format_weight", 0.0),
                    use_adaptive_weights=self.config["reward"].get("use_adaptive_weights", True),
                    adaptation_rate=self.config["reward"].get("adaptation_rate", 0.01)
                )
                pbar.update(1)
                pbar.set_postfix({"status": "奖励组合器设置完成"})
                
                # 记录自适应权重状态
                if self.config["reward"].get("use_adaptive_weights", True):
                    self.logger.info("自适应权重功能已启用")
                    self.logger.info(f"初始权重 - 内在: {self.config['reward']['lambda_intrinsic']}, "
                                   f"正确性: {self.config['reward']['lambda_correctness']}, "
                                   f"推理: {self.config['reward'].get('reasoning_weight', 0.0)}, "
                                   f"格式: {self.config['reward'].get('format_weight', 0.0)}")
                    self.logger.info(f"权重适应率: {self.config['reward'].get('adaptation_rate', 0.01)}")
                else:
                    self.logger.info("自适应权重功能已禁用，使用固定权重")
                
                # 数据处理器
                print("📝 设置数据处理器...")
                if self.student_model is None or self.student_model.tokenizer is None:
                    raise ValueError("student_model 或 tokenizer 未设置。请先调用 setup_models() 方法")
                
                self.data_processor = GSM8KProcessor(
                    tokenizer=self.student_model.tokenizer,
                    max_length=self.config["ppo"]["max_length"]
                )
                pbar.update(1)
                pbar.set_postfix({"status": "数据处理器设置完成"})
                
                # 初始化并行处理组件
                print("⚡ 设置并行处理组件...")
                self._setup_parallel_components()
                pbar.update(1)
                pbar.set_postfix({"status": "并行处理组件设置完成"})
            
            print("✅ 组件设置完成")
            self.logger.info("组件设置完成")
            
        except Exception as e:
            self.logger.error(f"组件设置失败: {e}")
            raise
    
    def _setup_parallel_components(self):
        """设置并行处理组件"""
        try:
            # 检查是否启用并行处理
            use_parallel = self._use_parallel
            if not use_parallel:
                self.logger.info("并行处理已禁用")
                return
            
            # 并行奖励处理器
            self.parallel_processor = create_parallel_processor(self.config)
            
            # 并行模型推理器
            self.parallel_inference_student = create_parallel_inference(
                self.student_model, self.config
            )
            self.parallel_inference_teacher = create_parallel_inference(
                self.teacher_model, self.config
            )
            
            # 异步缓存管理器
            self.async_cache_manager = create_async_cache_manager(
                self.cache_manager, self.config
            )
            
            # 启动异步缓存工作线程
            self.async_cache_manager.start_async_worker()
            
            self.logger.info("并行处理组件设置完成")
            
        except Exception as e:
            self.logger.error(f"并行处理组件设置失败: {e}")
            # 如果并行处理设置失败，回退到串行处理
            self.logger.warning("回退到串行处理模式")
    
    def setup_ppo_trainer(self):
        """设置PPO训练器"""
        try:
            # 检查ppo_model是否已设置
            if self.ppo_model is None:
                raise ValueError("ppo_model 未设置。请先调用 setup_models() 方法")
            
            # 检查student_model和tokenizer是否已设置
            if self.student_model is None or self.student_model.tokenizer is None:
                raise ValueError("student_model 或 tokenizer 未设置。请先调用 setup_models() 方法")
            
            from inspect import signature

            raw = dict(
                model_name=self.config["model"]["student_model_name"],
                learning_rate=float(self.config["ppo"]["learning_rate"]),
                batch_size=self.config["ppo"]["batch_size"],
                mini_batch_size=self.config["ppo"]["mini_batch_size"],
                ppo_epochs=self.config["ppo"]["ppo_epochs"],
                # 兼容 clip_range / clip_ratio
                clip_range=self.config["ppo"].get("clip_range", self.config["ppo"].get("clip_ratio", 0.2)),
                value_loss_coef=self.config["ppo"].get("value_loss_coef", 0.5),
                entropy_beta=self.config["ppo"].get("entropy_coef", 0.0),  # 兼容 entropy_coef
                kl_coef=self.config["ppo"].get("kl_coef", 0.01),
                gamma=self.config["ppo"].get("gamma", 0.99),
                lambda_gae=self.config["ppo"].get("lambda_gae", 0.95),
                max_grad_norm=self.config["ppo"].get("max_grad_norm", 1.0),
                # ✅ 添加 forward_batch_size 配置，降低前向批处理峰值显存
                forward_batch_size=self.config["ppo"].get("forward_batch_size", 2),
                # ✅ 添加生成参数，确保KL散度计算正确
                temperature=self.config["ppo"].get("temperature", 0.7),
                top_k=self.config["ppo"].get("top_k", 50),
                top_p=self.config["ppo"].get("top_p", 1.0),
                log_with="wandb" if self.config.get("logging", {}).get("use_wandb", False) else None,
                tracker_project_name=self.config.get("logging", {}).get("wandb_project", "intrinsic-reward-distillation"),
            )

            # 只保留 PPOConfig.__init__ 真正支持的键
            allowed = set(signature(PPOConfig.__init__).parameters.keys())
            filtered_raw = {k: v for k, v in raw.items() if k in allowed}
            
            # 调试：打印被过滤掉的键
            filtered_out = {k: v for k, v in raw.items() if k not in allowed}
            if filtered_out:
                self.logger.warning(f"⚠️ PPOConfig不支持的参数（将被忽略）: {filtered_out.keys()}")
            
            ppo_config = PPOConfig(**filtered_raw)
            
            # 注意：我们不在这里传入 dataset，因为数据是在训练循环中动态创建的
            # 显式设置 dataset=None 以避免警告
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.ppo_model,
                tokenizer=self.student_model.tokenizer,
                dataset=None  # 使用自定义数据加载逻辑，不传入数据集
            )
            
            # 🔍 诊断：检查ref_model设置
            if hasattr(self.ppo_trainer, 'ref_model'):
                self.logger.info("✅ PPO trainer有ref_model")
                # 检查ref_model参数是否冻结
                if hasattr(self.ppo_trainer.ref_model, 'parameters'):
                    frozen_params = sum(1 for p in self.ppo_trainer.ref_model.parameters() if not p.requires_grad)
                    total_params = sum(1 for _ in self.ppo_trainer.ref_model.parameters())
                    self.logger.info(f"   Ref model冻结参数: {frozen_params}/{total_params}")
            else:
                self.logger.error("❌ PPO trainer没有ref_model！这是导致KL=0的根源！")
            
            # ✅ 应用梯度检查点配置（如果启用）：在PPO trainer初始化后启用
            # 注意：gradient_checkpointing已经在setup_for_ppo中启用，但确保PPO trainer也应用
            if self.config["ppo"].get("gradient_checkpointing", False):
                try:
                    # 确保PPO trainer的模型也启用了梯度检查点
                    if hasattr(self.ppo_trainer, 'model') and hasattr(self.ppo_trainer.model, 'gradient_checkpointing_enable'):
                        self.ppo_trainer.model.gradient_checkpointing_enable()
                        self.logger.info("✅ PPO Trainer模型已启用梯度检查点")
                    elif hasattr(self.ppo_trainer, 'ref_model') and hasattr(self.ppo_trainer.ref_model, 'gradient_checkpointing_enable'):
                        # 确保ref_model也启用（用于KL散度计算）
                        self.ppo_trainer.ref_model.gradient_checkpointing_enable()
                        self.logger.info("✅ PPO Trainer ref_model已启用梯度检查点")
                except Exception as e:
                    self.logger.warning(f"⚠️ 在PPO trainer中启用梯度检查点失败: {e}")
            
            # 📊 GPU分配说明：
            # - Teacher模型：自动分布在GPU 0和1（已设置max_memory）
            # - Student基础模型：在GPU 2（已设置device="cuda:2"）
            # - PPO模型（policy + ref）：由Accelerator自动管理，通常在GPU 0（主设备）
            #   原因：PPO训练需要policy和ref在同一设备，以便计算KL散度和梯度同步
            # 
            # ⚠️ 重要：PPO模型不会自动分配到不同GPU
            # - 这是PPO trainer的设计限制，不是bug
            # - 通过gradient_checkpointing、forward_batch_size=1等优化避免OOM
            # - GPU 0总使用约75GB < 80GB，安全余量充足
            
            num_gpus = torch.cuda.device_count()
            # 打印GPU分配情况（适用于所有GPU配置）
            try:
                self.logger.info("📊 模型设备分配情况：")
                
                def get_model_device(model_obj):
                    """安全地获取模型设备"""
                    if model_obj is None:
                        return None
                    try:
                        # 尝试获取第一个参数的设备
                        for param in model_obj.parameters():
                            return param.device
                    except:
                        pass
                    # 尝试从pretrained_model获取
                    if hasattr(model_obj, 'pretrained_model'):
                        try:
                            for param in model_obj.pretrained_model.parameters():
                                return param.device
                        except:
                            pass
                    return None
                
                # 检查Teacher模型分布
                if self.teacher_model and hasattr(self.teacher_model, 'model'):
                    try:
                        teacher_params = list(self.teacher_model.model.parameters())[:5]  # 检查前5个参数
                        devices = [p.device for p in teacher_params]
                        unique_devices = set(str(d) for d in devices)
                        self.logger.info(f"   Teacher模型设备: {', '.join(unique_devices)}")
                    except:
                        self.logger.info(f"   Teacher模型设备: 未知")
                
                # 检查Student基础模型
                if self.student_model and hasattr(self.student_model, 'model'):
                    student_device = get_model_device(self.student_model.model)
                    if student_device:
                        self.logger.info(f"   Student基础模型设备: {student_device}")
                
                # 检查PPO模型
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    policy_device = get_model_device(self.ppo_trainer.model)
                    if policy_device:
                        self.logger.info(f"   PPO Policy模型设备: {policy_device} (由Accelerator管理)")
                    else:
                        self.logger.warning("   ⚠️ 无法获取Policy模型设备")
                
                if hasattr(self.ppo_trainer, 'ref_model') and self.ppo_trainer.ref_model is not None:
                    ref_device = get_model_device(self.ppo_trainer.ref_model)
                    if ref_device:
                        self.logger.info(f"   PPO Ref模型设备: {ref_device} (由Accelerator管理)")
                    else:
                        self.logger.warning("   ⚠️ 无法获取Ref模型设备")
                
                # 打印显存使用情况
                self.logger.info("📊 各GPU显存使用情况：")
                for gpu_id in range(num_gpus):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    max_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    usage_pct = (allocated / max_memory * 100) if max_memory > 0 else 0
                    self.logger.info(f"   GPU {gpu_id}: {allocated:.2f}GB / {max_memory:.2f}GB ({usage_pct:.1f}%)")
                    
                # 根据GPU数量提供配置说明
                if num_gpus >= 4:
                    self.logger.info("💡 4卡配置说明：")
                    self.logger.info("   - Teacher模型分布在GPU 0,1（通过device_map='auto'）")
                    self.logger.info("   - PPO模型（policy+ref）在GPU 0（Accelerator自动管理，这是正常行为）")
                elif num_gpus >= 2:
                    self.logger.info("💡 2卡配置说明：")
                    self.logger.info("   - Teacher模型在GPU 1")
                    self.logger.info("   - Student+PPO模型在GPU 0")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 检查模型设备分配时出错: {e}")
            
            self.logger.info("PPO训练器设置完成")
            
        except Exception as e:
            self.logger.error(f"PPO训练器设置失败: {e}")
            raise
    
    def compute_intrinsic_rewards(self, questions: List[str], 
                                 student_responses: List[str]) -> torch.Tensor:
        """计算内在奖励（支持并行处理）"""
        use_parallel = self._use_parallel
        
        if use_parallel and self.parallel_processor:
            return self._compute_intrinsic_rewards_parallel(questions, student_responses)
        else:
            return self._compute_intrinsic_rewards_sequential(questions, student_responses)
    
    def _compute_intrinsic_rewards_sequential(self, questions: List[str], 
                                            student_responses: List[str]) -> torch.Tensor:
        """串行计算内在奖励（原始实现）"""
        # 检查必要的组件
        if self.teacher_model is None:
            raise ValueError("teacher_model 未设置。请先调用 setup_models() 方法")
        if self.student_model is None:
            raise ValueError("student_model 未设置。请先调用 setup_models() 方法")
        if self.cache_manager is None:
            raise ValueError("cache_manager 未设置。请先调用 setup_components() 方法")
        if self.intrinsic_computer is None:
            raise ValueError("intrinsic_computer 未设置。请先调用 setup_components() 方法")
        if self.reward_normalizer is None:
            raise ValueError("reward_normalizer 未设置。请先调用 setup_components() 方法")
        
        intrinsic_rewards = []
        
        # 创建进度条
        progress_bar = self._create_progress_bar(
            zip(questions, student_responses), 
            desc="计算内在奖励"
        )
        
        for question, response in progress_bar:
            # 构建完整序列
            full_sequence = question + response
            
            # 🔥 禁用缓存：直接计算teacher logits（因为命中率一直是0%）
            teacher_logits = self.teacher_model.get_logits(full_sequence, use_cache=False)
            
            # 获取学生tokens（在循环中调用，但tokenizer本身很快）
            student_tokens = self.student_model.tokenizer.encode(response, add_special_tokens=False)
            student_tokens = torch.tensor(student_tokens).unsqueeze(0)
            
            # 计算问题部分的长度
            question_tokens = self.student_model.tokenizer.encode(question, add_special_tokens=False)
            question_length = len(question_tokens)
            
            # 计算内在奖励
            intrinsic_reward = self.intrinsic_computer.compute_intrinsic_reward(
                teacher_logits, student_tokens, question_length
            )
            
            # 归一化
            normalized_intrinsic = self.reward_normalizer.normalize_intrinsic_rewards(
                intrinsic_reward
            )
            
            # 计算trajectory级别奖励
            trajectory_reward = self.intrinsic_computer.compute_trajectory_reward(
                normalized_intrinsic
            )
            
            intrinsic_rewards.append(trajectory_reward)
        
        return torch.tensor(intrinsic_rewards)
    
    def _compute_intrinsic_rewards_parallel(self, questions: List[str], 
                                          student_responses: List[str]) -> torch.Tensor:
        """并行计算内在奖励（使用教师模型并行推理）"""
        # 构建完整序列列表
        full_sequences = [question + response for question, response in zip(questions, student_responses)]
        
        # 使用教师模型并行推理获取logits
        teacher_logits_list = []
        if self.parallel_inference_teacher:
            # 使用并行推理获取教师logits
            self.logger.info("使用教师模型并行推理计算logits")
            teacher_logits_list = self.parallel_inference_teacher.get_logits_batch_parallel(full_sequences)
        else:
            # 回退到串行推理
            self.logger.info("使用教师模型串行推理计算logits")
            for full_sequence in full_sequences:
                # 🔥 禁用缓存：直接计算teacher logits（因为命中率一直是0%）
                teacher_logits = self.teacher_model.get_logits(full_sequence, use_cache=False)
                teacher_logits_list.append(teacher_logits)
        
        # 并行计算内在奖励
        with self.parallel_processor as processor:
            def compute_single_intrinsic_reward(question: str, response: str, teacher_logits: torch.Tensor) -> float:
                try:
                    # 获取学生tokens
                    student_tokens = self.student_model.tokenizer.encode(response, add_special_tokens=False)
                    student_tokens = torch.tensor(student_tokens).unsqueeze(0)
                    
                    # 计算问题部分的长度
                    question_tokens = self.student_model.tokenizer.encode(question, add_special_tokens=False)
                    question_length = len(question_tokens)
                    
                    # 计算内在奖励
                    intrinsic_reward = self.intrinsic_computer.compute_intrinsic_reward(
                        teacher_logits, student_tokens, question_length
                    )
                    
                    # 归一化
                    normalized_intrinsic = self.reward_normalizer.normalize_intrinsic_rewards(
                        intrinsic_reward
                    )
                    
                    # 计算trajectory级别奖励
                    trajectory_reward = self.intrinsic_computer.compute_trajectory_reward(
                        normalized_intrinsic
                    )
                    
                    return trajectory_reward
                    
                except Exception as e:
                    self.logger.error(f"并行内在奖励计算失败: {e}")
                    return 0.0
            
            # 并行计算奖励
            intrinsic_rewards = []
            for i, (question, response) in enumerate(zip(questions, student_responses)):
                if i < len(teacher_logits_list) and teacher_logits_list[i] is not None:
                    reward = compute_single_intrinsic_reward(question, response, teacher_logits_list[i])
                    intrinsic_rewards.append(reward)
                else:
                    self.logger.warning(f"教师logits缺失，使用默认奖励: {i}")
                    intrinsic_rewards.append(0.0)
        
        return torch.tensor(intrinsic_rewards)
    
    def compute_correctness_rewards(self, questions: List[str], 
                                   student_responses: List[str]) -> torch.Tensor:
        """计算答案正确性奖励（支持并行处理）"""
        use_parallel = self._use_parallel
        
        if use_parallel and self.parallel_processor:
            return self._compute_correctness_rewards_parallel(questions, student_responses)
        else:
            return self._compute_correctness_rewards_sequential(questions, student_responses)
    
    def _compute_correctness_rewards_sequential(self, questions: List[str], 
                                              student_responses: List[str]) -> torch.Tensor:
        """串行计算答案正确性奖励（原始实现）"""
        # 检查必要的组件
        if self.teacher_model is None:
            raise ValueError("teacher_model 未设置。请先调用 setup_models() 方法")
        
        correctness_rewards = []
        
        # 创建进度条
        progress_bar = self._create_progress_bar(
            zip(questions, student_responses), 
            desc="计算正确性奖励"
        )
        
        for question, response in progress_bar:
            # 提取学生答案
            student_answer = extract_final_answer(response)
            
            # 提取正确答案（从问题中或使用教师模型生成）
            # 这里简化处理，实际应该从数据集中获取正确答案
            teacher_response = self.teacher_model.generate_response(question, max_length=256)
            correct_answer = extract_final_answer(teacher_response)
            
            # 判断答案是否正确
            is_correct = is_answer_correct(student_answer, correct_answer)
            
            correctness_rewards.append(1.0 if is_correct else 0.0)
        
        return torch.tensor(correctness_rewards)
    
    def _compute_correctness_rewards_parallel(self, questions: List[str], 
                                            student_responses: List[str]) -> torch.Tensor:
        """并行计算答案正确性奖励"""
        with self.parallel_processor as processor:
            # 定义奖励计算函数
            def compute_single_correctness_reward(question: str, response: str) -> float:
                try:
                    # 提取学生答案
                    student_answer = extract_final_answer(response)
                    
                    # 提取正确答案（从问题中或使用教师模型生成）
                    # 这里简化处理，实际应该从数据集中获取正确答案
                    teacher_response = self.teacher_model.generate_response(question, max_length=256)
                    correct_answer = extract_final_answer(teacher_response)
                    
                    # 判断答案是否正确
                    is_correct = is_answer_correct(student_answer, correct_answer)
                    
                    return 1.0 if is_correct else 0.0
                    
                except Exception as e:
                    self.logger.error(f"并行正确性奖励计算失败: {e}")
                    return 0.0
            
            # 并行计算奖励
            correctness_rewards = processor.compute_rewards_parallel(
                questions, student_responses, compute_single_correctness_reward
            )
            
            return torch.tensor(correctness_rewards)
    
    def compute_combined_rewards(self, questions: List[str], 
                               student_responses: List[str]) -> torch.Tensor:
        """计算组合奖励（单线程串行处理，避免tokenizer线程安全问题）"""
        # 检查奖励组合器是否已设置
        if self.reward_combiner is None:
            raise ValueError("reward_combiner 未设置。请先调用 setup_components() 方法")
        
        # 强制使用串行计算，避免多线程导致的tokenizer线程安全问题
        # 串行计算
        intrinsic_rewards = self.compute_intrinsic_rewards(questions, student_responses)
        correctness_rewards = self.compute_correctness_rewards(questions, student_responses)
        
        # 组合奖励
        combined_rewards = self.reward_combiner.combine_rewards(
            intrinsic_rewards, correctness_rewards
        )
        
        return combined_rewards
    
    def train_step(self, batch: Dict[str, List[str]]) -> Dict[str, float]:
        """执行一步训练（支持并行处理）"""
        # 注意：不使用 @handle_errors，因为需要确保失败时抛出异常，而不是返回 None
        # 这样调用者可以决定如何处理错误
        try:
            # 检查必要的组件
            if self.student_model is None:
                raise ValueError("student_model 未设置。请先调用 setup_models() 方法")
            if self.ppo_trainer is None:
                raise ValueError("ppo_trainer 未设置。请先调用 setup_ppo_trainer() 方法")
            
            # 验证数据批次
            if "questions" not in batch or not batch["questions"]:
                raise ValueError("❌ 缺少问题数据")
            
            if not isinstance(batch["questions"], list):
                raise ValueError("❌ 问题数据不是列表")
            
            questions = batch["questions"]
            
            # 检查问题数量
            if len(questions) == 0:
                raise ValueError("❌ 问题列表为空")
            
            if len(questions) > 100:  # 防止批次过大
                self.logger.warning(f"⚠️ 批次大小过大: {len(questions)}, 截断到100")
                questions = questions[:100]
            
            # 学生模型生成响应（支持并行）
            use_parallel = self._use_parallel
            
            # 使用no_grad进行推理，节省内存
            with torch.no_grad():
                if use_parallel and self.parallel_inference_student:
                    # ✅ 统一使用配置中的max_length
                    student_responses = self.parallel_inference_student.generate_batch_parallel(
                        questions,
                        max_length=self.config["ppo"]["max_length"],  # ✅ 统一使用配置的max_length
                        temperature=self.config["ppo"]["temperature"],
                        do_sample=self.config["ppo"]["do_sample"]
                    )
                else:
                    # ✅ 统一使用配置中的max_length，确保一致性
                    # 注意：生成时的max_length是生成新token数，实际总长度 = query_length + max_length
                    # 但这里使用配置的max_length作为参考，实际生成会更短（生成时限制）
                    student_responses = self.student_model.generate(
                        questions,
                        max_length=self.config["ppo"]["max_length"],  # ✅ 统一使用配置的max_length
                        temperature=self.config["ppo"]["temperature"],
                        do_sample=self.config["ppo"]["do_sample"]
                    )
                    # 确保返回值是列表类型（student_model.generate 在单个prompt时可能返回单个字符串）
                    if isinstance(student_responses, str):
                        student_responses = [student_responses]
                    if not isinstance(student_responses, list):
                        raise TypeError(f"student_model.generate 返回了意外的类型: {type(student_responses)}")
            
            # 计算奖励
            combined_rewards = self.compute_combined_rewards(questions, student_responses)
            
            # 🔍 诊断：打印奖励统计信息（前几步）
            if self.training_stats["step"] < 3:
                self.logger.info(f"🎁 Step {self.training_stats['step'] + 1} - 奖励诊断:")
                self.logger.info(f"   Combined rewards: shape={combined_rewards.shape}, "
                               f"mean={combined_rewards.mean():.4f}, std={combined_rewards.std():.4f}, "
                               f"min={combined_rewards.min():.4f}, max={combined_rewards.max():.4f}")
                self.logger.info(f"   奖励值分布: {combined_rewards.tolist()}")
            
            # 🔥 关键优化：Teacher推理后立即清理显存（Teacher模型占用大量显存）
            # 奖励计算完成后，Teacher的中间激活不再需要，立即释放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 验证长度匹配
            if len(combined_rewards) != len(questions):
                raise ValueError(f"奖励数量 {len(combined_rewards)} 与问题数量 {len(questions)} 不匹配")
            if len(student_responses) != len(questions):
                raise ValueError(f"响应数量 {len(student_responses)} 与问题数量 {len(questions)} 不匹配")
            
            # 将问题转换为tokenized张量列表
            # 使用批量tokenize以提高效率并避免线程安全问题
            # 注意：tokenizer是线程安全的，但为了更安全，使用批量处理
            try:
                tokenized_queries_batch = self.student_model.tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config["ppo"]["max_length"]
                )
                tokenized_queries = [tokenized_queries_batch["input_ids"][i] for i in range(len(questions))]
            except RuntimeError as e:
                if "Already borrowed" in str(e):
                    # 如果遇到线程安全问题，使用单线程处理
                    self.logger.warning("检测到tokenizer线程安全问题，使用单线程处理")
                    tokenized_queries = []
                    for question in questions:
                        tokenized = self.student_model.tokenizer(
                            question,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config["ppo"]["max_length"]
                        )
                        tokenized_queries.append(tokenized["input_ids"])
                else:
                    raise
            
            # 将响应转换为tokenized张量列表
            try:
                tokenized_responses_batch = self.student_model.tokenizer(
                    student_responses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config["ppo"]["max_length"]
                )
                tokenized_responses = [tokenized_responses_batch["input_ids"][i] for i in range(len(student_responses))]
            except RuntimeError as e:
                if "Already borrowed" in str(e):
                    # 如果遇到线程安全问题，使用单线程处理
                    self.logger.warning("检测到tokenizer线程安全问题，使用单线程处理")
                    tokenized_responses = []
                    for response in student_responses:
                        tokenized = self.student_model.tokenizer(
                            response,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config["ppo"]["max_length"]
                        )
                        tokenized_responses.append(tokenized["input_ids"])
                else:
                    raise
            
            # PPO更新
            # 确保分数都是张量而不是 float
            device = getattr(self.ppo_trainer.accelerator, "device", "cuda")
            
            # 确保combined_rewards是1D tensor，然后转换为张量列表
            if combined_rewards.dim() > 1:
                combined_rewards = combined_rewards.squeeze()
            
            # 验证tokenized列表长度匹配
            if len(tokenized_queries) != len(tokenized_responses):
                raise ValueError(f"问题token数量 {len(tokenized_queries)} 与响应token数量 {len(tokenized_responses)} 不匹配")
            if len(tokenized_queries) != len(combined_rewards):
                raise ValueError(f"问题token数量 {len(tokenized_queries)} 与奖励数量 {len(combined_rewards)} 不匹配")
            
            scores = []
            for i in range(len(combined_rewards)):
                reward = combined_rewards[i]
                if torch.is_tensor(reward):
                    scores.append(reward.to(device=device, dtype=torch.float32))
                else:
                    scores.append(torch.tensor(reward, dtype=torch.float32, device=device))
            
            # 最终验证scores长度
            if len(scores) != len(tokenized_queries):
                raise ValueError(f"分数数量 {len(scores)} 与查询数量 {len(tokenized_queries)} 不匹配")
            
            # ✅ 修复：在删除combined_rewards之前保存统计信息（用于后续日志和统计更新）
            # 保存均值奖励（移到CPU并转为Python标量，释放GPU显存）
            mean_reward_for_stats = torch.mean(combined_rewards).cpu().item()
            # 保存combined_rewards的CPU副本用于统计更新（保持为tensor格式，因为_update_training_stats需要）
            combined_rewards_cpu = combined_rewards.cpu().clone()
            
            # 🔥 关键优化：在PPO step之前立即清理内存和释放不需要的变量
            # 释放原始字符串数据（已tokenize，不再需要）
            if 'questions' in locals():
                del questions
            if 'student_responses' in locals():
                del student_responses
            # 释放批次tokenized张量（已经提取为列表）
            if 'tokenized_queries_batch' in locals():
                del tokenized_queries_batch
            if 'tokenized_responses_batch' in locals():
                del tokenized_responses_batch
            # 释放combined_rewards（已转换为scores，CPU副本已保存）
            del combined_rewards
            
            # 🔥 极端显存清理：在PPO step之前（log_softmax是显存峰值）
            # 在4×GPU配置下，log_softmax需要同时计算policy和ref模型，显存压力极大
            if torch.cuda.is_available():
                # 清理所有GPU的显存
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()  # Python垃圾回收
            
            # 🔥 核心问题：PPO trainer在batched_forward_pass中计算log_softmax时会展开全词表矩阵 (B×T×V)
            # 即使batch_size=1, seq_len=192，vocab_size=100k时，这个矩阵在bfloat16下也需要约38MB
            # 但policy和ref模型同时计算，加上梯度，显存峰值会飙升到几百MB甚至GB
            # 关键：即使4张卡数据并行，每张卡仍要在自己的micro-batch上计算完整 (seq_len × vocab_size)
            
            # ✅ 优化策略：在调用PPO trainer前，检查并截断超长序列
            max_allowed_length = self.config["ppo"]["max_length"]
            
            # 截断超长序列（避免log_softmax OOM）
            truncated_queries = []
            truncated_responses = []
            for q, r in zip(tokenized_queries, tokenized_responses):
                # 计算总长度（query + response）
                total_len = len(q) + len(r)
                
                if total_len > max_allowed_length:
                    # 如果超长，优先保留query，截断response
                    query_len = len(q)
                    max_response_len = max(0, max_allowed_length - query_len)
                    
                    if max_response_len > 0:
                        # 截断response到允许的最大长度
                        truncated_r = r[:max_response_len]
                        truncated_queries.append(q)
                        truncated_responses.append(truncated_r)
                        if self.training_stats["step"] < 3:  # 只在前几步警告
                            self.logger.warning(f"⚠️ 序列超长（{total_len} > {max_allowed_length}），已截断response到{max_response_len}。建议降低max_length避免log_softmax OOM。")
                    else:
                        # query本身太长，跳过这个样本
                        self.logger.warning(f"⚠️ Query太长（{query_len} > {max_allowed_length}），跳过此样本。")
                        continue
                else:
                    truncated_queries.append(q)
                    truncated_responses.append(r)
            
            # 如果所有样本都被截断，跳过这一步
            if len(truncated_queries) == 0:
                self.logger.error("❌ 所有序列都被截断，跳过此训练步骤")
                return None
            
            # 🔍 设置梯度hook来捕获梯度范数（在PPO step之前）
            grad_norm_from_hook = None
            grad_hook_handles = []
            grad_norms = []  # 在外部定义，确保在整个try-finally块中可见
            
            # 为可训练参数注册hook（仅在需要时）
            try:
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    # 存储梯度用于计算范数
                    for name, param in self.ppo_trainer.model.named_parameters():
                        if param.requires_grad:
                            # 注册hook，在梯度计算时记录
                            def make_hook(n=name):
                                def hook(grad):
                                    if grad is not None:
                                        grad_norms.append((n, grad.norm().item()))
                                    return grad  # 保持梯度不变，只用于监控
                                return hook
                            handle = param.register_hook(make_hook(name))
                            grad_hook_handles.append(handle)
            except Exception as e:
                # Hook注册失败不影响训练
                pass
            
            # 执行PPO step（log_softmax计算发生在这里）
            try:
                stats = self.ppo_trainer.step(
                    queries=truncated_queries,
                    responses=truncated_responses,
                    scores=scores[:len(truncated_queries)]  # 调整scores长度匹配截断后的序列
                )
                
                # 从hook中获取梯度范数（如果有）
                if grad_norms:
                    total_grad_norm_sq = sum(norm**2 for _, norm in grad_norms)
                    grad_norm_from_hook = (total_grad_norm_sq ** 0.5)
            except torch.cuda.OutOfMemoryError as e:
                # 🔥 如果log_softmax阶段OOM，进行极端清理并提供详细诊断
                self.logger.error("❌ PPO step中log_softmax阶段OOM，执行极端显存清理...")
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                gc.collect()
                
                # 详细诊断信息
                max_seq_len = max((len(q) + len(r) for q, r in zip(truncated_queries, truncated_responses)), default=0)
                vocab_size = len(self.student_model.tokenizer) if hasattr(self.student_model, 'tokenizer') else "unknown"
                estimated_memory_mb = (max_seq_len * vocab_size * 2) / (1024 * 1024) if isinstance(vocab_size, int) else "unknown"
                
                error_msg = (
                    f"PPO step log_softmax OOM（即使batch_size=1, max_length={max_allowed_length}）。\n"
                    f"诊断信息：\n"
                    f"  - 最大序列长度: {max_seq_len}\n"
                    f"  - 词表大小: {vocab_size}\n"
                    f"  - 估算log_softmax显存: ~{estimated_memory_mb}MB（单模型，不含梯度）\n"
                    f"  - 实际显存需求: 估算值 × 2（policy+ref） × 2（梯度）≈ {estimated_memory_mb * 4 if isinstance(estimated_memory_mb, (int, float)) else 'unknown'}MB\n"
                    f"解决方案：\n"
                    f"  1) 降低max_length到128或更小（最有效）\n"
                    f"  2) 检查GPU分配是否均匀（Teacher不应集中在一张卡）\n"
                    f"  3) 考虑使用2×GPU而非4×GPU（减少数据并行开销）\n"
                    f"  4) 使用更大的GPU（H100 120GB）\n"
                    f"原始错误: {e}"
                )
                raise RuntimeError(error_msg)
            finally:
                # 清理hook
                for handle in grad_hook_handles:
                    handle.remove()
                grad_hook_handles.clear()
                grad_norms.clear()  # 清空梯度范数列表
            
            # 🔥 极端显存清理：在PPO step之后立即清理
            # log_softmax计算完成后的显存碎片需要立即清理
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            # 🔍 诊断：打印stats详细信息（每步都打印前3步，然后每10步）
            should_log_verbose = self.training_stats["step"] < 3 or self.training_stats["step"] % 10 == 0
            if should_log_verbose and stats is not None and isinstance(stats, dict):
                available_keys = list(stats.keys())
                self.logger.info(f"📊 Step {self.training_stats['step'] + 1} - PPO stats可用键: {available_keys}")
                
                # 打印所有stats的值，方便调试
                for key in available_keys:
                    try:
                        value = stats[key]
                        # 尝试转换为标量
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            value = value.item() if hasattr(value, 'item') else float(value)
                        self.logger.info(f"  {key} = {value}")
                    except Exception as e:
                        self.logger.warning(f"  无法打印 {key}: {e}")
                
                # 🔍 额外诊断：检查关键指标
                self.logger.info(f"🔍 关键诊断:")
                self.logger.info(f"  奖励: mean={mean_reward_for_stats:.4f}")
                self.logger.info(f"  stats类型: {type(stats)}")
                self.logger.info(f"  stats键数量: {len(stats)}")
            
            # 🔍 额外诊断：如果stats为空或只包含极少数键，发出警告
            if stats is not None and isinstance(stats, dict) and len(stats) < 3:
                self.logger.error(f"❌ PPO stats字典异常：只有{len(stats)}个键，可能PPOTrainer未正确计算！")
                self.logger.error(f"   可用键: {list(stats.keys())}")
                self.logger.error(f"   stats内容: {stats}")
            
            # 🔍 诊断：检查策略是否真的在更新（根据日志分析，发现KL为0的严重问题）
            if stats is not None and isinstance(stats, dict):
                # 定义to_scalar辅助函数（用于诊断）
                def _to_scalar(value, default=0):
                    if value is None:
                        return default
                    if isinstance(value, (np.ndarray, np.generic)):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    if isinstance(value, torch.Tensor):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    return float(value)
                
                # 检查KL散度
                approx_kl = _to_scalar(stats.get("ppo/policy/approxkl") or stats.get("ppo/policy/policykl") or 0)
                advantages_mean = _to_scalar(stats.get("ppo/policy/advantages_mean") or 0)
                clipfrac = _to_scalar(stats.get("ppo/policy/clipfrac") or 0)
                policy_loss_val = _to_scalar(stats.get("ppo/loss/policy") or stats.get("ppo/policy/loss") or 0)
                
                # 如果KL为0且优势接近0，说明策略几乎没有更新
                # 注意：训练初期（前几步）出现这种情况可能正常，只有在持续多步后才警告
                if abs(approx_kl) < 1e-6 and abs(advantages_mean) < 1e-6:
                    # 只在第5步之后才警告，避免训练初期的正常波动
                    if self.training_stats["step"] >= 5 and (self.training_stats["step"] % 50 == 0 or self.training_stats["step"] < 20):
                        # 检查最近几步是否都是这样（排除单次波动）
                        if len(self.training_stats["kl_divergences"]) >= 5:
                            recent_kls = self.training_stats["kl_divergences"][-5:]
                            all_kl_zero = all(abs(k) < 1e-6 for k in recent_kls if k is not None)
                            if all_kl_zero:
                                self.logger.warning(f"⚠️ 警告：策略可能没有更新！（已持续至少5步）")
                                self.logger.warning(f"   KL散度: {approx_kl:.10f} (接近0)")
                                self.logger.warning(f"   优势均值: {advantages_mean:.10f} (接近0)")
                                self.logger.warning(f"   策略损失: {policy_loss_val:.10f}")
                                self.logger.warning(f"   裁剪比例: {clipfrac:.4f}")
                                self.logger.warning(f"   可能原因:")
                                self.logger.warning(f"     1. 奖励尺度问题（奖励太小或太大）")
                                self.logger.warning(f"     2. 优势归一化问题")
                                self.logger.warning(f"     3. 学习率太小")
                                self.logger.warning(f"     4. policy和ref_model完全相同（应该不同）")
                        elif self.training_stats["step"] < 10:
                            # 前几步只做信息记录，不警告
                            self.logger.info(f"ℹ️ Step {self.training_stats['step'] + 1}: KL={approx_kl:.10f}, 优势={advantages_mean:.10f} (训练初期，继续观察)")
                
                # 检查价值函数训练
                val_var_explained = _to_scalar(stats.get("ppo/val/var_explained") or 0)
                if val_var_explained < 0:
                    # 只在第5步之后且持续多步才警告
                    if self.training_stats["step"] >= 5 and (self.training_stats["step"] % 50 == 0 or self.training_stats["step"] < 20):
                        self.logger.warning(f"⚠️ 价值函数训练异常：var_explained = {val_var_explained:.4f} (负值表示价值函数比简单预测均值还差)")
            
            # 🔍 检查模型参数是否在更新（用于诊断策略是否真的在训练）
            # 注意：PPOTrainer在step()内部完成梯度计算和优化，step()返回后梯度已被清除
            # 所以我们通过检查模型参数的变化来验证更新是否发生
            
            # 初始化：保存初始参数状态（仅在第一步）
            if not hasattr(self, '_prev_model_params'):
                if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                    self._prev_model_params = {}
                    for name, param in self.ppo_trainer.model.named_parameters():
                        if param.requires_grad:
                            self._prev_model_params[name] = param.data.clone()
                    if self.training_stats["step"] == 0:
                        self.logger.info(f"📊 已保存初始模型参数状态（{len(self._prev_model_params)}个可训练参数）")
            
            # 检查参数变化（每次都计算，用于wandb记录）
            param_change_info = {}
            if hasattr(self, '_prev_model_params'):
                try:
                    if hasattr(self.ppo_trainer, 'model') and self.ppo_trainer.model is not None:
                        max_change = 0.0
                        total_change = 0.0
                        changed_params = 0
                        total_params = 0
                        
                        for name, param in self.ppo_trainer.model.named_parameters():
                            if param.requires_grad and name in self._prev_model_params:
                                total_params += 1
                                prev_param = self._prev_model_params[name]
                                # 计算参数变化
                                param_diff = (param.data - prev_param).abs()
                                max_param_change = param_diff.max().item()
                                mean_param_change = param_diff.mean().item()
                                
                                if max_param_change > 1e-8:  # 有显著变化
                                    changed_params += 1
                                    max_change = max(max_change, max_param_change)
                                    total_change += mean_param_change
                                
                                # 更新保存的参数
                                self._prev_model_params[name] = param.data.clone()
                        
                        if total_params > 0:
                            param_change_info = {
                                "total_params": total_params,
                                "changed_params": changed_params,
                                "max_change": max_change,
                                "avg_change": total_change / changed_params if changed_params > 0 else 0.0,
                                "change_ratio": changed_params / total_params
                            }
                except Exception as e:
                    param_change_info = {"error": str(e)}
            
            # 打印参数变化信息（每5步或50步）
            if param_change_info and "error" not in param_change_info:
                if self.training_stats["step"] < 10 or self.training_stats["step"] % 50 == 0:
                    self.logger.info(
                        f"📊 Step {self.training_stats['step'] + 1} - 参数更新: "
                        f"{param_change_info['changed_params']}/{param_change_info['total_params']} 个参数有变化, "
                        f"最大变化={param_change_info['max_change']:.8f}, "
                        f"平均变化={param_change_info['avg_change']:.8f}"
                    )
                    
                    # 如果参数完全没有变化，发出警告
                    if param_change_info['changed_params'] == 0 and self.training_stats["step"] >= 5:
                        self.logger.warning(
                            f"⚠️ Step {self.training_stats['step'] + 1} - 模型参数完全没有更新！"
                            f"这可能表明策略训练有问题（梯度为0或优化器未执行）。"
                        )
            
            # 🔍 使用hook获取的梯度范数（优先）
            grad_norm = grad_norm_from_hook
            
            # 更新统计信息（使用CPU副本）
            self._update_training_stats(stats, combined_rewards_cpu)
            
            # 记录到wandb
            if self.config.get("logging", {}).get("use_wandb", False):
                # 确保所有值都转换为Python标量
                def to_scalar(value, default=0):
                    if value is None:
                        return default
                    if isinstance(value, (np.ndarray, np.generic)):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    if isinstance(value, torch.Tensor):
                        return float(value.item() if hasattr(value, 'item') else float(value))
                    return float(value)
                
                # 🔍 正确获取loss值：不要用clipfrac冒充loss
                # 先尝试获取真正的loss值
                policy_loss_key = None
                value_loss_key = None
                for key in stats.keys():
                    if 'policy' in key and 'loss' in key and 'clip' not in key:
                        policy_loss_key = key
                    if 'value' in key and 'loss' in key and 'clip' not in key:
                        value_loss_key = key
                
                log_data = {
                    "step": self.training_stats["step"],
                    "mean_reward": mean_reward_for_stats,  # ✅ 使用已保存的均值
                    # 🔍 使用正确的键名获取loss（从日志中发现实际键名是ppo/loss/policy和ppo/loss/value）
                    "policy_loss": to_scalar(stats.get(policy_loss_key) if policy_loss_key else stats.get("ppo/loss/policy") or stats.get("ppo/policy/loss", 0)),
                    "value_loss": to_scalar(stats.get(value_loss_key) if value_loss_key else stats.get("ppo/loss/value") or stats.get("ppo/val/loss", 0)),
                    "kl_divergence": to_scalar(stats.get("ppo/policy/kl") or stats.get("objective/kl") or 0),
                    "policy_clipfrac": to_scalar(stats.get("ppo/policy/clipfrac") or 0),  # ✅ Clip率
                    "value_clipfrac": to_scalar(stats.get("ppo/val/clipfrac") or 0),  # ✅ 价值函数Clip率
                    "objective/clipfrac": to_scalar(stats.get("objective/clipfrac") or 0),
                    "objective/entropy": to_scalar(stats.get("objective/entropy") or stats.get("ppo/policy/entropy") or 0),
                }
                
                # 添加梯度信息（如果有）
                if grad_norm is not None:
                    log_data["grad_norm"] = grad_norm
                else:
                    # 尝试从stats中获取（某些PPO实现可能会记录梯度范数）
                    if isinstance(stats, dict):
                        grad_norm_from_stats = stats.get("ppo/grad_norm") or stats.get("train/grad_norm") or stats.get("grad_norm")
                        if grad_norm_from_stats is not None:
                            log_data["grad_norm"] = to_scalar(grad_norm_from_stats)
                
                # 添加参数变化信息（如果有）
                if param_change_info and "error" not in param_change_info:
                    log_data.update({
                        "param_change/ratio": param_change_info.get("change_ratio", 0.0),
                        "param_change/changed_count": param_change_info.get("changed_params", 0),
                        "param_change/total_count": param_change_info.get("total_params", 0),
                        "param_change/max": param_change_info.get("max_change", 0.0),
                        "param_change/avg": param_change_info.get("avg_change", 0.0),
                    })
                
                # 🔍 诊断：如果loss为0，打印所有stats键帮助调试
                if log_data["policy_loss"] == 0 and log_data["value_loss"] == 0:
                    self.logger.warning(f"⚠️ PPO losses are zero! Available stats keys: {list(stats.keys())}")
                
                # 添加自适应权重信息
                if self.config["reward"].get("use_adaptive_weights", True):
                    weight_stats = self.reward_combiner.get_statistics()
                    if "adaptive_weights" in weight_stats:
                        log_data.update({
                            "adaptive_weight_intrinsic": float(weight_stats["adaptive_weights"].get("intrinsic", 0.0)),
                            "adaptive_weight_correctness": float(weight_stats["adaptive_weights"].get("correctness", 0.0))
                        })
                    if "weight_performance" in weight_stats:
                        log_data.update({
                            "weight_performance_intrinsic": float(weight_stats["weight_performance"].get("intrinsic", 0.0)),
                            "weight_performance_correctness": float(weight_stats["weight_performance"].get("correctness", 0.0))
                        })
                    
                    # 添加权重变化趋势
                    if "weight_trend" in weight_stats and weight_stats["weight_trend"]:
                        log_data.update({
                            "weight_trend_intrinsic": float(weight_stats["weight_trend"].get("intrinsic", 0.0)),
                            "weight_trend_correctness": float(weight_stats["weight_trend"].get("correctness", 0.0))
                        })
                
                wandb.log(log_data)
            
            self.training_stats["step"] += 1
            
            # 🔥 每步结束前清理显存，避免累积导致OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"训练步骤失败: {e}")
            raise
    
    def _update_training_stats(self, stats: Dict, rewards: torch.Tensor):
        """更新训练统计信息"""
        # 检查stats是否为None
        if stats is None:
            self.logger.warning("stats为None，使用默认值更新训练统计")
            stats = {}
        
        # 🔍 确保rewards是tensor，然后转换为Python标量
        if isinstance(rewards, torch.Tensor):
            mean_reward = torch.mean(rewards).item()
        elif isinstance(rewards, (list, tuple)):
            # 如果是列表，转换为tensor再计算
            mean_reward = float(np.mean([float(r.item() if hasattr(r, 'item') else float(r)) for r in rewards]))
        else:
            mean_reward = float(rewards) if isinstance(rewards, (int, float)) else 0.0
        
        self.training_stats["total_rewards"].append(mean_reward)
        
        # 确保从stats中获取的值转换为Python标量
        def to_scalar(value, default=0):
            if value is None:
                return default
            if isinstance(value, (np.ndarray, np.generic)):
                return float(value.item() if hasattr(value, 'item') else float(value))
            if isinstance(value, torch.Tensor):
                return float(value.item() if hasattr(value, 'item') else float(value))
            if isinstance(value, (np.int64, np.int32)):
                return int(value)
            if isinstance(value, (np.float64, np.float32)):
                return float(value)
            return float(value)
        
        # 🔍 修复：使用正确的指标名称，不应用clipfrac冒充loss
        # TRL可能使用不同的键名：ppo/policy/loss, objective/policy_loss等
        # 先找到真正的loss键
        policy_loss_key = None
        value_loss_key = None
        if isinstance(stats, dict):
            # 🔍 诊断：打印所有可用的键（只在第一步）
            if len(self.training_stats["policy_losses"]) == 0:
                self.logger.info(f"🔍 第1步stats字典的所有键: {list(stats.keys())}")
            
            # 尝试多种可能的键名（根据实际日志，TRL使用ppo/loss/policy格式）
            possible_policy_keys = [
                'ppo/loss/policy',  # ✅ 实际键名（从日志中发现）
                'ppo/policy/loss', 'objective/policy_loss', 'policy_loss',
                'ppo/policy/clipped_objective', 'objective/clipped_surrogate',
                'train/policy/loss', 'loss/policy'
            ]
            possible_value_keys = [
                'ppo/loss/value',  # ✅ 实际键名（从日志中发现）
                'ppo/value/loss', 'ppo/val/loss', 'value_loss',
                'objective/value_loss', 'train/value/loss', 'loss/value'
            ]
            
            # 先尝试精确匹配
            for key in stats.keys():
                key_lower = key.lower()
                if 'policy' in key_lower and 'loss' in key_lower and 'clip' not in key_lower:
                    policy_loss_key = key
                if 'value' in key_lower and 'loss' in key_lower and 'clip' not in key_lower:
                    value_loss_key = key
            
            # 如果精确匹配失败，尝试可能的键名
            if policy_loss_key is None:
                for possible_key in possible_policy_keys:
                    if possible_key in stats:
                        policy_loss_key = possible_key
                        break
            
            if value_loss_key is None:
                for possible_key in possible_value_keys:
                    if possible_key in stats:
                        value_loss_key = possible_key
                        break
        
        # 🔍 诊断：如果找不到policy loss键，记录警告并打印所有键
        if policy_loss_key is None and isinstance(stats, dict):
            if len(self.training_stats["policy_losses"]) < 10:  # 前10步警告
                self.logger.warning(f"⚠️ 无法找到policy_loss键！")
                self.logger.warning(f"   可用键: {list(stats.keys())}")
                self.logger.warning(f"   尝试的键名: {possible_policy_keys}")
                # 尝试查找任何包含'loss'的键
                loss_keys = [k for k in stats.keys() if 'loss' in k.lower()]
                if loss_keys:
                    self.logger.warning(f"   包含'loss'的键: {loss_keys}")
        
        if value_loss_key is None and isinstance(stats, dict):
            if len(self.training_stats["value_losses"]) < 10:  # 前10步警告
                self.logger.warning(f"⚠️ 无法找到value_loss键！")
                self.logger.warning(f"   可用键: {list(stats.keys())}")
        
        # 尝试获取损失值，如果找不到就使用默认值0，但记录警告
        if isinstance(stats, dict):
            if policy_loss_key:
                policy_loss_value = to_scalar(stats.get(policy_loss_key), 0)
            else:
                # 尝试最后一个备用方案：直接查找所有可能的值
                policy_loss_value = to_scalar(
                    stats.get("ppo/loss/policy") or  # ✅ 实际键名（从日志中发现）
                    stats.get("ppo/policy/loss") or 
                    stats.get("objective/policy_loss") or 
                    stats.get("policy_loss") or 0
                )
                if policy_loss_value == 0 and len(self.training_stats["policy_losses"]) < 5:
                    # 检查是否所有损失相关的键都是0或者不存在
                    all_loss_zero = True
                    for key in stats.keys():
                        if 'loss' in key.lower() and to_scalar(stats.get(key), -1) != 0:
                            all_loss_zero = False
                            break
                    if all_loss_zero:
                        self.logger.error(f"❌ 所有损失相关的键都为0或不存在！")
                        self.logger.error(f"   这可能是PPO训练未正确执行！")
            
            if value_loss_key:
                value_loss_value = to_scalar(stats.get(value_loss_key), 0)
            else:
                value_loss_value = to_scalar(
                    stats.get("ppo/loss/value") or  # ✅ 实际键名（从日志中发现）
                    stats.get("ppo/value/loss") or 
                    stats.get("ppo/val/loss") or 
                    stats.get("value_loss") or 0
                )
        else:
            policy_loss_value = 0
            value_loss_value = 0
        
        # 🔍 确保所有值都是Python标量（防止JSON序列化错误）
        self.training_stats["policy_losses"].append(float(policy_loss_value))
        self.training_stats["value_losses"].append(float(value_loss_value))
        kl_value = to_scalar(stats.get("ppo/policy/kl") or stats.get("objective/kl") or 0 if isinstance(stats, dict) else 0)
        self.training_stats["kl_divergences"].append(float(kl_value))
        
        # 保持最近1000步的统计
        max_history = 1000
        for key in self.training_stats:
            if isinstance(self.training_stats[key], list) and len(self.training_stats[key]) > max_history:
                self.training_stats[key] = self.training_stats[key][-max_history:]
        
        # 定期清理内存
        current_step = len(self.training_stats["total_rewards"])
        self._cleanup_memory(current_step)
    
    def train(self, train_dataset, max_steps: Optional[int] = None):
        """开始训练（支持并行数据加载）"""
        try:
            max_steps = max_steps or self.config["training"]["max_steps"]
            
            # 检查是否从检查点恢复
            current_step = self.training_stats.get("step", 0)
            start_step = current_step
            remaining_steps = max_steps - current_step
            
            if current_step > 0:
                self.logger.info(f"从检查点恢复训练，当前步数: {current_step}")
                self.logger.info(f"剩余训练步数: {remaining_steps}")
            else:
                self.logger.info(f"开始新的RL训练，最大步数: {max_steps}")
            
            # 设置并行数据加载器
            use_parallel = self._use_parallel
            if use_parallel and self.config.get("parallel", {}).get("use_parallel_data_loader", True):
                batch_size = self.config["ppo"]["batch_size"]
                num_workers = self.config.get("parallel", {}).get("data_loader_workers", 4)
                self.parallel_data_loader = ParallelDataLoader(
                    train_dataset, batch_size, num_workers, shuffle=True
                )
                self.logger.info("并行数据加载器已启用")
            
            # 创建进度条
            progress_bar = tqdm(
                range(start_step, max_steps), 
                initial=current_step,
                desc="RL训练进度", 
                unit="step",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            # 训练统计
            start_time = time.time()
            step_times = []
            
            for step in progress_bar:
                step_start_time = time.time()
                
                # ⭐⭐⭐⭐☆ 每2步强制清理内存（更频繁）
                if step > 0 and step % self._force_cleanup_every_n_steps == 0:
                    self._cleanup_memory(step, force=True)
                
                # 创建批次数据
                batch = None  # ✅ 初始化batch变量，避免UnboundLocalError
                try:
                    batch = self._create_batch(train_dataset)
                    
                    # 执行训练步骤
                    stats = self.train_step(batch)
                except torch.cuda.OutOfMemoryError as e:
                    # 🔥 OOM错误处理：清理内存
                    self.logger.error(f"❌ Step {step + 1} OOM错误，清理内存...")
                    self._cleanup_memory(step, force=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    raise RuntimeError(f"OOM错误，建议进一步减小batch_size或max_length: {e}")
                except RuntimeError as e:
                    error_str = str(e)
                    # 🔍 CUDA device-side assert错误处理
                    if "device-side assert" in error_str or "CUDA error" in error_str:
                        self.logger.error(f"❌ Step {step + 1} CUDA device-side assert错误")
                        self.logger.error(f"   错误信息: {error_str[:200]}")
                        # 清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        # 🔍 关键：CUDA错误可能损坏模型状态，需要跳过这一步
                        self.logger.warning(f"   跳过此训练步骤，继续训练...")
                        stats = None  # 标记为失败
                    else:
                        # 其他RuntimeError直接抛出
                        raise
                finally:
                    # ✅ 修复：安全清理batch，检查是否存在
                    if batch is not None:
                        try:
                            del batch
                        except:
                            pass
                    
                    # 定期清理内存
                    if step % self._memory_cleanup_interval == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # 检查train_step是否成功（可能返回None）
                if stats is None:
                    self.logger.warning(f"训练步骤 {step + 1} 失败，跳过本次更新")
                    # 计算步骤时间
                    step_time = time.time() - step_start_time
                    step_times.append(step_time)
                    # 使用空的stats字典继续
                    stats = {}
                    continue
                
                # 计算步骤时间
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                
                # 更新进度条信息
                avg_reward = float(np.mean(self.training_stats["total_rewards"][-10:])) if self.training_stats["total_rewards"] else 0.0
                avg_step_time = float(np.mean(step_times[-10:])) if step_times and len(step_times) > 0 else 0.0
                progress_bar.set_postfix({
                    'avg_reward': f'{avg_reward:.4f}',
                    'step_time': f'{step_time:.2f}s',
                    'avg_step_time': f'{avg_step_time:.2f}s'
                })
                
                # 定期保存和评估
                if (step + 1) % self.config["training"]["save_steps"] == 0:
                    try:
                        self.save_checkpoint(step + 1)
                        progress_bar.write(f"✅ 检查点已保存: step {step + 1}")
                    except Exception as e:
                        # 🔍 关键修复：检查点保存失败不应该中断训练
                        self.logger.error(f"❌ Step {step + 1} 检查点保存失败: {e}")
                        self.logger.error(f"   训练将继续，但此检查点可能不完整或丢失")
                        import traceback
                        self.logger.error(f"   详细错误: {traceback.format_exc()}")
                        # 不抛出异常，继续训练
                    
                    # 保存自适应权重
                    if self.config["reward"].get("use_adaptive_weights", True):
                        weight_file = f"{self.config['ppo']['output_dir']}/adaptive_weights_step_{step + 1}.json"
                        self.reward_combiner.save_weights(weight_file)
                
                if (step + 1) % self.config["training"]["eval_steps"] == 0:
                    self.evaluate_model()
                    progress_bar.write(f"📊 模型评估完成: step {step + 1}")
                
                # 日志输出
                if (step + 1) % self.config["training"]["logging_steps"] == 0:
                    # 确保stats不是None
                    if stats is not None:
                        self._log_training_progress(step + 1, stats)
                    else:
                        self.logger.warning(f"步骤 {step + 1} 的stats为None，跳过日志记录")
                    progress_bar.write(f"📝 训练日志: step {step + 1}")
                    
                    # 输出自适应权重状态
                    if self.config["reward"].get("use_adaptive_weights", True):
                        weight_stats = self.reward_combiner.get_statistics()
                        if "adaptive_weights" in weight_stats:
                            intrinsic_weight = float(weight_stats['adaptive_weights'].get('intrinsic', 0.0))
                            correctness_weight = float(weight_stats['adaptive_weights'].get('correctness', 0.0))
                            progress_bar.write(f"🎯 自适应权重: 内在={intrinsic_weight:.4f}, "
                                             f"正确性={correctness_weight:.4f}")
            
            # 关闭进度条
            progress_bar.close()
            
            # 保存最终模型
            final_model_dir = self.config['ppo']['output_dir']
            self.save_final_model(final_model_dir)
            
            # 训练完成统计
            total_time = time.time() - start_time
            avg_step_time = float(np.mean(step_times)) if step_times and len(step_times) > 0 else 0.0
            
            self.logger.info("🎉 RL训练完成!")
            self.logger.info(f"⏱️  总训练时间: {total_time:.2f}秒")
            self.logger.info(f"⚡ 平均每步时间: {float(avg_step_time):.2f}秒")
            final_avg_reward = float(np.mean(self.training_stats['total_rewards'][-10:])) if self.training_stats['total_rewards'] else 0.0
            self.logger.info(f"📈 最终平均奖励: {final_avg_reward:.4f}")
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
    
    def _create_batch(self, dataset) -> Dict[str, List[str]]:
        """创建批次数据（支持并行数据加载）"""
        batch_size = self.config["ppo"]["batch_size"]
        
        # 检查数据集是否为空
        if len(dataset) == 0:
            raise ValueError("数据集为空，无法创建批次")
        
        # 如果批次大小大于数据集大小，使用数据集大小并允许重复采样
        actual_batch_size = min(batch_size, len(dataset))
        replace = actual_batch_size > len(dataset)
        
        # 使用并行数据加载器或直接随机选择
        if self.parallel_data_loader:
            # 使用并行数据加载器
            try:
                batch_data = next(iter(self.parallel_data_loader))
                questions = [item["question"] for item in batch_data]
            except StopIteration:
                # 如果数据加载器为空，回退到随机选择
                self.logger.warning("并行数据加载器为空，回退到随机选择")
                indices = np.random.choice(len(dataset), size=actual_batch_size, replace=replace)
                questions = [dataset[int(i)]["question"] for i in indices]
        else:
            # 随机选择样本
            indices = np.random.choice(len(dataset), size=actual_batch_size, replace=replace)
            # 将 numpy.int64 转换为 Python int，避免 TypeError
            questions = [dataset[int(i)]["question"] for i in indices]
        
        return {"questions": questions}
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = f"{self.config['ppo']['output_dir']}/checkpoint-{step}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        save_success = True
        failed_parts = []
        
        # 保存学生模型
        try:
            self.student_model.save_model(checkpoint_dir)
            self.logger.debug("✓ 学生模型已保存")
        except Exception as e:
            self.logger.error(f"❌ 学生模型保存失败: {e}")
            save_success = False
            failed_parts.append("student_model")
        
        # 保存缓存
        try:
            cache_file = os.path.join(checkpoint_dir, "teacher_cache.pkl")
            self.cache_manager.save_cache(cache_file)
            self.logger.debug("✓ 教师缓存已保存")
        except Exception as e:
            self.logger.error(f"❌ 教师缓存保存失败: {e}")
            # 缓存失败不视为严重错误，只记录警告
            failed_parts.append("teacher_cache")
        
        # 保存训练统计
        try:
            stats_file = os.path.join(checkpoint_dir, "training_stats.json")
            import json
            
            # 🔍 确保所有值都是JSON可序列化的
            def make_json_serializable(obj):
                """递归地将对象转换为JSON可序列化格式"""
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.ndarray, np.generic)):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, torch.Tensor):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    # 尝试转换为字符串（最后的备选方案）
                    try:
                        return str(obj)
                    except:
                        return None
            
            # 创建可序列化的训练统计副本
            serializable_stats = make_json_serializable(self.training_stats)
            
            # 🔍 验证保存前的数据完整性
            if not isinstance(serializable_stats.get("step"), (int, float)):
                self.logger.warning(f"⚠️ 'step'类型异常: {type(serializable_stats.get('step'))}，修复为int")
                serializable_stats["step"] = int(self.training_stats.get("step", 0))
            
            # 验证列表长度一致性（用于调试）
            list_keys = ["total_rewards", "policy_losses", "value_losses", "kl_divergences"]
            list_lengths = {key: len(serializable_stats.get(key, [])) for key in list_keys}
            if len(set(list_lengths.values())) > 1:
                self.logger.warning(f"⚠️ 训练统计列表长度不一致: {list_lengths}")
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            # 验证文件已保存
            if os.path.exists(stats_file):
                file_size = os.path.getsize(stats_file) / 1024  # KB
                self.logger.debug(f"✓ 训练统计已保存 ({file_size:.2f} KB)")
            else:
                raise FileNotFoundError(f"训练统计文件保存后不存在: {stats_file}")
        except TypeError as e:
            self.logger.error(f"❌ 训练统计JSON序列化失败（类型错误）: {e}")
            self.logger.error(f"   这可能是由于训练统计中包含不可序列化的类型（如torch.Tensor）")
            # 尝试保存一个简化版本
            try:
                simple_stats = {
                    "step": int(self.training_stats.get("step", 0)),
                    "total_rewards_count": len(self.training_stats.get("total_rewards", [])),
                    "policy_losses_count": len(self.training_stats.get("policy_losses", [])),
                    "value_losses_count": len(self.training_stats.get("value_losses", [])),
                    "kl_divergences_count": len(self.training_stats.get("kl_divergences", [])),
                    "last_10_rewards": [float(r) if isinstance(r, (int, float)) else 0.0 
                                      for r in self.training_stats.get("total_rewards", [])[-10:]],
                    "last_10_policy_losses": [float(l) if isinstance(l, (int, float)) else 0.0 
                                            for l in self.training_stats.get("policy_losses", [])[-10:]],
                    "note": "完整统计序列化失败，仅保存摘要信息"
                }
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(simple_stats, f, indent=2, ensure_ascii=False)
                self.logger.warning(f"⚠️ 已保存简化版训练统计（摘要信息）")
            except Exception as e2:
                self.logger.error(f"❌ 保存简化版训练统计也失败: {e2}")
                save_success = False
                failed_parts.append("training_stats")
        except Exception as e:
            self.logger.error(f"❌ 训练统计保存失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            save_success = False
            failed_parts.append("training_stats")
        
        # 总结保存结果
        if save_success:
            if failed_parts:
                self.logger.warning(f"⚠️ 检查点已部分保存: {checkpoint_dir} (失败部分: {failed_parts})")
            else:
                self.logger.info(f"✓ 检查点已保存: {checkpoint_dir}")
        else:
            self.logger.error(f"❌ 检查点保存失败: {checkpoint_dir} (失败部分: {failed_parts})")
            raise RuntimeError(f"检查点保存失败，关键组件未保存: {failed_parts}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """从检查点恢复训练"""
        try:
            import json
            import pickle
            
            if not Path(checkpoint_dir).exists():
                raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
            
            self.logger.info(f"正在从检查点恢复: {checkpoint_dir}")
            
            # 加载模型（需要先初始化模型）
            if self.student_model is None:
                raise ValueError("模型未初始化，无法加载检查点。请先调用 setup_models()")
            
            # 加载学生模型权重
            self.logger.info("加载学生模型权重...")
            self.student_model.load_model(checkpoint_dir, load_adapter=True)
            
            # 重新设置PPO模型（因为模型权重已更新）
            self.logger.info("重新设置PPO模型...")
            self.ppo_model = self.student_model.setup_for_ppo()
            
            # 重新设置PPO训练器
            self.logger.info("重新设置PPO训练器...")
            self.setup_ppo_trainer()
            
            # 加载缓存
            cache_file = os.path.join(checkpoint_dir, "teacher_cache.pkl")
            if os.path.exists(cache_file):
                self.logger.info("加载缓存...")
                self.cache_manager.load_cache(cache_file)
            
            # 加载训练统计
            stats_file = os.path.join(checkpoint_dir, "training_stats.json")
            if os.path.exists(stats_file):
                self.logger.info("加载训练统计...")
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        loaded_stats = json.load(f)
                    
                    # 🔍 验证加载的统计数据的完整性
                    required_keys = ["step", "total_rewards", "policy_losses", "value_losses", "kl_divergences"]
                    missing_keys = [key for key in required_keys if key not in loaded_stats]
                    if missing_keys:
                        self.logger.warning(f"⚠️ 加载的训练统计缺少键: {missing_keys}，将使用默认值")
                        # 补充缺失的键
                        for key in missing_keys:
                            if key == "step":
                                loaded_stats[key] = 0
                            else:
                                loaded_stats[key] = []
                    
                    # 验证数据类型
                    if not isinstance(loaded_stats.get("step"), int):
                        self.logger.warning(f"⚠️ 'step'类型不正确: {type(loaded_stats.get('step'))}，转换为int")
                        loaded_stats["step"] = int(loaded_stats.get("step", 0))
                    
                    for key in ["total_rewards", "policy_losses", "value_losses", "kl_divergences"]:
                        if key in loaded_stats and not isinstance(loaded_stats[key], list):
                            self.logger.warning(f"⚠️ '{key}'类型不正确: {type(loaded_stats[key])}，转换为列表")
                            loaded_stats[key] = []
                    
                    self.training_stats = loaded_stats
                    self.logger.info(f"✓ 训练统计加载成功")
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ 训练统计JSON解析失败: {e}")
                    self.logger.warning("将使用空的训练统计，从步数0开始")
                    # 不抛出异常，继续训练
                except Exception as e:
                    self.logger.error(f"❌ 加载训练统计失败: {e}")
                    import traceback
                    self.logger.error(f"详细错误: {traceback.format_exc()}")
                    self.logger.warning("将使用空的训练统计，从步数0开始")
                    # 不抛出异常，继续训练
            
            self.logger.info(f"✅ 成功从检查点恢复")
            # 🔍 确保step是int类型
            current_step = int(self.training_stats.get('step', 0))
            self.training_stats['step'] = current_step
            self.logger.info(f"   当前步数: {current_step}")
            
            # 安全地计算平均奖励（避免列表为空或类型错误）
            rewards_list = self.training_stats.get('total_rewards', [])
            if rewards_list and len(rewards_list) > 0:
                try:
                    # 确保所有值都是数值类型
                    numeric_rewards = [float(r) for r in rewards_list[-100:] if isinstance(r, (int, float))]
                    if numeric_rewards:
                        avg_reward = np.mean(numeric_rewards)
                        self.logger.info(f"   平均奖励: {avg_reward:.4f}")
                    else:
                        self.logger.warning("   平均奖励: 无有效奖励数据")
                except Exception as e:
                    self.logger.warning(f"   计算平均奖励失败: {e}")
            else:
                self.logger.warning("   平均奖励: 奖励列表为空")
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查点加载失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise
    
    def save_final_model(self, save_path: str):
        """保存最终训练完成的模型"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # 保存学生模型
            self.student_model.save_model(save_path)
            
            # 保存训练配置
            config_file = os.path.join(save_path, "training_config.yaml")
            import yaml
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # 保存最终训练统计
            final_stats_file = os.path.join(save_path, "final_training_stats.json")
            import json
            
            # 🔍 确保所有值都是JSON可序列化的（复用相同的函数）
            def make_json_serializable(obj):
                """递归地将对象转换为JSON可序列化格式"""
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.ndarray, np.generic)):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, torch.Tensor):
                    return float(obj.item() if hasattr(obj, 'item') else float(obj))
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    try:
                        return str(obj)
                    except:
                        return None
            
            # 创建可序列化的训练统计副本
            serializable_stats = make_json_serializable(self.training_stats)
            
            # 🔍 验证保存前的数据完整性
            if not isinstance(serializable_stats.get("step"), (int, float)):
                self.logger.warning(f"⚠️ 'step'类型异常: {type(serializable_stats.get('step'))}，修复为int")
                serializable_stats["step"] = int(self.training_stats.get("step", 0))
            
            with open(final_stats_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            # 验证文件已保存
            if os.path.exists(final_stats_file):
                file_size = os.path.getsize(final_stats_file) / 1024  # KB
                self.logger.info(f"✓ 最终训练统计已保存 ({file_size:.2f} KB)")
            else:
                raise FileNotFoundError(f"最终训练统计文件保存后不存在: {final_stats_file}")
            
            # 保存模型信息
            model_info = {
                "model_type": "RL_trained_student_model",
                "base_model": self.config["model"]["student_model_name"],
                "teacher_model": self.config["model"]["teacher_model_name"],
                "training_steps": self.training_stats["step"],
                "final_reward": self.training_stats["total_rewards"][-1] if self.training_stats["total_rewards"] else 0.0,
                "training_date": str(Path().cwd()),
                "config_summary": {
                    "ppo_learning_rate": self.config["ppo"]["learning_rate"],
                    "ppo_epochs": self.config["ppo"]["ppo_epochs"],
                    "reward_lambda_intrinsic": self.config["reward"]["lambda_intrinsic"],
                    "reward_lambda_correctness": self.config["reward"]["lambda_correctness"]
                }
            }
            
            model_info_file = os.path.join(save_path, "model_info.json")
            with open(model_info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"最终模型已保存到: {save_path}")
            self.logger.info(f"模型信息: {model_info}")
            
        except Exception as e:
            self.logger.error(f"最终模型保存失败: {e}")
            raise
    
    def evaluate_model(self):
        """评估模型"""
        try:
            # 这里可以实现具体的评估逻辑
            # 例如在验证集上测试准确率等
            
            avg_reward = np.mean(self.training_stats["total_rewards"][-100:]) if self.training_stats["total_rewards"] else 0.0
            
            self.logger.info(f"模型评估 - 平均奖励: {avg_reward:.4f}")
            
            if self.config.get("logging", {}).get("use_wandb", False):
                wandb.log({
                    "eval/avg_reward": avg_reward,
                    "eval/step": self.training_stats["step"]
                })
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
    
    def _log_training_progress(self, step: int, stats: Dict):
        """记录训练进度"""
        # 检查stats是否为None
        if stats is None:
            self.logger.warning(f"步骤 {step} 的stats为None，无法记录训练进度")
            return
        
        # 确保转换为Python标量，避免numpy数组格式化错误
        avg_reward = float(np.mean(self.training_stats["total_rewards"][-100:])) if self.training_stats["total_rewards"] else 0.0
        avg_kl = float(np.mean(self.training_stats["kl_divergences"][-100:])) if self.training_stats["kl_divergences"] else 0.0
        
        # 从stats中获取值并转换为Python标量
        # 🔍 正确获取loss值：不使用clipfrac冒充loss
        policy_loss = 0.0
        if isinstance(stats, dict):
            # 先尝试找到真正的loss键
            for key in stats.keys():
                if 'policy' in key and 'loss' in key and 'clip' not in key:
                    policy_loss = stats[key]
                    break
            else:
                # 如果找不到，尝试标准键名
                policy_loss = stats.get('ppo/policy/loss', 0)
            
            if isinstance(policy_loss, (np.ndarray, np.generic)):
                policy_loss = float(policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss))
            else:
                policy_loss = float(policy_loss)
        
        # 基础训练信息
        log_message = (
            f"Step {step}: "
            f"Avg Reward: {avg_reward:.4f}, "
            f"Avg KL: {avg_kl:.4f}, "
            f"Policy Loss: {policy_loss:.4f}"
        )
        
        # 如果启用了自适应权重，添加权重信息
        if self.config["reward"].get("use_adaptive_weights", False):
            current_weights = self.reward_combiner.get_current_weights()
            log_message += (
                f" | Weights - Intrinsic: {current_weights['intrinsic']:.3f}, "
                f"Correctness: {current_weights['correctness']:.3f}"
            )
            if current_weights.get('reasoning', 0) > 0:
                log_message += f", Reasoning: {current_weights['reasoning']:.3f}"
            if current_weights.get('format', 0) > 0:
                log_message += f", Format: {current_weights['format']:.3f}"
        
        self.logger.info(log_message)
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止异步缓存工作线程
            if self.async_cache_manager:
                self.async_cache_manager.stop_async_worker()
            
            # 清理缓存管理器
            if self.cache_manager:
                self.cache_manager.cleanup()
            
            # 清理并行处理器
            if self.parallel_processor:
                self.parallel_processor = None
            
            # 清理并行推理器
            if self.parallel_inference_student:
                self.parallel_inference_student = None
            if self.parallel_inference_teacher:
                self.parallel_inference_teacher = None
            
            # 清理并行数据加载器
            if self.parallel_data_loader:
                self.parallel_data_loader = None
            
            if self.config.get("logging", {}).get("use_wandb", False):
                wandb.finish()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    # 加载配置
    config = load_config("config/training_config.yaml")
    
    # 创建训练器
    trainer = RLTrainer(config)
    
    # 设置模型和组件
    trainer.setup_models()
    trainer.setup_components()
    trainer.setup_ppo_trainer()
    
    # 加载GSM8K数据集
    from datasets import load_dataset
    from data.gsm8k_processor import GSM8KProcessor
    
    print("正在加载GSM8K数据集...")
    try:
        # 加载GSM8K数据集
        gsm8k_dataset = load_dataset("gsm8k", "main")
        
        # 创建GSM8K处理器
        processor = GSM8KProcessor(trainer.student_model.tokenizer, max_length=config["ppo"]["max_length"])
        
        # 使用训练集作为训练数据
        dataset = gsm8k_dataset["train"]
        
        print(f"训练集大小: {len(dataset)}")
        
        # 验证数据集质量
        processor.validate_data(dataset, num_samples=3)
        
        # 训练
        trainer.train(dataset, max_steps=100)
        
    except Exception as e:
        print(f"加载GSM8K数据集失败: {e}")
        print("无法进行训练，请检查网络连接和依赖项")
        return
        
    finally:
        # 清理资源
        trainer.cleanup()


if __name__ == "__main__":
    main()



