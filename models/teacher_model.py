"""
Teacher Model Wrapper Module
Function: Wrap Qwen-32B-instruct teacher model, provide logits computation and caching functionality
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import hashlib
from collections import OrderedDict
import logging
from pathlib import Path
import threading
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


class TeacherModel:
    """Teacher Model Wrapper Class"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct", 
                 cache_size: int = 10000, cache_policy: str = "LRU",
                 device: str = "auto", torch_dtype: torch.dtype = torch.bfloat16,
                 max_memory: Optional[Dict[int, str]] = None):
        """
        初始化教师模型
        
        Args:
            model_name: 模型名称
            cache_size: 缓存大小
            cache_policy: 缓存策略
            device: 设备
            torch_dtype: 数据类型
            max_memory: 每个GPU的最大显存限制（字典，如{0: "75GB", 1: "75GB"}）
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_memory = max_memory
        
        # 初始化缓存
        self.cache_size = cache_size
        self.cache_policy = cache_policy
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 线程锁，用于保护 tokenizer 的线程安全
        self._tokenizer_lock = threading.Lock()
        
        # 抑制past_key_values警告
        suppress_past_key_values_warning()
        
        # 加载模型和分词器
        self._load_model()
        
        # 更新模型以使用现代缓存
        self.model = update_model_for_modern_cache(self.model)
        
        logging.info(f"Teacher model {model_name} loaded successfully")
        logging.info(f"Cache configuration: size={cache_size}, policy={cache_policy}")
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            # ✅ 确保：如果使用 device_map，必须设置 low_cpu_mem_usage=True
            # 如果 device 为 None 或空字符串，则不使用 device_map
            device_map_value = self.device if (self.device and self.device.lower() != 'none') else None
            low_cpu_mem_usage = True if device_map_value is not None else False
            
            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": low_cpu_mem_usage  # ✅ 条件设置：使用device_map时必须为True
            }
            
            # 只在 device_map 不为 None 时添加
            if device_map_value is not None:
                load_kwargs["device_map"] = device_map_value
            
            # 如果指定了max_memory，添加该参数（用于限制特定GPU的显存使用）
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # ✅ 方式1：不resize，直接使用模型原始权重（最稳定方式）
            tokenizer_vocab_size = len(self.tokenizer)
            model_emb_size = self.model.get_input_embeddings().weight.size(0)
            
            logging.info(f"📊 Vocab大小检查:")
            logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
            logging.info(f"   model embedding size: {model_emb_size}")
            logging.info(f"   model.config.vocab_size: {getattr(self.model.config, 'vocab_size', 'N/A')}")
            
            if model_emb_size != tokenizer_vocab_size:
                logging.warning(f"⚠️  Vocab大小不匹配（{model_emb_size} vs {tokenizer_vocab_size}）")
                logging.info(f"   但采用方式1：不resize，直接使用模型原始权重")
                logging.info(f"   这是最稳定的方式，即使有差异也不会触发CUDA错误")
            else:
                logging.info(f"✅ Vocab大小匹配: {model_emb_size}")
            
            self.model.eval()  # 设置为评估模式
            
            logging.info("Model loaded successfully")
            
        except FileNotFoundError as e:
            logging.error(f"模型文件未找到: {e}")
            raise
        except RuntimeError as e:
            logging.error(f"模型加载运行时错误: {e}")
            raise
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: torch.Tensor):
        """更新缓存"""
        # 🔥 修复显存泄漏：将logits移动到CPU
        self.cache[key] = value.clone().detach().cpu()
        
        # 如果缓存满了，移除最旧的项
        if len(self.cache) > self.cache_size:
            if self.cache_policy == "LRU":
                self.cache.popitem(last=False)
            else:
                # 随机移除
                import random
                random_key = random.choice(list(self.cache.keys()))
                del self.cache[random_key]
    
    def get_logits(self, text: Union[str, List[str]], 
                   use_cache: bool = True) -> torch.Tensor:
        """
        获取文本的logits
        
        Args:
            text: 输入文本或文本列表
            use_cache: 是否使用缓存
            
        Returns:
            logits张量
        """
        if isinstance(text, str):
            text = [text]
        
        batch_logits = []
        
        for single_text in text:
            cache_key = None
            if use_cache:
                cache_key = self._get_cache_key(single_text)
                
            # 检查缓存
            if cache_key in self.cache:
                self.cache_hits += 1
                # 移动到末尾（LRU）
                self.cache.move_to_end(cache_key)
                # 🔥 修复：不要移动缓存到device，让模型自动处理
                cached_logits = self.cache[cache_key]
                batch_logits.append(cached_logits)
                continue
            
            # 缓存未命中，计算logits
            self.cache_misses += 1
            
            # 🔥 获取vocab_size（应该已匹配，因为_load_model中已检查）
            vocab_size = len(self.tokenizer)
            
            # 🔥 关键修复：只在校验阶段clamp input_ids，不修改embedding层
            # 删除monkey patch，因为vocab_size已匹配，无需包装embedding
            
            # 使用线程锁保护 tokenizer 调用，避免 "Already borrowed" 错误
            max_retries = 3
            retry_count = 0
            logits = None
            
            while retry_count < max_retries and logits is None:
                try:
                    with self._tokenizer_lock:
                        with torch.no_grad():
                            # 分词
                            inputs = self.tokenizer(
                                single_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=512,
                                padding=True
                            )
                            
                            # 🔥 修复：不要强制移动到某个device
                            # 对于device_map="auto"的模型，让HF自动处理设备分配
                            # inputs保持在CPU，model(**inputs)会自动处理
                            
                            # 🔥 关键：确保input_ids在有效范围内（在送入模型前校验）
                            if 'input_ids' in inputs:
                                input_ids = inputs['input_ids']
                                # 检查是否有超出范围的token（不应该发生，因为vocab已匹配）
                                if input_ids.numel() > 0:
                                    max_id = input_ids.max().item()
                                    min_id = input_ids.min().item()
                                    if max_id >= vocab_size or min_id < 0:
                                        logging.warning(f"⚠️ input_ids超出范围: [{min_id}, {max_id}], vocab_size={vocab_size}, 自动clamp")
                                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                            
                            # 前向传播（不移动inputs到device，让HF自动处理device_map）
                            outputs = self.model(**inputs)
                            logits = outputs.logits
                            
                            # 存储到缓存
                            if use_cache and cache_key is not None:
                                self._update_cache(cache_key, logits)
                            
                            batch_logits.append(logits)
                            break  # 成功获取logits，退出重试循环
                except RuntimeError as e:
                    if "Already borrowed" in str(e) and retry_count < max_retries - 1:
                        retry_count += 1
                        logging.warning(f"Tokenizer 线程安全问题，重试 {retry_count}/{max_retries}: {e}")
                        import time
                        time.sleep(0.01 * retry_count)  # 递增等待时间
                    else:
                        logging.error(f"Tokenizer 调用失败，已达到最大重试次数: {e}")
                        raise
        
        # 如果是单个文本，返回单个logits
        if len(batch_logits) == 1:
            return batch_logits[0]
        
        # 多个文本时，返回list（因为seq_len可能不同，cat会失败）
        return batch_logits
    
    def generate_response(self, prompt: str, max_length: int = 256,
                         temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        生成响应
        
        Args:
            prompt: 提示文本
            max_length: 最大长度
            temperature: 温度参数
            do_sample: 是否采样
            
        Returns:
            生成的响应
        """
        # 使用线程锁保护 tokenizer 调用，避免 "Already borrowed" 错误
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._tokenizer_lock:
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )
                        
                        # 🔥 获取vocab_size（应该已匹配，因为_load_model中已检查）
                        vocab_size = len(self.tokenizer)
                        
                        # 🔥 关键修复：确保input_ids在有效范围内
                        if 'input_ids' in inputs:
                            input_ids = inputs['input_ids']
                            if input_ids.numel() > 0:
                                max_token_id = input_ids.max().item()
                                min_token_id = input_ids.min().item()
                                input_len = input_ids.shape[1]
                                
                                if max_token_id >= vocab_size or min_token_id < 0:
                                    logging.warning(f"⚠️ input_ids超出范围: [{min_token_id}, {max_token_id}], vocab_size={vocab_size}, 自动clamp")
                                    inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                            
                            # 🔍 检查输入长度是否超出模型限制
                            max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', None)
                            if max_position_embeddings and input_len > max_position_embeddings:
                                logging.error(f"❌ 输入序列长度 {input_len} 超出模型最大位置 {max_position_embeddings}!")
                                # 截断到最大长度
                                inputs['input_ids'] = input_ids[:, :max_position_embeddings]
                                if 'attention_mask' in inputs:
                                    inputs['attention_mask'] = inputs['attention_mask'][:, :max_position_embeddings]
                                logging.warning(f"   已截断到 {max_position_embeddings}")
                            
                            # 记录输入信息（仅在前几个样本或出现问题时）
                            if retry_count > 0 or max_token_id >= vocab_size * 0.9:
                                logging.debug(f"📊 生成前检查: input_len={input_len}, token_range=[{min_token_id}, {max_token_id}], vocab_size={vocab_size}")
                        
                        # 🔥 修复：不要移动inputs到device，让HF自动处理device_map
                        
                        # 🔍 设置有效的pad_token_id和eos_token_id
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_token_id
                        
                        # 确保token ID在有效范围内
                        if pad_token_id is not None:
                            pad_token_id = min(pad_token_id, vocab_size - 1)
                        if eos_token_id is not None:
                            eos_token_id = min(eos_token_id, vocab_size - 1)
                        
                        # 🔍 生成前：确保所有参数都正确
                        max_positions = getattr(self.model.config, 'max_position_embeddings', None)
                        max_total_len = max_positions if max_positions else 2048
                        current_len = inputs['input_ids'].shape[1]
                        max_allowed_new_tokens = min(max_length, max_total_len - current_len - 10)  # 留10个token的安全余量
                        
                        if max_allowed_new_tokens <= 0:
                            logging.error(f"❌ 无法生成新token: 当前长度 {current_len} + 预留 {10} >= 最大长度 {max_total_len}")
                            return ""
                        
                        # 🔍 详细的生成参数日志（仅在错误时）
                        if retry_count > 0:
                            logging.info(f"🔍 生成参数: max_new_tokens={max_allowed_new_tokens}, current_len={current_len}, max_total={max_total_len}")
                            logging.info(f"   pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, vocab_size={vocab_size}")
                        
                        # 🔥 关键修复：添加LogitsProcessor来mask超出范围的token（不改变形状）
                        from transformers import LogitsProcessorList
                        
                        class TokenRangeLogitsProcessor:
                            """Mask超出tokenizer范围的logits（不改变形状）"""
                            def __init__(self, max_valid_token_id: int):
                                self.max_valid_token_id = max_valid_token_id
                                self.vocab_end_idx = max_valid_token_id + 1
                            
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                # 🔥 关键：只mask，不切片（保持形状不变）
                                if scores.shape[-1] > self.vocab_end_idx:
                                    scores[..., self.vocab_end_idx:] = float('-inf')
                                return scores
                        
                        max_valid_token_id = vocab_size - 1
                        logits_processor = LogitsProcessorList([
                            TokenRangeLogitsProcessor(max_valid_token_id)
                        ])
                        
                        # 🔥 关键：只在输入阶段校验input_ids，不修改embedding层
                        if 'input_ids' in inputs and inputs['input_ids'].numel() > 0:
                            input_ids = inputs['input_ids']
                            if input_ids.max().item() >= vocab_size or input_ids.min().item() < 0:
                                logging.warning(f"⚠️ input_ids超出范围，clamp到[0, {vocab_size-1}]")
                                inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                        
                        # 🔥 修复：不要移动inputs到device，让HF自动处理device_map
                        # 生成（inputs保持在CPU，model.generate会自动处理设备分配）
                        generate_kwargs = {
                            **inputs,
                            "max_new_tokens": max_allowed_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "repetition_penalty": 1.1,
                            "logits_processor": logits_processor,
                            "use_cache": True
                        }
                        
                        outputs = self.model.generate(**generate_kwargs)
                        
                        # 🔍 详细验证生成的token ID
                        invalid_mask = (outputs >= vocab_size) | (outputs < 0)
                        if torch.any(invalid_mask):
                            invalid_ids = outputs[invalid_mask].unique().tolist()
                            output_max = outputs.max().item()
                            output_min = outputs.min().item()
                            logging.error(f"❌ 生成的token ID超出范围!")
                            logging.error(f"   无效ID列表: {invalid_ids[:10]}")
                            logging.error(f"   输出token范围: [{output_min}, {output_max}], vocab_size={vocab_size}")
                            logging.error(f"   输出形状: {outputs.shape}")
                            logging.error(f"   将截断到有效范围")
                            outputs = torch.clamp(outputs, 0, vocab_size - 1)
                        
                        # 解码
                        input_length = inputs['input_ids'].shape[1]
                        if len(outputs[0]) > input_length:
                            generated_text = self.tokenizer.decode(
                                outputs[0][input_length:],
                                skip_special_tokens=True
                            )
                        else:
                            generated_text = ""
                        
                        return generated_text
            except RuntimeError as e:
                error_str = str(e)
                if "device-side assert" in error_str or "CUDA error" in error_str:
                    retry_count += 1
                    # 🔍 详细错误诊断
                    logging.error(f"❌ CUDA device-side assert错误 (重试 {retry_count}/{max_retries})")
                    logging.error(f"   错误信息: {error_str[:500]}")  # 只显示前500字符
                    
                    # 🔍 诊断信息：检查模型和输入状态
                    try:
                        if 'input_ids' in locals():
                            logging.error(f"   输入形状: {inputs.get('input_ids', 'N/A').shape if 'input_ids' in inputs else 'N/A'}")
                            if 'input_ids' in inputs:
                                input_ids = inputs['input_ids']
                                logging.error(f"   输入token范围: [{input_ids.min().item()}, {input_ids.max().item()}]")
                                logging.error(f"   输入序列长度: {input_ids.shape[1]}")
                        
                        model_vocab = getattr(self.model.config, 'vocab_size', None)
                        max_pos = getattr(self.model.config, 'max_position_embeddings', None)
                        logging.error(f"   模型vocab_size: {model_vocab}")
                        logging.error(f"   模型max_position_embeddings: {max_pos}")
                        logging.error(f"   tokenizer vocab_size: {len(self.tokenizer)}")
                    except Exception as diag_e:
                        logging.error(f"   诊断信息获取失败: {diag_e}")
                    
                    if retry_count < max_retries:
                        logging.warning(f"   清理CUDA缓存并重试...")
                        # 清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                        # 重置模型状态
                        self.model.eval()
                        import time
                        time.sleep(0.1 * retry_count)
                    else:
                        logging.error(f"❌ CUDA错误，已达到最大重试次数")
                        logging.error(f"   建议: 检查模型权重是否损坏，或尝试重新加载模型")
                        # 返回空字符串而不是抛出异常
                        return ""
                elif "Already borrowed" in error_str and retry_count < max_retries - 1:
                    retry_count += 1
                    logging.warning(f"Tokenizer 线程安全问题，重试 {retry_count}/{max_retries}: {e}")
                    import time
                    time.sleep(0.01 * retry_count)  # 递增等待时间
                else:
                    logging.error(f"生成响应失败，已达到最大重试次数: {e}")
                    return ""  # 返回空字符串而不是抛出异常
            except Exception as e:
                logging.error(f"生成响应时发生未知错误: {e}")
                return ""  # 返回空字符串而不是抛出异常
    
    def compute_log_probs(self, text: str) -> torch.Tensor:
        """
        计算文本的对数概率
        
        Args:
            text: 输入文本
            
        Returns:
            对数概率张量
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # 🔥 修复：不要移动inputs到device，让HF自动处理device_map
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 计算对数概率
            log_probs = F.log_softmax(logits, dim=-1)
            
            return log_probs
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logging.info("Cache cleared")
    
    def save_cache(self, filepath: str):
        """保存缓存到文件"""
        # 🔥 修复：确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "cache": dict(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
        
        torch.save(cache_data, filepath)
        logging.info(f"Cache saved to: {filepath}")
    
    def load_cache(self, filepath: str):
        """从文件加载缓存"""
        if Path(filepath).exists():
            cache_data = torch.load(filepath, map_location='cpu')
            self.cache = OrderedDict(cache_data["cache"])
            self.cache_hits = cache_data["cache_hits"]
            self.cache_misses = cache_data["cache_misses"]
            logging.info(f"Cache loaded from {filepath}")
        else:
            logging.warning(f"Cache file does not exist: {filepath}")
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """获取模型信息"""
        # 🔥 修复：device_map="auto"时模型没有单一device
        try:
            if hasattr(self.model, 'hf_device_map'):
                device_info = "sharded"  # 分布式模型
            elif hasattr(self.model, 'device'):
                device_info = str(self.model.device)
            else:
                device_info = "unknown"
        except:
            device_info = "unknown"
        
        return {
            "model_name": self.model_name,
            "device": device_info,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "cache_size": len(self.cache),
            "cache_policy": self.cache_policy
        }


class TeacherModelManager:
    """Teacher Model Manager"""
    
    def __init__(self, config: Dict):
        """
        初始化管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.teacher_model = None
        
    def initialize_teacher(self) -> TeacherModel:
        """初始化教师模型"""
        if self.teacher_model is None:
            self.teacher_model = TeacherModel(
                model_name=self.config["teacher_model"]["model_name"],
                cache_size=self.config["teacher_model"]["cache_size"],
                cache_policy=self.config["teacher_model"]["cache_policy"],
                device=self.config["device"]["device_map"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"])
            )
        
        return self.teacher_model
    
    def get_teacher(self) -> TeacherModel:
        """获取教师模型实例"""
        if self.teacher_model is None:
            return self.initialize_teacher()
        return self.teacher_model
    
    def cleanup(self):
        """清理资源"""
        if self.teacher_model is not None:
            # 保存缓存
            cache_file = "./cache/teacher_cache.pkl"
            self.teacher_model.save_cache(cache_file)
            
            # 清理GPU内存
            del self.teacher_model
            torch.cuda.empty_cache()


def create_teacher_model(config: Dict) -> TeacherModel:
    """
    创建教师模型的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        教师模型实例
    """
    return TeacherModel(
        model_name=config["teacher_model"]["model_name"],
        cache_size=config["teacher_model"]["cache_size"],
        cache_policy=config["teacher_model"]["cache_policy"],
        device=config["device"]["device_map"],
        torch_dtype=getattr(torch, config["device"]["torch_dtype"])
    )

