"""
Student Model Wrapper Module
Function: Wrap Qwen-7B-math student model, support LoRA fine-tuning and PPO training
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import AutoModelForCausalLMWithValueHead
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


class StudentModel:
    """Student Model Wrapper Class"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Math",
                 lora_config: Optional[Dict] = None,
                 device: str = "auto", 
                 torch_dtype: torch.dtype = torch.bfloat16,
                 use_lora: bool = True):
        """
        初始化学生模型
        
        Args:
            model_name: 模型名称
            lora_config: LoRA配置
            device: 设备
            torch_dtype: 数据类型
            use_lora: 是否使用LoRA
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_lora = use_lora
        self.lora_config = lora_config or self._default_lora_config()
        
        # 抑制past_key_values警告
        suppress_past_key_values_warning()
        
        # 加载模型和分词器
        self._load_model()
        
        # 更新模型以使用现代缓存
        self.model = update_model_for_modern_cache(self.model)
        
        logging.info(f"Student model {model_name} loaded successfully")
        logging.info(f"LoRA configuration: {self.lora_config}")
    
    def _default_lora_config(self) -> Dict:
        """默认LoRA配置"""
        return {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            import os
            from pathlib import Path
            
            # 检查model_name是本地路径还是HuggingFace模型名
            model_path = Path(self.model_name)
            is_local_path = model_path.exists() and model_path.is_dir()
            
            # 检查是否是已训练的模型（包含adapter文件）
            is_trained_model = False
            if is_local_path:
                adapter_files = [
                    model_path / "adapter_model.bin",
                    model_path / "adapter_model.safetensors",
                    model_path / "adapter_config.json"
                ]
                is_trained_model = any(f.exists() for f in adapter_files[:2])  # 检查权重文件
                
                if is_trained_model:
                    logging.info(f"检测到已训练的模型目录: {self.model_name}")
                    logging.info("将加载已训练的LoRA适配器")
            
            if is_trained_model and self.use_lora:
                # 情况1: 加载已训练的LoRA适配器
                # 首先需要加载基础模型（从配置或模型目录的父目录获取）
                base_model_name = None
                
                # 检查是否有adapter_config.json说明基础模型
                config_path = model_path / "adapter_config.json"
                if config_path.exists():
                    import json
                    with open(config_path, 'r') as f:
                        adapter_config = json.load(f)
                        # PEFT适配器配置中包含基础模型路径
                        base_model_name = adapter_config.get("base_model_name_or_path", None)
                
                # 如果无法从适配器配置获取基础模型，使用默认基础模型
                if base_model_name is None:
                    # 使用配置文件中的基础模型名，或默认值
                    default_base = "Qwen/Qwen2.5-7B-Instruct"
                    logging.info(f"从适配器配置未找到基础模型路径，使用默认: {default_base}")
                    base_model_name = default_base
                else:
                    # 检查base_model_name是否是本地路径且不存在
                    if Path(base_model_name).exists():
                        logging.info(f"从适配器配置找到基础模型路径: {base_model_name}")
                    else:
                        # 可能是HuggingFace模型名，直接使用
                        logging.info(f"从适配器配置找到基础模型: {base_model_name}")
                
                # ✅ 修复：先加载分词器（从基础模型或本地目录）
                # 如果本地目录有tokenizer.json，优先使用本地
                if (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists():
                    logging.info(f"从本地目录加载tokenizer: {self.model_name}")
                    tokenizer_path = self.model_name
                else:
                    logging.info(f"从基础模型加载tokenizer: {base_model_name}")
                    tokenizer_path = base_model_name
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # 设置pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # 加载基础模型
                # 注意：如果使用 device_map，必须设置 low_cpu_mem_usage=True
                # RL 训练需要模型完全在 GPU 上，不能使用 offload
                device_map_for_load = None if self.device == "auto" else self.device
                # ✅ 修复：如果使用 device_map，必须设置 low_cpu_mem_usage=True
                low_cpu_mem_usage = True if device_map_for_load is not None else False
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map_for_load,  # 如果为None则不使用device_map
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage  # ✅ 修复：使用device_map时必须为True
                )
                
                # 如果 device_map 是 None，需要手动移动到设备
                if device_map_for_load is None and torch.cuda.is_available():
                    self.base_model = self.base_model.to(torch.device("cuda:0"))
                
                # 🔥 关键修复：先基座、再LoRA - resize_token_embeddings必须在加载LoRA之前执行
                tokenizer_vocab_size = len(self.tokenizer)
                try:
                    # 获取基础模型的真实embedding大小
                    input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                    output_emb_size = None
                    if hasattr(self.base_model, 'get_output_embeddings') and self.base_model.get_output_embeddings() is not None:
                        output_emb_size = self.base_model.get_output_embeddings().weight.size(0)
                    
                    logging.info(f"📊 Embedding大小检查（加载LoRA前）:")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {input_emb_size}")
                    if output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {output_emb_size}")
                    logging.info(f"   model.config.vocab_size: {getattr(self.base_model.config, 'vocab_size', 'N/A')}")
                    
                    # 如果embedding大小与tokenizer不匹配，执行resize
                    if input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ 模型embedding大小 ({input_emb_size}) != tokenizer大小 ({tokenizer_vocab_size})")
                        logging.info(f"   正在resize_token_embeddings到 {tokenizer_vocab_size}...")
                        self.base_model.resize_token_embeddings(tokenizer_vocab_size)
                        logging.info(f"✅ resize_token_embeddings完成")
                        
                        # 验证resize是否成功
                        new_input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                        if new_input_emb_size != tokenizer_vocab_size:
                            logging.error(f"❌ resize_token_embeddings失败！新大小: {new_input_emb_size} != {tokenizer_vocab_size}")
                            logging.warning(f"   将保留'限域+clamp'的保险策略")
                        else:
                            logging.info(f"✅ resize成功验证: input_embeddings.size(0) = {new_input_emb_size}")
                    else:
                        logging.info(f"✅ embedding大小与tokenizer匹配，无需resize")
                    
                    # 🔥 关键修复：确保model.config中的vocab_size和pad/eos_token_id与tokenizer一致
                    self.base_model.config.vocab_size = tokenizer_vocab_size
                    self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
                    logging.info(f"✅ 已更新base_model.config: vocab_size={tokenizer_vocab_size}, pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
                        
                except Exception as e:
                    logging.warning(f"⚠️ resize_token_embeddings时出错（可能不支持或已量化）: {e}")
                    logging.warning(f"   将保留'限域+clamp'的保险策略")
                
                # 加载已训练的LoRA适配器
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    self.model_name,
                    torch_dtype=self.torch_dtype
                )
                logging.info(f"成功加载已训练的LoRA适配器从: {self.model_name}")
                
                # 🔥 关键：加载LoRA后再次验证embedding大小，如果仍不匹配则再次resize
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    final_output_emb_size = None
                    if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                        final_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                    logging.info(f"📊 Embedding大小检查（加载LoRA后）:")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {final_input_emb_size}")
                    if final_output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {final_output_emb_size}")
                    
                    # 🔥 关键修复：如果加载LoRA后embedding大小不匹配，再次resize
                    if final_input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ 加载LoRA后，input_embeddings大小 ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        logging.info(f"   正在再次resize_token_embeddings到 {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            # 验证resize是否成功
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                logging.info(f"✅ LoRA后resize成功: input_embeddings = {new_final_input_emb_size}")
                            else:
                                logging.error(f"❌ LoRA后resize失败: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e:
                            logging.warning(f"⚠️ LoRA后resize失败: {e}")
                            logging.warning(f"   将使用'限域+clamp'策略")
                    
                    # 🔥 关键：检查并resize output embeddings (lm_head)
                    if final_output_emb_size is not None and final_output_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ output_embeddings大小 ({final_output_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        # resize_token_embeddings应该同时resize input和output，但如果失败，手动检查
                        try:
                            # 检查resize后是否已修复
                            check_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                            if check_output_emb_size != tokenizer_vocab_size:
                                logging.warning(f"   output_embeddings仍不匹配，将使用'限域+clamp'策略")
                        except:
                            pass
                except Exception as e:
                    logging.warning(f"⚠️ 无法检查LoRA后的embedding大小: {e}")
                
            else:
                # 情况2: 加载基础模型并应用新的LoRA配置
                # 先加载分词器
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # 设置pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # 注意：如果使用 device_map，必须设置 low_cpu_mem_usage=True
                # RL 训练需要模型完全在 GPU 上，不能使用 offload
                device_map_for_load = None if self.device == "auto" else self.device
                # ✅ 修复：如果使用 device_map，必须设置 low_cpu_mem_usage=True
                low_cpu_mem_usage = True if device_map_for_load is not None else False
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map_for_load,  # 如果为None则不使用device_map
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage  # ✅ 修复：使用device_map时必须为True
                )
                
                # 如果 device_map 是 None，需要手动移动到设备
                if device_map_for_load is None and torch.cuda.is_available():
                    self.base_model = self.base_model.to(torch.device("cuda:0"))
                
                # 🔥 关键修复：先基座、再LoRA - resize_token_embeddings必须在加载LoRA之前执行
                tokenizer_vocab_size = len(self.tokenizer)
                try:
                    # 获取基础模型的真实embedding大小
                    input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                    output_emb_size = None
                    if hasattr(self.base_model, 'get_output_embeddings') and self.base_model.get_output_embeddings() is not None:
                        output_emb_size = self.base_model.get_output_embeddings().weight.size(0)
                    
                    logging.info(f"📊 Embedding大小检查（加载LoRA前）:")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {input_emb_size}")
                    if output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {output_emb_size}")
                    logging.info(f"   model.config.vocab_size: {getattr(self.base_model.config, 'vocab_size', 'N/A')}")
                    
                    # 如果embedding大小与tokenizer不匹配，执行resize
                    if input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ 模型embedding大小 ({input_emb_size}) != tokenizer大小 ({tokenizer_vocab_size})")
                        logging.info(f"   正在resize_token_embeddings到 {tokenizer_vocab_size}...")
                        self.base_model.resize_token_embeddings(tokenizer_vocab_size)
                        logging.info(f"✅ resize_token_embeddings完成")
                        
                        # 验证resize是否成功
                        new_input_emb_size = self.base_model.get_input_embeddings().weight.size(0)
                        if new_input_emb_size != tokenizer_vocab_size:
                            logging.error(f"❌ resize_token_embeddings失败！新大小: {new_input_emb_size} != {tokenizer_vocab_size}")
                            logging.warning(f"   将保留'限域+clamp'的保险策略")
                        else:
                            logging.info(f"✅ resize成功验证: input_embeddings.size(0) = {new_input_emb_size}")
                    else:
                        logging.info(f"✅ embedding大小与tokenizer匹配，无需resize")
                    
                    # 🔥 关键修复：确保model.config中的vocab_size和pad/eos_token_id与tokenizer一致
                    self.base_model.config.vocab_size = tokenizer_vocab_size
                    self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                    self.base_model.config.eos_token_id = self.tokenizer.eos_token_id
                    logging.info(f"✅ 已更新base_model.config: vocab_size={tokenizer_vocab_size}, pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
                        
                except Exception as e:
                    logging.warning(f"⚠️ resize_token_embeddings时出错（可能不支持或已量化）: {e}")
                    logging.warning(f"   将保留'限域+clamp'的保险策略")
                
                # 应用LoRA
                if self.use_lora:
                    peft_config = LoraConfig(**self.lora_config)
                    self.model = get_peft_model(self.base_model, peft_config)
                    logging.info("应用新的LoRA配置")
                else:
                    self.model = self.base_model
                
                # 🔥 关键：应用LoRA后再次验证embedding大小，如果仍不匹配则再次resize
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    final_output_emb_size = None
                    if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                        final_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                    logging.info(f"📊 Embedding大小检查（加载LoRA后）:")
                    logging.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                    logging.info(f"   model input_embeddings.size(0): {final_input_emb_size}")
                    if final_output_emb_size is not None:
                        logging.info(f"   model output_embeddings.size(0): {final_output_emb_size}")
                    
                    # 🔥 关键修复：如果应用LoRA后embedding大小不匹配，再次resize
                    if final_input_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ 应用LoRA后，input_embeddings大小 ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        logging.info(f"   正在再次resize_token_embeddings到 {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            # 验证resize是否成功
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                logging.info(f"✅ LoRA后resize成功: input_embeddings = {new_final_input_emb_size}")
                            else:
                                logging.error(f"❌ LoRA后resize失败: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e:
                            logging.warning(f"⚠️ LoRA后resize失败: {e}")
                            logging.warning(f"   将使用'限域+clamp'策略")
                    
                    # 🔥 关键：检查并resize output embeddings (lm_head)
                    if final_output_emb_size is not None and final_output_emb_size != tokenizer_vocab_size:
                        logging.warning(f"⚠️ output_embeddings大小 ({final_output_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        # resize_token_embeddings应该同时resize input和output，但如果失败，手动检查
                        try:
                            # 检查resize后是否已修复
                            check_output_emb_size = self.model.get_output_embeddings().weight.size(0)
                            if check_output_emb_size != tokenizer_vocab_size:
                                logging.warning(f"   output_embeddings仍不匹配，将使用'限域+clamp'策略")
                        except:
                            pass
                except Exception as e:
                    logging.warning(f"⚠️ 无法检查LoRA后的embedding大小: {e}")
            
            logging.info("Student model loaded successfully")
            
        except Exception as e:
            logging.error(f"Student model loading failed: {e}")
            raise
    
    def setup_for_ppo(self) -> AutoModelForCausalLMWithValueHead:
        """
        设置模型用于PPO训练
        
        Returns:
            带价值头的模型
        """
        try:
            # ValueHead 模型不支持 CPU/磁盘卸载，需要确保模型完全在GPU上
            # 检查基础模型是否使用了 device_map="auto"（可能部分层被卸载）
            model_to_check = self.base_model if hasattr(self, 'base_model') else self.model
            has_device_map = False
            
            # 检查设备映射（可能在不同位置）
            if hasattr(model_to_check, 'hf_device_map') and model_to_check.hf_device_map:
                has_device_map = True
            elif hasattr(model_to_check, 'device_map') and model_to_check.device_map:
                has_device_map = True
            
            if has_device_map:
                logging.warning("检测到模型使用了设备映射，需要将所有层移动到单一设备以避免ValueHead不支持卸载的问题")
                # 找到第一个GPU设备
                target_device = None
                if torch.cuda.is_available():
                    target_device = torch.device("cuda:0")
                    logging.info(f"将所有模型参数移动到设备: {target_device}")
                else:
                    target_device = torch.device("cpu")
                    logging.warning("未检测到CUDA，使用CPU（可能影响性能）")
                
                # 检查模型是否在 meta device 上
                # 如果第一个参数的 device 类型是 meta，需要使用 to_empty 而不是 to
                try:
                    first_param = next(model_to_check.parameters())
                    is_meta_device = first_param.device.type == 'meta'
                except StopIteration:
                    is_meta_device = False
                
                # 将模型移动到单一设备
                # 对于 PEFT 模型，需要同时移动基础模型和 PEFT 模型
                if hasattr(self, 'base_model'):
                    # 移动基础模型
                    if is_meta_device:
                        logging.warning("检测到模型在 meta device 上，需要先加载权重")
                        # 对于 meta device，应该重新加载而不是移动
                        # 这种情况通常不应该发生，因为我们已经在加载时关闭了 low_cpu_mem_usage
                        raise RuntimeError("模型在 meta device 上。请确保加载模型时使用 low_cpu_mem_usage=False")
                    else:
                        if hasattr(self.base_model, 'to'):
                            self.base_model = self.base_model.to(target_device)
                    
                    # 移动 PEFT 模型
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(target_device)
                else:
                    # 普通模型
                    if is_meta_device:
                        logging.warning("检测到模型在 meta device 上，需要先加载权重")
                        raise RuntimeError("模型在 meta device 上。请确保加载模型时使用 low_cpu_mem_usage=False")
                    else:
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(target_device)
                
                # ✅ 修复：不能将 hf_device_map 设置为 None，因为 TRL 库期望它是字典
                # 应该将其设置为表示所有层都在同一设备上的字典格式
                if hasattr(model_to_check, 'hf_device_map'):
                    # 获取目标设备的字符串表示
                    if isinstance(target_device, torch.device):
                        device_str = str(target_device)
                    else:
                        device_str = target_device
                    
                    # 如果 hf_device_map 存在且不为 None，将其更新为单一设备的映射
                    if model_to_check.hf_device_map is not None and isinstance(model_to_check.hf_device_map, dict):
                        # 将所有设备映射统一到目标设备
                        model_to_check.hf_device_map = {name: device_str for name in model_to_check.hf_device_map.keys()}
                    else:
                        # 如果 hf_device_map 是 None 或不是字典，创建一个基本映射
                        # TRL 库需要它是一个字典，所以至少提供一个键值对
                        model_to_check.hf_device_map = {"model": device_str}
                
                # device_map 可以设置为 None，因为它不被 TRL 库使用
                if hasattr(model_to_check, 'device_map'):
                    model_to_check.device_map = None
            
            # 创建带价值头的模型
            # 注意：不使用 device_map="auto"，因为 ValueHead 不支持卸载
            # 确保模型已经在正确的设备上（不是 meta device）
            
            # 检查模型当前设备
            try:
                current_device = next(self.model.parameters()).device
                if current_device.type == 'meta':
                    raise RuntimeError("模型仍在 meta device 上。请确保加载模型时使用 low_cpu_mem_usage=False")
            except StopIteration:
                # 模型没有参数，这是异常情况
                raise RuntimeError("模型没有参数，无法确定设备位置")
            
            # 如果模型在 CPU 上，移动到原始GPU设备（如果有）
            if torch.cuda.is_available() and current_device.type == 'cpu':
                # 尝试保持模型在原始设备上，而不是强制移动到cuda:0
                # 检查模型参数的实际设备
                try:
                    # 获取第一个参数的设备作为目标设备
                    for param in self.model.parameters():
                        if param.device.type == 'cuda':
                            target_device = param.device
                            break
                    else:
                        # 如果没有找到GPU参数，使用cuda:0
                        target_device = torch.device("cuda:0")
                except:
                    target_device = torch.device("cuda:0")
                
                logging.info(f"将模型移动到GPU设备: {target_device}（保持原始设备）")
                self.model = self.model.to(target_device)
            
            # ✅ 修复：确保在创建 ValueHead 之前，hf_device_map 存在且是字典格式
            # TRL 库的 post_init 会检查 hf_device_map.values()，不能是 None
            if hasattr(self.model, 'hf_device_map'):
                if self.model.hf_device_map is None or not isinstance(self.model.hf_device_map, dict):
                    # 获取当前设备
                    try:
                        current_device_str = str(next(self.model.parameters()).device)
                    except:
                        current_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
                    # 创建一个有效的设备映射
                    self.model.hf_device_map = {"model": current_device_str}
            
            # 创建 ValueHead 模型
            # 注意：from_pretrained 的第一个参数可以是模型实例或路径
            # 这里传入模型实例，确保使用已加载的权重
            ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model,  # 传入模型实例，不是路径
                torch_dtype=self.torch_dtype,
                device_map=None  # 不使用自动设备映射，避免卸载
            )
            
            # ✅ 启用梯度检查点（总是启用以最大化显存节省）：牺牲训练速度换取显存
            # 梯度检查点可以节省30-40%的激活显存，对于log_softmax OOM特别有效
            # 注意：配置在rl_trainer中会再次检查，这里总是启用以最大化显存节省
            try:
                # 尝试在ppo_model上启用
                if hasattr(ppo_model, 'gradient_checkpointing_enable'):
                    ppo_model.gradient_checkpointing_enable()
                    logging.info("✅ 已启用梯度检查点（节省激活显存）")
                # 尝试在pretrained_model上启用（AutoModelForCausalLMWithValueHead的结构）
                elif hasattr(ppo_model, 'pretrained_model') and hasattr(ppo_model.pretrained_model, 'gradient_checkpointing_enable'):
                    ppo_model.pretrained_model.gradient_checkpointing_enable()
                    logging.info("✅ 已在pretrained_model上启用梯度检查点（节省激活显存）")
                # 尝试在基础模型上启用
                elif hasattr(ppo_model, 'base_model') and hasattr(ppo_model.base_model, 'gradient_checkpointing_enable'):
                    ppo_model.base_model.gradient_checkpointing_enable()
                    logging.info("✅ 已在base_model上启用梯度检查点（节省激活显存）")
            except Exception as e:
                logging.warning(f"⚠️ 启用梯度检查点失败（可能不被支持）: {e}")
            
            logging.info("PPO model setup completed")
            return ppo_model
            
        except Exception as e:
            logging.error(f"PPO model setup failed: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], 
                max_length: int = 256,
                temperature: float = 0.7,
                do_sample: bool = True,
                top_p: float = 0.9,
                top_k: int = 50) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            prompts: 提示文本或列表
            max_length: 最大长度
            temperature: 温度参数
            do_sample: 是否采样
            top_p: top_p参数
            top_k: top_k参数
            
        Returns:
            生成的文本
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        self.model.eval()
        
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with torch.no_grad():
                    # 分词
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    
                    # 🔍 详细诊断：验证token ID范围，防止CUDA索引越界
                    # ⚠️ 关键修复：使用tokenizer的vocab_size作为限制（更严格）
                    # 因为生成后需要用tokenizer解码，必须确保生成的token ID在tokenizer范围内
                    model_vocab_size = getattr(self.model.config, 'vocab_size', None)
                    tokenizer_vocab_size = len(self.tokenizer)
                    
                    # 🔥 使用较小的vocab_size（tokenizer的实际范围），防止生成无效token
                    if model_vocab_size is not None and tokenizer_vocab_size is not None:
                        if model_vocab_size > tokenizer_vocab_size:
                            # 🔍 关键问题：模型和tokenizer词汇表不匹配
                            vocab_diff = model_vocab_size - tokenizer_vocab_size
                            if retry_count == 0:  # 只在第一次记录
                                logging.warning(f"⚠️ 模型vocab_size ({model_vocab_size}) > tokenizer vocab_size ({tokenizer_vocab_size})")
                                logging.warning(f"   差异: {vocab_diff} 个token，这可能导致生成无效token")
                                logging.warning(f"   将使用tokenizer范围 ({tokenizer_vocab_size}) 作为限制")
                            vocab_size = tokenizer_vocab_size  # 使用更严格的限制
                        else:
                            vocab_size = min(model_vocab_size, tokenizer_vocab_size)
                    else:
                        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
                    
                    # 🔍 记录最终使用的vocab_size
                    if retry_count == 0 and model_vocab_size is not None and tokenizer_vocab_size is not None:
                        if model_vocab_size != tokenizer_vocab_size:
                            logging.debug(f"📊 Vocab大小检查: 模型={model_vocab_size}, tokenizer={tokenizer_vocab_size}, 使用={vocab_size}")
                    
                    if 'input_ids' in inputs:
                        input_ids = inputs['input_ids']
                        # 🔍 详细检查：打印输入信息
                        max_token_id = input_ids.max().item()
                        min_token_id = input_ids.min().item()
                        input_len = input_ids.shape[1]
                        
                        # 检查所有token ID是否在有效范围内
                        invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                        if torch.any(invalid_mask):
                            invalid_ids = input_ids[invalid_mask].unique().tolist()
                            logging.error(f"❌ 检测到无效token ID!")
                            logging.error(f"   无效ID列表: {invalid_ids[:10]}")  # 只显示前10个
                            logging.error(f"   输入序列长度: {input_len}")
                            logging.error(f"   最大token ID: {max_token_id}, 最小: {min_token_id}")
                            logging.error(f"   模型vocab_size: {model_vocab_size}")
                            logging.error(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                            logging.error(f"   使用的vocab_size: {vocab_size}")
                            logging.error(f"   将截断到有效范围 [0, {vocab_size-1}]")
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
                    
                    # 🔥 修复：检查是否使用device_map，如果有则不强制移动
                    # 对于device_map="auto"的模型，让HF自动处理设备分配
                    has_device_map = (hasattr(self.model, 'hf_device_map') and self.model.hf_device_map) or \
                                   (hasattr(self.model, 'device_map') and self.model.device_map)
                    if not has_device_map and hasattr(self.model, 'device'):
                        # 只有在没有device_map且模型有单一device时才移动
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    # 否则让HF自动处理
                    
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
                        return [""] * len(prompts) if len(prompts) > 1 else ""
                    
                    # 🔍 详细的生成参数日志（仅在错误时）
                    if retry_count > 0:
                        logging.info(f"🔍 生成参数: max_new_tokens={max_allowed_new_tokens}, current_len={current_len}, max_total={max_total_len}")
                        logging.info(f"   pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, vocab_size={vocab_size}")
                    
                    # 🔥 关键修复：添加LogitsProcessor来强制模型只生成tokenizer范围内的token
                    from transformers import LogitsProcessorList
                    
                    class TokenRangeLogitsProcessor:
                        """强制模型只生成有效范围内的token（严格限制到tokenizer范围）"""
                        def __init__(self, max_valid_token_id: int):
                            self.max_valid_token_id = max_valid_token_id
                            self.vocab_end_idx = max_valid_token_id + 1
                        
                        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                            # 🔥 关键修复：只mask，不切片（保持形状不变）
                            # Transformers约定logits的最后一维形状必须保持不变
                            if scores.shape[-1] > self.vocab_end_idx:
                                scores[..., self.vocab_end_idx:] = float('-inf')
                            return scores
                    
                    # 🔥 关键：使用tokenizer的实际最大token ID（vocab_size - 1）
                    # 这是唯一的安全范围，因为tokenizer无法解码超出此范围的token
                    max_valid_token_id = vocab_size - 1
                    logits_processor = LogitsProcessorList([
                        TokenRangeLogitsProcessor(max_valid_token_id)
                    ])
                    
                    # 🔍 验证LogitsProcessor设置
                    if retry_count > 0:
                        logging.info(f"   LogitsProcessor: 最大有效token ID = {max_valid_token_id} (vocab_size={vocab_size})")
                    
                    # 🔥 关键修复：包装模型的forward方法，确保所有input_ids都在有效范围内
                    # 这防止在生成过程中，模型的内部操作访问超出tokenizer范围的embedding
                    original_forward = None
                    original_model = self.model
                    
                    # 获取实际的模型（可能是PeftModel包装的基础模型）
                    model_to_wrap = self.model
                    if hasattr(self.model, 'get_base_model'):
                        model_to_wrap = self.model.get_base_model()
                    elif hasattr(self.model, 'base_model'):
                        model_to_wrap = self.model.base_model.model if hasattr(self.model.base_model, 'model') else self.model.base_model
                    
                    # 包装embedding层，确保所有token ID都在有效范围内
                    def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id, layer_name="embedding"):
                        """创建安全的embedding包装器"""
                        original_embed = embedding_layer.forward
                        
                        # 获取embedding层的实际大小
                        try:
                            actual_emb_size = embedding_layer.weight.size(0)
                        except:
                            actual_emb_size = None
                        
                        def safe_forward(input_ids, *args, **kwargs):
                            # 🔥 关键：在embedding查表前，将所有token ID限制在有效范围
                            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                                # 检查是否有超出实际embedding大小的token
                                if actual_emb_size is not None:
                                    max_id_in_input = input_ids.max().item() if input_ids.numel() > 0 else -1
                                    if max_id_in_input >= actual_emb_size:
                                        if not hasattr(safe_forward, '_warned'):
                                            logging.error(f"❌ {layer_name}: input_ids包含超出embedding大小的token! max={max_id_in_input}, embedding_size={actual_emb_size}, 限制到={max_valid_token_id}")
                                            safe_forward._warned = True
                                        # 使用更严格的限制：min(实际embedding大小, tokenizer大小)
                                        safe_max = min(actual_emb_size - 1, max_valid_token_id)
                                        input_ids = torch.clamp(input_ids, 0, safe_max)
                                    else:
                                        # 即使没有超出，也限制到tokenizer范围
                                        input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                                else:
                                    # 如果无法获取实际大小，使用tokenizer范围
                                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                            return original_embed(input_ids, *args, **kwargs)
                        
                        embedding_layer.forward = safe_forward
                        return original_embed
                    
                    # 包装embedding层（如果存在）
                    # 尝试多个可能的路径找到embedding层
                    restored_embeddings = []
                    embedding_layers_to_wrap = []
                    
                    # 检查多个可能的embedding层路径
                    if hasattr(model_to_wrap, 'embed_tokens'):
                        embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
                    elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
                        embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
                    elif hasattr(model_to_wrap, 'wte'):  # GPT风格模型
                        embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
                    
                    for layer_name, embedding_layer in embedding_layers_to_wrap:
                        original_embed = create_safe_embedding_wrapper(
                            embedding_layer, 
                            vocab_size - 1,
                            layer_name
                        )
                        restored_embeddings.append((embedding_layer, original_embed, layer_name))
                        if retry_count == 0:
                            logging.debug(f"✅ 已包装embedding层: {layer_name}，限制范围: [0, {vocab_size - 1}]")
                    
                    try:
                        # 🔥 关键：确保模型使用tokenizer的vocab_size，而不是模型自己的vocab_size
                        # 通过限制logits维度并在生成参数中显式指定vocab_size
                        generate_kwargs = {
                            **inputs,
                            "max_new_tokens": max_allowed_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                            "top_p": top_p,
                            "top_k": min(top_k, vocab_size - 1),  # 确保top_k不超过tokenizer范围
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "repetition_penalty": 1.1,
                            "logits_processor": logits_processor,  # 🔥 关键：强制使用tokenizer范围
                            "use_cache": True
                        }
                        
                        # 生成
                        outputs = self.model.generate(**generate_kwargs)
                    finally:
                        # 恢复原始的embedding forward方法
                        for embedding_layer, original_embed, layer_name in restored_embeddings:
                            embedding_layer.forward = original_embed
                    
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
                    generated_texts = []
                    for i, output in enumerate(outputs):
                        try:
                            input_length = inputs['input_ids'][i].shape[0]
                            if len(output) > input_length:
                                generated_text = self.tokenizer.decode(
                                    output[input_length:],
                                    skip_special_tokens=True
                                )
                            else:
                                generated_text = ""
                            generated_texts.append(generated_text)
                        except Exception as e:
                            logging.warning(f"解码失败: {e}，使用空字符串")
                            generated_texts.append("")
                    
                return generated_texts if len(generated_texts) > 1 else generated_texts[0]
                
            except RuntimeError as e:
                error_str = str(e)
                if "device-side assert" in error_str or "CUDA error" in error_str:
                    retry_count += 1
                    # 🔍 详细错误诊断
                    logging.error(f"❌ CUDA device-side assert错误 (重试 {retry_count}/{max_retries})")
                    logging.error(f"   错误信息: {error_str[:500]}")  # 只显示前500字符
                    
                    # 🔍 诊断信息：检查模型和输入状态
                    try:
                        if 'inputs' in locals():
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
                        logging.error(f"   提示数量: {len(prompts)}")
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
                        return [""] * len(prompts) if len(prompts) > 1 else ""
                else:
                    # 其他错误直接抛出
                    raise
            except Exception as e:
                logging.error(f"生成文本失败: {e}")
                # 返回空字符串而不是抛出异常
                return [""] * len(prompts) if len(prompts) > 1 else ""
    
    def get_logits(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        获取文本的logits
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            logits张量
        """
        if isinstance(text, str):
            text = [text]
        
        self.model.eval()
        
        # 🔥 关键修复：获取tokenizer的实际vocab_size作为限制
        model_vocab_size = getattr(self.model.config, 'vocab_size', None)
        tokenizer_vocab_size = len(self.tokenizer)
        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
        if model_vocab_size is not None and tokenizer_vocab_size is not None:
            if model_vocab_size > tokenizer_vocab_size:
                vocab_size = tokenizer_vocab_size  # 使用更严格的限制
        
        # 🔥 关键修复：包装embedding层以确保token ID在有效范围内
        restored_embeddings = []
        model_to_wrap = self.model
        if hasattr(self.model, 'get_base_model'):
            model_to_wrap = self.model.get_base_model()
        elif hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                model_to_wrap = self.model.base_model.model
            else:
                model_to_wrap = self.model.base_model
        elif hasattr(self.model, 'pretrained_model'):
            model_to_wrap = self.model.pretrained_model
        
        def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id):
            """创建安全的embedding包装器"""
            original_embed = embedding_layer.forward
            
            def safe_forward(input_ids, *args, **kwargs):
                if input_ids is not None and isinstance(input_ids, torch.Tensor):
                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                return original_embed(input_ids, *args, **kwargs)
            
            embedding_layer.forward = safe_forward
            return original_embed
        
        embedding_layers_to_wrap = []
        if hasattr(model_to_wrap, 'embed_tokens'):
            embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
        elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
            embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
        elif hasattr(model_to_wrap, 'wte'):
            embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
        
        for layer_name, embedding_layer in embedding_layers_to_wrap:
            original_embed = create_safe_embedding_wrapper(embedding_layer, vocab_size - 1)
            restored_embeddings.append((embedding_layer, original_embed, layer_name))
        
        try:
            with torch.no_grad():
                # 分词
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # 🔥 修复：检查是否使用device_map，如果有则不强制移动
                has_device_map = (hasattr(self.model, 'hf_device_map') and self.model.hf_device_map) or \
                               (hasattr(self.model, 'device_map') and self.model.device_map)
                if not has_device_map and hasattr(self.model, 'device'):
                    # 只有在没有device_map且模型有单一device时才移动
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                # 否则让HF自动处理
                
                # 🔥 关键：确保input_ids在有效范围内
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                    if torch.any(invalid_mask):
                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                
                # 前向传播
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 🔥 注意：不再切片logits，因为vocab_size已匹配（如果还有问题应该在加载时报错）
                
                # 如果是单个文本，去掉batch维度（与TeacherModel一致）
                if logits.shape[0] == 1:
                    return logits[0]
                
                return logits
        finally:
            # 恢复原始的embedding forward方法
            for embedding_layer, original_embed, layer_name in restored_embeddings:
                embedding_layer.forward = original_embed
    
    def compute_log_probs(self, text: str) -> torch.Tensor:
        """
        计算文本的对数概率
        
        Args:
            text: 输入文本
            
        Returns:
            对数概率张量
        """
        # 🔥 关键修复：获取tokenizer的实际vocab_size作为限制
        model_vocab_size = getattr(self.model.config, 'vocab_size', None)
        tokenizer_vocab_size = len(self.tokenizer)
        vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else (model_vocab_size if model_vocab_size is not None else 50000)
        if model_vocab_size is not None and tokenizer_vocab_size is not None:
            if model_vocab_size > tokenizer_vocab_size:
                vocab_size = tokenizer_vocab_size  # 使用更严格的限制
        
        # 🔥 关键修复：包装embedding层以确保token ID在有效范围内
        restored_embeddings = []
        model_to_wrap = self.model
        if hasattr(self.model, 'get_base_model'):
            model_to_wrap = self.model.get_base_model()
        elif hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'model'):
                model_to_wrap = self.model.base_model.model
            else:
                model_to_wrap = self.model.base_model
        elif hasattr(self.model, 'pretrained_model'):
            model_to_wrap = self.model.pretrained_model
        
        def create_safe_embedding_wrapper(embedding_layer, max_valid_token_id):
            """创建安全的embedding包装器"""
            original_embed = embedding_layer.forward
            
            def safe_forward(input_ids, *args, **kwargs):
                if input_ids is not None and isinstance(input_ids, torch.Tensor):
                    input_ids = torch.clamp(input_ids, 0, max_valid_token_id)
                return original_embed(input_ids, *args, **kwargs)
            
            embedding_layer.forward = safe_forward
            return original_embed
        
        embedding_layers_to_wrap = []
        if hasattr(model_to_wrap, 'embed_tokens'):
            embedding_layers_to_wrap.append(('embed_tokens', model_to_wrap.embed_tokens))
        elif hasattr(model_to_wrap, 'model') and hasattr(model_to_wrap.model, 'embed_tokens'):
            embedding_layers_to_wrap.append(('model.embed_tokens', model_to_wrap.model.embed_tokens))
        elif hasattr(model_to_wrap, 'wte'):
            embedding_layers_to_wrap.append(('wte', model_to_wrap.wte))
        
        for layer_name, embedding_layer in embedding_layers_to_wrap:
            original_embed = create_safe_embedding_wrapper(embedding_layer, vocab_size - 1)
            restored_embeddings.append((embedding_layer, original_embed, layer_name))
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # 🔥 修复：检查是否使用device_map，如果有则不强制移动
                has_device_map = (hasattr(self.model, 'hf_device_map') and self.model.hf_device_map) or \
                               (hasattr(self.model, 'device_map') and self.model.device_map)
                if not has_device_map and hasattr(self.model, 'device'):
                    # 只有在没有device_map且模型有单一device时才移动
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                # 否则让HF自动处理
                
                # 🔥 关键：确保input_ids在有效范围内
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
                    if torch.any(invalid_mask):
                        inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 计算对数概率
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                return log_probs
        finally:
            # 恢复原始的embedding forward方法
            for embedding_layer, original_embed, layer_name in restored_embeddings:
                embedding_layer.forward = original_embed
    
    def save_model(self, save_path: str, save_adapter: bool = True):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            save_adapter: 是否只保存适配器
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        if save_adapter and self.use_lora:
            # 只保存LoRA适配器
            self.model.save_pretrained(save_path)
            # ✅ 修复：保存tokenizer，评估时需要
            self.tokenizer.save_pretrained(save_path)
            logging.info(f"LoRA adapter and tokenizer saved to: {save_path}")
        else:
            # 保存完整模型
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logging.info(f"Complete model saved to: {save_path}")
    
    def load_model(self, load_path: str, load_adapter: bool = True):
        """
        加载模型
        
        Args:
            load_path: 加载路径
            load_adapter: 是否只加载适配器
        """
        try:
            if load_adapter and self.use_lora:
                # 加载LoRA适配器
                self.model = PeftModel.from_pretrained(self.base_model, load_path)
                logging.info(f"LoRA adapter loaded from {load_path}")
            else:
                # 加载完整模型
                self.model = AutoModelForCausalLM.from_pretrained(load_path)
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                logging.info(f"Complete model loaded from {load_path}")
                
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model_name,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "use_lora": self.use_lora
        }
        
        if self.use_lora:
            info["lora_config"] = self.lora_config
        
        return info
    
    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logging.info("Base model parameters frozen")
    
    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        logging.info("Base model parameters unfrozen")
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}")


class StudentModelManager:
    """Student Model Manager"""
    
    def __init__(self, config: Dict):
        """
        初始化管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.student_model = None
        self.ppo_model = None
        
    def initialize_student(self) -> StudentModel:
        """初始化学生模型"""
        if self.student_model is None:
            self.student_model = StudentModel(
                model_name=self.config["student_model"]["base_model_name"],
                lora_config=self.config["student_model"]["lora_config"],
                device=self.config["device"]["device_map"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                use_lora=self.config["student_model"]["use_lora"]
            )
        
        return self.student_model
    
    def get_student(self) -> StudentModel:
        """获取学生模型实例"""
        if self.student_model is None:
            return self.initialize_student()
        return self.student_model
    
    def setup_ppo_model(self) -> AutoModelForCausalLMWithValueHead:
        """设置PPO模型"""
        if self.ppo_model is None:
            student = self.get_student()
            self.ppo_model = student.setup_for_ppo()
        
        return self.ppo_model
    
    def cleanup(self):
        """清理资源"""
        if self.ppo_model is not None:
            del self.ppo_model
        if self.student_model is not None:
            del self.student_model
        torch.cuda.empty_cache()


def create_student_model(config: Dict) -> StudentModel:
    """
    创建学生模型的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        学生模型实例
    """
    return StudentModel(
        model_name=config["student_model"]["base_model_name"],
        lora_config=config["student_model"]["lora_config"],
        device=config["device"]["device_map"],
        torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
        use_lora=config["student_model"]["use_lora"]
    )

