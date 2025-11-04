"""
Supervised Fine-tuning Trainer
Function: Implement supervised fine-tuning of Qwen-7B-math on GSM8K dataset
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import wandb
from torch.utils.data import DataLoader
from utils.cache_utils import suppress_past_key_values_warning, update_model_for_modern_cache


class SFTTrainer:
    """Supervised Fine-tuning Trainer"""
    
    def __init__(self, config: Dict):
        """
        初始化SFT训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 抑制past_key_values警告
        suppress_past_key_values_warning()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化wandb（如果启用）
        if config.get("logging", {}).get("use_wandb", False):
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config
            )
    
    def setup_model(self):
        """设置模型和分词器"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["student_model_name"],
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["student_model_name"],
                torch_dtype=getattr(torch, self.config["device"]["torch_dtype"]),
                device_map=self.config["device"]["device_map"],
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 🔥 关键修复：检查并修复embedding大小以匹配tokenizer（在应用LoRA之前）
            # 这可以防止SFT阶段训练时出现vocab_size不匹配，避免RL阶段加载checkpoint时的问题
            tokenizer_vocab_size = len(self.tokenizer)
            try:
                input_emb_size = self.model.get_input_embeddings().weight.size(0)
                output_emb_size = None
                if hasattr(self.model, 'get_output_embeddings') and self.model.get_output_embeddings() is not None:
                    output_emb_size = self.model.get_output_embeddings().weight.size(0)
                
                self.logger.info(f"📊 Embedding大小检查（应用LoRA前）:")
                self.logger.info(f"   tokenizer vocab_size: {tokenizer_vocab_size}")
                self.logger.info(f"   model input_embeddings.size(0): {input_emb_size}")
                if output_emb_size is not None:
                    self.logger.info(f"   model output_embeddings.size(0): {output_emb_size}")
                self.logger.info(f"   model.config.vocab_size: {getattr(self.model.config, 'vocab_size', 'N/A')}")
                
                # 如果embedding大小与tokenizer不匹配，执行resize
                if input_emb_size != tokenizer_vocab_size:
                    self.logger.warning(f"⚠️ 模型embedding大小 ({input_emb_size}) != tokenizer大小 ({tokenizer_vocab_size})")
                    self.logger.info(f"   正在resize_token_embeddings到 {tokenizer_vocab_size}...")
                    self.model.resize_token_embeddings(tokenizer_vocab_size)
                    self.logger.info(f"✅ resize_token_embeddings完成")
                    
                    # 验证resize是否成功
                    new_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    if new_input_emb_size != tokenizer_vocab_size:
                        self.logger.error(f"❌ resize_token_embeddings失败！新大小: {new_input_emb_size} != {tokenizer_vocab_size}")
                        raise ValueError(f"Resize失败: {new_input_emb_size} != {tokenizer_vocab_size}")
                    else:
                        self.logger.info(f"✅ resize成功验证: input_embeddings.size(0) = {new_input_emb_size}")
                else:
                    self.logger.info(f"✅ embedding大小与tokenizer匹配，无需resize")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ resize_token_embeddings时出错（可能不支持或已量化）: {e}")
                # 如果是ValueError（resize失败），应该抛出异常
                if isinstance(e, ValueError):
                    raise
                # 其他错误（如模型不支持resize），记录警告但继续
            
            # 应用LoRA（现在embedding大小已经匹配）
            if self.config["model"].get("use_lora", True):
                lora_config = LoraConfig(**self.config["lora"])
                self.model = get_peft_model(self.model, lora_config)
                self.logger.info("LoRA configuration applied")
                
                # 🔥 关键：应用LoRA后再次验证embedding大小（LoRA不应该改变embedding大小，但检查一下）
                try:
                    final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                    if final_input_emb_size != tokenizer_vocab_size:
                        self.logger.warning(f"⚠️ 应用LoRA后，input_embeddings大小 ({final_input_emb_size}) != tokenizer ({tokenizer_vocab_size})")
                        self.logger.info(f"   正在再次resize_token_embeddings到 {tokenizer_vocab_size}...")
                        try:
                            self.model.resize_token_embeddings(tokenizer_vocab_size)
                            new_final_input_emb_size = self.model.get_input_embeddings().weight.size(0)
                            if new_final_input_emb_size == tokenizer_vocab_size:
                                self.logger.info(f"✅ LoRA后resize成功: input_embeddings = {new_final_input_emb_size}")
                            else:
                                self.logger.error(f"❌ LoRA后resize失败: {new_final_input_emb_size} != {tokenizer_vocab_size}")
                        except Exception as e2:
                            self.logger.warning(f"⚠️ LoRA后resize失败: {e2}")
                except Exception as e:
                    self.logger.warning(f"⚠️ 无法检查LoRA后的embedding大小: {e}")
            
            # 更新模型以使用现代缓存
            self.model = update_model_for_modern_cache(self.model)
            
            self.logger.info("Model setup completed")
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """准备数据集"""
        def preprocess_function(examples):
            texts = []
            for question, answer in zip(examples["question"], examples["answer"]):
                prompt = f"Question: {question}\nAnswer: "
                full_text = prompt + answer + self.tokenizer.eos_token
                texts.append(full_text)
            return {"text": texts}
        
        # 预处理数据
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 分词
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["sft"]["max_length"],
                return_tensors="pt"
            )
        
        tokenized_dataset = processed_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        self.logger.info(f"Dataset preparation completed: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def create_data_collator(self):
        """创建数据整理器"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def setup_training_arguments(self):
        """设置训练参数"""
        return TrainingArguments(
            output_dir=self.config["sft"]["output_dir"],
            per_device_train_batch_size=self.config["sft"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["sft"]["per_device_eval_batch_size"],
            num_train_epochs=self.config["sft"]["num_train_epochs"],
            learning_rate=float(self.config["sft"]["learning_rate"]),
            save_strategy=self.config["sft"]["save_strategy"],
            eval_strategy=self.config["sft"].get("eval_strategy", self.config["sft"].get("evaluation_strategy", "epoch")),
            logging_steps=self.config["sft"]["logging_steps"],
            save_total_limit=self.config["sft"]["save_total_limit"],
            load_best_model_at_end=self.config["sft"]["load_best_model_at_end"],
            metric_for_best_model=self.config["sft"]["metric_for_best_model"],
            greater_is_better=self.config["sft"]["greater_is_better"],
            warmup_steps=self.config["sft"]["warmup_steps"],
            fp16=self.config["training"]["fp16"],
            bf16=self.config["training"].get("bf16", False),  # 添加BF16支持
            dataloader_num_workers=self.config["training"]["dataloader_num_workers"],
            remove_unused_columns=self.config["training"]["remove_unused_columns"],
            report_to="wandb" if self.config.get("logging", {}).get("use_wandb", False) else None,
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """开始训练"""
        try:
            # 准备数据
            train_dataset = self.prepare_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = self.prepare_dataset(eval_dataset)
            
            # 创建数据整理器
            data_collator = self.create_data_collator()
            
            # 设置训练参数
            training_args = self.setup_training_arguments()
            
            # 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # 开始训练
            self.logger.info("Starting SFT training...")
            self.trainer.train()
            
            # 保存最终模型
            self.save_model(self.config["sft"]["output_dir"])
            
            self.logger.info("SFT training completed")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, save_path: str):
        """保存模型"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            if hasattr(self.model, 'save_pretrained'):
                # 保存LoRA适配器
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
            else:
                # 保存完整模型
                torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                self.tokenizer.save_pretrained(save_path)
            
            self.logger.info(f"Model saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Model save failed: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized")
        
        # 准备评估数据
        eval_dataset = self.prepare_dataset(eval_dataset)
        
        # 评估
        eval_results = self.trainer.evaluate(eval_dataset)
        
        self.logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def generate_sample(self, prompt: str, max_length: int = 256) -> str:
        """生成样本"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text


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
    trainer = SFTTrainer(config)
    
    # 设置模型
    trainer.setup_model()
    
    # 加载GSM8K数据集
    from datasets import load_dataset
    from data.gsm8k_processor import GSM8KProcessor
    
    print("Loading GSM8K dataset...")
    try:
        # 加载GSM8K数据集
        gsm8k_dataset = load_dataset("gsm8k", "main")
        
        # 创建GSM8K处理器
        processor = GSM8KProcessor(trainer.tokenizer, max_length=config["sft"]["max_length"])
        
        # 使用训练集作为训练数据
        train_dataset = gsm8k_dataset["train"]
        
        # 使用测试集作为验证集（全量数据）
        eval_dataset = gsm8k_dataset["test"]
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(eval_dataset)}")
        
        # 验证数据集质量
        processor.validate_data(train_dataset, num_samples=3)
        processor.validate_data(eval_dataset, num_samples=3)
        
    except Exception as e:
        print(f"Failed to load GSM8K dataset: {e}")
        print("Cannot proceed with training, please check network connection and dependencies")
        return
    
    # 训练
    trainer.train(train_dataset, eval_dataset)
    
    # 评估
    eval_results = trainer.evaluate(eval_dataset)
    print(f"Final evaluation results: {eval_results}")
    
    # 生成样本
    sample_questions = [
        "Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does he run a week?\nAnswer: ",
        "Question: A robe takes 2 bolts of blue fabric and half that much white fabric. How many bolts of fabric does it take?\nAnswer: ",
        "Question: Josh decides to try flipping a house. He buys it for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer: "
    ]
    
    print("\n=== Sample Generation Test ===")
    for i, sample_prompt in enumerate(sample_questions, 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample_prompt}")
        sample_response = trainer.generate_sample(sample_prompt, max_length=200)
        print(f"Generated Answer: {sample_response}")
        print("-" * 50)


if __name__ == "__main__":
    main()





