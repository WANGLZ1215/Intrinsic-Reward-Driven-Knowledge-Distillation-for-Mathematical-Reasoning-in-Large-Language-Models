#!/usr/bin/env python3
"""
监督微调训练脚本
功能：在GSM8K数据集上对Qwen-7B-math进行监督微调
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from training.sft_trainer import SFTTrainer
from data.gsm8k_processor import GSM8KProcessor
from datasets import load_dataset


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """准备数据"""
    # 加载GSM8K数据集
    dataset = load_dataset("gsm8k", "main")
    
    # 数据限制（如果配置了）
    max_train_samples = config["data"].get("max_train_samples")
    max_eval_samples = config["data"].get("max_eval_samples")
    
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    
    if max_eval_samples:
        dataset["test"] = dataset["test"].select(range(min(max_eval_samples, len(dataset["test"]))))
    
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"测试集大小: {len(dataset['test'])}")
    
    return dataset


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="监督微调训练")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（覆盖配置文件设置）")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="日志级别")
    # 注意：SFT训练使用Transformers Trainer，会自动检测检查点并恢复
    # 如果输出目录中有checkpoint-*文件夹，会自动从最新检查点恢复
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 覆盖输出目录
        if args.output_dir:
            config["sft"]["output_dir"] = args.output_dir
        
        # 创建输出目录
        output_dir = config["sft"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("开始监督微调训练")
        logger.info(f"配置: {config}")
        
        # 准备数据
        logger.info("准备数据集...")
        dataset = prepare_data(config)
        
        # 创建训练器
        logger.info("初始化训练器...")
        trainer = SFTTrainer(config)
        
        # 设置模型
        logger.info("设置模型...")
        trainer.setup_model()
        
        # 开始训练
        logger.info("开始训练...")
        train_dataset = dataset[config["data"]["train_split"]]
        eval_dataset = dataset[config["data"]["eval_split"]]
        
        trainer.train(train_dataset, eval_dataset)
        
        # 最终评估
        logger.info("进行最终评估...")
        eval_results = trainer.evaluate(eval_dataset)
        logger.info(f"最终评估结果: {eval_results}")
        
        # 保存评估结果
        eval_results_file = os.path.join(output_dir, "eval_results.yaml")
        with open(eval_results_file, 'w', encoding='utf-8') as f:
            yaml.dump(eval_results, f, default_flow_style=False)
        
        logger.info("监督微调训练完成！")
        logger.info(f"模型保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    main()






