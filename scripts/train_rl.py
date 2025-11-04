#!/usr/bin/env python3
"""
Reinforcement Learning Training Script
Function: PPO training based on intrinsic rewards
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
import sys
from tqdm import tqdm
import time

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from training.rl_trainer import RLTrainer
from data.gsm8k_processor import GSM8KProcessor
from datasets import load_dataset


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    import os
    from pathlib import Path
    from datetime import datetime
    
    # 创建logs目录
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 如果没有指定日志文件，使用默认名称
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"rl_training_{timestamp}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志，同时输出到控制台和文件
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")


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
    
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    return dataset


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强化学习训练")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（覆盖配置文件设置）")
    parser.add_argument("--student_model_path", type=str, default=None,
                       help="学生模型路径（SFT后的模型）")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大训练步数")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="日志级别")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径（默认：logs/rl_training_YYYYMMDD_HHMMSS.log）")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = args.log_file if hasattr(args, 'log_file') and args.log_file else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # 总体进度条（如果从检查点恢复则有6步，否则5步）
    total_steps = 6 if args.resume_from_checkpoint else 5
    main_progress = tqdm(total=total_steps, desc="RL训练总进度", ncols=100, position=0)
    
    try:
        start_time = time.time()
        
        # 步骤1: 加载配置
        main_progress.set_description("📋 加载配置")
        config = load_config(args.config)
        
        # 覆盖配置
        if args.output_dir:
            config["ppo"]["output_dir"] = args.output_dir
        
        if args.student_model_path:
            config["model"]["student_model_name"] = args.student_model_path
        
        if args.max_steps:
            config["training"]["max_steps"] = args.max_steps
        
        # 创建输出目录
        output_dir = config["ppo"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        main_progress.update(1)
        main_progress.set_postfix({"status": "配置加载完成"})
        
        # 步骤2: 准备数据
        main_progress.set_description("📊 准备数据")
        logger.info("Preparing dataset...")
        dataset = prepare_data(config)
        main_progress.update(1)
        main_progress.set_postfix({"status": "数据准备完成"})
        
        # 步骤3: 创建训练器
        main_progress.set_description("🏗️ 初始化训练器")
        logger.info("Initializing RL trainer...")
        trainer = RLTrainer(config)
        main_progress.update(1)
        main_progress.set_postfix({"status": "训练器初始化完成"})
        
        # 步骤4: 设置模型和组件
        main_progress.set_description("⚙️ 设置模型和组件")
        logger.info("Setting up models...")
        trainer.setup_models()
        trainer.setup_components()
        trainer.setup_ppo_trainer()
        main_progress.update(1)
        main_progress.set_postfix({"status": "模型和组件设置完成"})
        
        # 步骤5: 如果需要从检查点恢复
        if args.resume_from_checkpoint:
            main_progress.set_description("🔄 恢复检查点")
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.load_checkpoint(args.resume_from_checkpoint)
            main_progress.update(1)
            main_progress.set_postfix({"status": "检查点恢复完成"})
        
        # 步骤6: 开始训练
        main_progress.set_description("🚀 开始训练")
        logger.info("Starting RL training...")
        train_dataset = dataset[config["data"]["train_split"]]
        
        trainer.train(train_dataset, max_steps=config["training"]["max_steps"])
        main_progress.update(1)
        main_progress.set_postfix({"status": "训练完成"})
        
        # 训练完成统计
        total_time = time.time() - start_time
        main_progress.close()
        
        print(f"\n🎉 强化学习训练完成!")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📁 模型保存位置: {output_dir}")
        
        logger.info("Reinforcement learning training completed!")
        logger.info(f"Model saved to: {output_dir}")
        
    except Exception as e:
        main_progress.close()
        logger.error(f"RL training failed: {e}")
        raise
    
    finally:
        # 清理资源
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()






