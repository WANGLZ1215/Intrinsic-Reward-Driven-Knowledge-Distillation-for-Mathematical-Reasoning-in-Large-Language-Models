#!/usr/bin/env python3
"""
数据准备脚本
功能：下载GSM8K数据集并显示基本信息
"""

import argparse
import logging
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from data.gsm8k_processor import GSM8KProcessor


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载GSM8K数据集")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="日志级别")
    parser.add_argument("--show_samples", type=int, default=3,
                       help="显示样本数量")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始下载GSM8K数据集...")
        
        # 下载数据集
        dataset = load_dataset("gsm8k", "main")
        
        logger.info(f"✅ 数据集下载成功!")
        logger.info(f"📊 训练集大小: {len(dataset['train'])} 样本")
        logger.info(f"📊 测试集大小: {len(dataset['test'])} 样本")
        
        # 显示样本
        if args.show_samples > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"显示 {args.show_samples} 个训练集样本:")
            logger.info(f"{'='*60}")
            
            for i in range(min(args.show_samples, len(dataset['train']))):
                sample = dataset['train'][i]
                logger.info(f"\n样本 {i+1}:")
                logger.info(f"问题: {sample['question'][:100]}...")
                logger.info(f"答案: {sample['answer'][:100]}...")
        
        logger.info("\n✅ 数据准备完成！数据集已缓存，可以在训练脚本中直接使用。")
        
    except Exception as e:
        logger.error(f"❌ 数据准备失败: {e}")
        raise


if __name__ == "__main__":
    main()






