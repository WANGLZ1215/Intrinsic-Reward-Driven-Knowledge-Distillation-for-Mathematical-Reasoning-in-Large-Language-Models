#!/usr/bin/env python3
"""
RL模型检查点评估脚本
功能：评估本地保存的RL训练检查点，生成完整的评估报告

设计用于VAST AI在线评估环境：
- 支持相对路径和绝对路径
- 使用项目统一的答案提取函数 extract_answer_unified
- 增强的错误处理和日志记录
- 检查点验证和路径解析
"""

import argparse
import yaml
import logging
import os
import json
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import torch

# 添加项目根目录到Python路径
# 支持从evaluation目录或scripts目录运行
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.student_model import StudentModel
from models.teacher_model import TeacherModel
from evaluation.reasoning_evaluator import ReasoningEvaluator
from evaluation.metrics import ComprehensiveEvaluator
from data.gsm8k_processor import GSM8KProcessor
from datasets import load_dataset
from utils.math_utils import extract_answer_unified  # 使用项目统一的答案提取函数


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    # 创建logs目录
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 如果没有指定日志文件，使用默认名称
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"evaluate_checkpoint_{timestamp}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志，同时输出到控制台和文件
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_checkpoint(
    checkpoint_path: str,
    teacher_model_path: str,
    config: dict,
    eval_samples: int = None,
    output_file: str = "evaluation_results.json",
    **kwargs
):
    """
    评估检查点模型
    
    Args:
        checkpoint_path: 检查点路径（如 checkpoints/rl_model/checkpoint-1000）
        teacher_model_path: 教师模型路径
        config: 配置字典
        eval_samples: 评估样本数量（None表示全部）
        output_file: 输出结果文件路径
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("开始评估RL模型检查点")
    logger.info(f"检查点路径: {checkpoint_path}")
    logger.info(f"教师模型: {teacher_model_path}")
    logger.info("=" * 80)
    
    # 初始化结果字典
    results = {
        "checkpoint_path": checkpoint_path,
        "teacher_model": teacher_model_path,
        "evaluation_time": datetime.now().isoformat(),
        "accuracy": 0.0,
        "reasoning_quality": {
            "overall_score": 0.0,
            "step_coverage": 0.0,
            "logical_consistency": 0.0,
            "kl_divergence": 0.0,
            "answer_correctness": 0.0
        },
        "distillation_effect": {
            "overall_score": 0.0,
            "kl_divergence": 0.0,
            "cosine_similarity": 0.0,
            "js_divergence": 0.0
        },
        "statistics": {
            "total_samples": 0,
            "correct_samples": 0,
            "incorrect_samples": 0,
            "average_reasoning_score": 0.0,
            "average_distillation_score": 0.0
        },
        "individual_results": []
    }
    
    try:
        # 步骤1: 加载学生模型（从检查点）
        logger.info("步骤1/5: 加载学生模型...")
        
        # 确保检查点路径是绝对路径（VAST AI兼容性）
        checkpoint_path_abs = Path(checkpoint_path).resolve()
        if not checkpoint_path_abs.exists():
            # 尝试相对于项目根目录
            project_root = Path(__file__).parent.parent
            checkpoint_path_abs = (project_root / checkpoint_path).resolve()
        
        if not checkpoint_path_abs.exists():
            raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path} (尝试了: {checkpoint_path_abs})")
        
        logger.info(f"使用检查点路径: {checkpoint_path_abs}")
        
        # 验证检查点文件
        required_files = [
            checkpoint_path_abs / "adapter_config.json"
        ]
        weight_files = [
            checkpoint_path_abs / "adapter_model.safetensors",
            checkpoint_path_abs / "adapter_model.bin"
        ]
        
        if not required_files[0].exists():
            raise FileNotFoundError(f"检查点缺少必需文件: {required_files[0]}")
        
        if not any(f.exists() for f in weight_files):
            raise FileNotFoundError(f"检查点缺少权重文件: {[str(f) for f in weight_files]}")
        
        logger.info(f"检查点验证通过: {[f.name for f in required_files + weight_files if f.exists()]}")
        
        student_model = StudentModel(
            model_name=str(checkpoint_path_abs),
            lora_config=config["lora"],
            device=config["device"]["device_map"],
            torch_dtype=getattr(torch, config["device"]["torch_dtype"]),
            use_lora=True
        )
        logger.info("✅ 学生模型加载成功")
        
        # 步骤2: 加载教师模型（可选）
        teacher_model = None
        skip_teacher = kwargs.get("skip_teacher", False)
        if not skip_teacher:
            logger.info("步骤2/5: 加载教师模型...")
            try:
                teacher_model = TeacherModel(
                    model_name=teacher_model_path,
                    cache_size=config["model"]["cache_size"],
                    cache_policy=config["model"]["cache_policy"],
                    device=config["device"]["device_map"],
                    torch_dtype=getattr(torch, config["device"]["torch_dtype"])
                )
                logger.info("✅ 教师模型加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 教师模型加载失败: {e}")
                logger.warning("   将继续评估但跳过教师模型相关指标")
                teacher_model = None
        else:
            logger.info("步骤2/5: 跳过教师模型加载（--skip_teacher）")
        
        # 步骤3: 加载评估数据
        logger.info("步骤3/5: 加载评估数据...")
        try:
            dataset = load_dataset("gsm8k", "main")
            logger.info(f"数据集加载成功: train={len(dataset['train'])}, test={len(dataset['test'])}")
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
        
        if eval_samples is not None:
            eval_samples = min(eval_samples, len(dataset["test"]))
            eval_dataset = dataset["test"].select(range(eval_samples))
            logger.info(f"使用 {eval_samples} 个测试样本（总共 {len(dataset['test'])} 个）")
        else:
            eval_dataset = dataset["test"]
            logger.info(f"使用全部 {len(eval_dataset)} 个测试样本")
        
        # 步骤4: 创建评估器
        logger.info("步骤4/5: 初始化评估器...")
        reasoning_evaluator = ReasoningEvaluator()
        comprehensive_evaluator = ComprehensiveEvaluator()
        logger.info("✅ 评估器初始化成功")
        
        # 步骤5: 执行评估
        logger.info("步骤5/5: 开始评估...")
        logger.info("=" * 80)
        
        total_samples = len(eval_dataset)
        correct_count = 0
        
        # 累积各项指标
        reasoning_scores = []
        distillation_scores = []
        step_coverage_scores = []
        logical_consistency_scores = []
        answer_correctness_scores = []
        kl_divergences = []
        cosine_similarities = []
        
        # 使用tqdm显示进度
        for idx, sample in enumerate(tqdm(eval_dataset, desc="评估进度", ncols=100)):
            question = sample["question"]
            ground_truth = sample["answer"]
            
            try:
                # 格式化问题提示
                formatted_question = f"Question: {question}\nAnswer: "
                
                # 学生模型生成
                try:
                    student_response = student_model.generate(
                        formatted_question,
                        max_length=512,
                        temperature=0.7,
                        do_sample=True
                    )
                    # 确保是字符串类型
                    if not isinstance(student_response, str):
                        student_response = str(student_response) if student_response else ""
                except Exception as e:
                    logger.warning(f"样本 {idx+1} 学生模型生成失败: {e}")
                    student_response = ""
                
                # 教师模型生成（可选）
                teacher_response = ""
                if teacher_model is not None:
                    try:
                        teacher_response = teacher_model.generate_response(
                            formatted_question,
                            max_length=512,
                            temperature=0.7
                        )
                        # 确保是字符串类型
                        if not isinstance(teacher_response, str):
                            teacher_response = str(teacher_response) if teacher_response else ""
                    except Exception as e:
                        logger.warning(f"样本 {idx+1} 教师模型生成失败: {e}")
                        teacher_response = ""
                else:
                    teacher_response = ""  # 跳过教师模型
                
                # 获取logits用于蒸馏评估（可选，失败不影响其他评估）
                student_logits = None
                teacher_logits = None
                if student_response and teacher_response:
                    try:
                        # 🔍 安全检查：确保输入不为空，避免索引越界
                        full_text = formatted_question + student_response
                        if len(full_text.strip()) > 0:
                            student_logits = student_model.get_logits(full_text)
                            # 🔍 验证logits维度（防止空tensor导致索引越界）
                            if student_logits is not None and student_logits.numel() == 0:
                                student_logits = None
                    except (IndexError, RuntimeError) as e:
                        logger.warning(f"样本 {idx+1} 获取学生logits失败（索引越界）: {e}")
                        student_logits = None
                    except Exception as e:
                        logger.debug(f"样本 {idx+1} 获取学生logits失败（跳过）: {e}")
                        student_logits = None
                    
                    try:
                        # 🔍 安全检查：确保输入不为空，避免索引越界
                        full_text = formatted_question + teacher_response
                        if len(full_text.strip()) > 0:
                            teacher_logits = teacher_model.get_logits(full_text)
                            # 🔍 验证logits维度（防止空tensor导致索引越界）
                            if teacher_logits is not None and teacher_logits.numel() == 0:
                                teacher_logits = None
                    except (IndexError, RuntimeError) as e:
                        logger.warning(f"样本 {idx+1} 获取教师logits失败（索引越界）: {e}")
                        teacher_logits = None
                    except Exception as e:
                        logger.debug(f"样本 {idx+1} 获取教师logits失败（跳过）: {e}")
                        teacher_logits = None
                
                # 提取答案（使用项目统一的答案提取函数 extract_answer_unified）
                try:
                    ground_truth_text, ground_truth_num = extract_answer_unified(ground_truth)
                except Exception as e:
                    logger.warning(f"样本 {idx+1} 提取ground_truth答案失败: {e}")
                    ground_truth_text, ground_truth_num = "", None
                
                try:
                    student_answer_text, student_answer_num = extract_answer_unified(student_response)
                except Exception as e:
                    logger.warning(f"样本 {idx+1} 提取student答案失败: {e}")
                    student_answer_text, student_answer_num = "", None
                
                # 评估答案正确性（使用数值比较，更准确）
                is_correct = False
                if ground_truth_num is not None and student_answer_num is not None:
                    # 使用数值比较（容忍小误差，与utils/math_utils.py保持一致）
                    tolerance = 1e-6
                    if abs(ground_truth_num) < 1e-10:
                        # 真值接近0，使用绝对误差
                        is_correct = abs(student_answer_num - ground_truth_num) < tolerance
                    else:
                        # 使用相对误差
                        relative_error = abs(student_answer_num - ground_truth_num) / abs(ground_truth_num)
                        is_correct = relative_error < tolerance
                elif ground_truth_text and student_answer_text:
                    # 如果无法提取数字，使用文本比较
                    is_correct = ground_truth_text.strip().lower() == student_answer_text.strip().lower()
                else:
                    # 如果都无法提取，标记为错误
                    is_correct = False
                    if idx < 5:  # 只对前几个样本详细日志
                        logger.debug(f"样本 {idx+1} 无法提取答案: ground_truth_text={ground_truth_text}, student_answer_text={student_answer_text}")
                
                if is_correct:
                    correct_count += 1
                
                # 评估推理质量（如果响应不为空）
                try:
                    if student_response and teacher_response:
                        reasoning_result = reasoning_evaluator.evaluate_reasoning_quality(
                            student_response=student_response,
                            teacher_response=teacher_response,
                            ground_truth_answer=ground_truth_num,
                            student_logits=student_logits,
                            teacher_logits=teacher_logits
                        )
                    else:
                        # 如果响应为空，使用默认值
                        reasoning_result = {
                            "overall_score": 0.0,
                            "step_coverage": {"step_coverage": 0.0},
                            "logical_consistency": {"overall_consistency": 0.0}
                        }
                except Exception as e:
                    logger.warning(f"样本 {idx+1} 推理质量评估失败: {e}")
                    reasoning_result = {
                        "overall_score": 0.0,
                        "step_coverage": {"step_coverage": 0.0},
                        "logical_consistency": {"overall_consistency": 0.0}
                    }
                
                # 评估蒸馏效果（如果响应不为空且logits可用）
                try:
                    if student_response and ground_truth:
                        distillation_result = comprehensive_evaluator.evaluate_comprehensive(
                            predictions=[student_response],
                            ground_truths=[ground_truth],
                            student_logits=student_logits,
                            teacher_logits=teacher_logits
                        )
                    else:
                        distillation_result = {"overall_score": 0.0}
                except Exception as e:
                    logger.warning(f"样本 {idx+1} 蒸馏效果评估失败: {e}")
                    distillation_result = {"overall_score": 0.0}
                
                # 累积指标
                reasoning_score = reasoning_result["overall_score"]
                reasoning_scores.append(reasoning_score)
                
                distillation_score = distillation_result.get("overall_score", 0.0)
                distillation_scores.append(distillation_score)
                
                step_coverage_scores.append(reasoning_result.get("step_coverage", {}).get("step_coverage", 0.0))
                logical_consistency_scores.append(
                    reasoning_result.get("logical_consistency", {}).get("overall_consistency", 0.0)
                )
                
                if "answer_correctness" in reasoning_result:
                    answer_correctness_scores.append(
                        reasoning_result["answer_correctness"].get("correctness_score", 0.0)
                    )
                
                if student_logits is not None and teacher_logits is not None:
                    kl_divergences.append(reasoning_result.get("kl_divergence", 0.0))
                    cosine_similarities.append(distillation_result.get("cosine_similarity", 0.0))
                
                # 保存个体结果（只保存关键信息，避免文件过大）
                individual_result = {
                    "index": idx + 1,
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "ground_truth": ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth,
                    "student_response": student_response[:200] + "..." if len(student_response) > 200 else student_response,
                    "teacher_response": teacher_response[:200] + "..." if len(teacher_response) > 200 else teacher_response,
                    "is_correct": is_correct,
                    "student_answer_text": student_answer_text if student_answer_text else "N/A",
                    "student_answer_num": student_answer_num if student_answer_num is not None else "N/A",
                    "ground_truth_text": ground_truth_text if ground_truth_text else "N/A",
                    "ground_truth_answer_num": ground_truth_num if ground_truth_num is not None else "N/A",
                    "reasoning_score": float(reasoning_score) if isinstance(reasoning_score, (int, float)) else 0.0,
                    "distillation_score": float(distillation_score) if isinstance(distillation_score, (int, float)) else 0.0
                }
                results["individual_results"].append(individual_result)
                
            except Exception as e:
                logger.error(f"评估样本 {idx+1} 时出错: {e}")
                # 记录错误但继续评估
                results["individual_results"].append({
                    "index": idx + 1,
                    "error": str(e),
                    "question": question[:200] if len(question) > 200 else question
                })
        
        # 计算总体指标
        logger.info("=" * 80)
        logger.info("计算总体指标...")
        
        results["accuracy"] = correct_count / total_samples if total_samples > 0 else 0.0
        
        # 推理质量统计
        if reasoning_scores:
            results["reasoning_quality"]["overall_score"] = sum(reasoning_scores) / len(reasoning_scores)
            results["reasoning_quality"]["step_coverage"] = sum(step_coverage_scores) / len(step_coverage_scores) if step_coverage_scores else 0.0
            results["reasoning_quality"]["logical_consistency"] = sum(logical_consistency_scores) / len(logical_consistency_scores) if logical_consistency_scores else 0.0
            results["reasoning_quality"]["answer_correctness"] = sum(answer_correctness_scores) / len(answer_correctness_scores) if answer_correctness_scores else 0.0
            results["reasoning_quality"]["kl_divergence"] = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
        
        # 蒸馏效果统计
        if distillation_scores:
            results["distillation_effect"]["overall_score"] = sum(distillation_scores) / len(distillation_scores)
            results["distillation_effect"]["kl_divergence"] = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
            results["distillation_effect"]["cosine_similarity"] = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
        
        # 总体统计
        results["statistics"]["total_samples"] = total_samples
        results["statistics"]["correct_samples"] = correct_count
        results["statistics"]["incorrect_samples"] = total_samples - correct_count
        results["statistics"]["average_reasoning_score"] = results["reasoning_quality"]["overall_score"]
        results["statistics"]["average_distillation_score"] = results["distillation_effect"]["overall_score"]
        
        # 打印结果摘要
        logger.info("=" * 80)
        logger.info("评估结果摘要")
        logger.info("=" * 80)
        logger.info(f"准确率: {results['accuracy']:.4f} ({correct_count}/{total_samples})")
        logger.info(f"推理质量总分: {results['reasoning_quality']['overall_score']:.4f}")
        logger.info(f"  - 步骤覆盖率: {results['reasoning_quality']['step_coverage']:.4f}")
        logger.info(f"  - 逻辑一致性: {results['reasoning_quality']['logical_consistency']:.4f}")
        logger.info(f"  - 答案正确性: {results['reasoning_quality']['answer_correctness']:.4f}")
        logger.info(f"  - KL散度: {results['reasoning_quality']['kl_divergence']:.4f}")
        logger.info(f"蒸馏效果总分: {results['distillation_effect']['overall_score']:.4f}")
        logger.info(f"  - 余弦相似度: {results['distillation_effect']['cosine_similarity']:.4f}")
        logger.info("=" * 80)
        
        # 保存结果
        output_path = Path(output_file)
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存评估结果到: {output_path.resolve()}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 验证文件已保存
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"✅ 评估完成！结果文件: {output_path.resolve()} ({file_size:.2f} MB)")
        else:
            logger.error(f"❌ 结果文件保存失败: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估RL模型检查点")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="检查点路径（如 checkpoints/rl_model/checkpoint-1000）")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                       help="教师模型路径")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="评估样本数量（None表示全部）")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="结果输出文件路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="日志级别")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径")
    parser.add_argument("--skip_teacher", action="store_true",
                       help="跳过教师模型生成（避免CUDA错误）")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    
    # 检查检查点路径（支持相对路径和绝对路径，VAST AI兼容）
    checkpoint_path_input = Path(args.checkpoint_path)
    project_root = Path(__file__).parent.parent
    current_dir = Path.cwd()
    
    # 尝试多个可能的路径
    possible_paths = [
        checkpoint_path_input,  # 原始路径
        checkpoint_path_input.resolve(),  # 绝对路径解析
        project_root / args.checkpoint_path,  # 相对于项目根目录
        current_dir / args.checkpoint_path,  # 相对于当前工作目录
    ]
    
    checkpoint_path = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            checkpoint_path = path.resolve()
            logger.info(f"找到检查点路径: {checkpoint_path}")
            break
    
    if checkpoint_path is None:
        logger.error(f"❌ 检查点路径不存在: {args.checkpoint_path}")
        logger.error(f"尝试的路径:")
        for path in possible_paths:
            logger.error(f"  - {path} (存在: {path.exists()}, 是目录: {path.is_dir() if path.exists() else 'N/A'})")
        logger.error(f"当前工作目录: {current_dir}")
        logger.error(f"项目根目录: {project_root}")
        logger.error(f"脚本位置: {Path(__file__).parent}")
        sys.exit(1)
    
    # 检查检查点目录中是否有adapter文件
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_weights = [
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "adapter_model.bin"
    ]
    
    if not adapter_config.exists():
        logger.error(f"❌ 检查点缺少配置文件: {adapter_config}")
        sys.exit(1)
    
    if not any(f.exists() for f in adapter_weights):
        logger.error(f"❌ 检查点缺少权重文件")
        logger.error(f"   查找路径: {[str(f) for f in adapter_weights]}")
        logger.error(f"   检查点目录内容: {list(checkpoint_path.iterdir())[:10]}")
        sys.exit(1)
    
    logger.info(f"✅ 检查点验证通过: {checkpoint_path}")
    logger.info(f"   配置文件: ✓ {adapter_config.name}")
    logger.info(f"   权重文件: ✓ {[f.name for f in adapter_weights if f.exists()]}")
    
    try:
        # 加载配置（支持相对路径和绝对路径）
        config_path = Path(args.config)
        if not config_path.exists():
            project_root = Path(__file__).parent.parent
            config_path = project_root / args.config
        if not config_path.exists():
            logger.error(f"配置文件不存在: {args.config}")
            sys.exit(1)
        
        config = load_config(str(config_path))
        logger.info(f"✅ 配置文件加载成功: {config_path}")
        
        # 执行评估
        # 使用绝对路径确保VAST AI兼容性
        results = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path.resolve()),
            teacher_model_path=args.teacher_model_path,
            config=config,
            eval_samples=args.eval_samples,
            output_file=args.output_file,
            skip_teacher=args.skip_teacher
        )
        
        logger.info("=" * 80)
        logger.info("评估任务完成！")
        logger.info(f"结果文件: {args.output_file}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

