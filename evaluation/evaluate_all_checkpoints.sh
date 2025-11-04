#!/bin/bash
# 批量评估所有RL检查点

# 配置
CHECKPOINT_DIR="checkpoints/rl_model"
OUTPUT_DIR="evaluation_results"
EVAL_SAMPLES=100  # 快速测试用100个样本，完整评估请设置为null或注释掉

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 评估所有检查点
echo "开始批量评估检查点..."
echo "检查点目录: $CHECKPOINT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "评估样本数: $EVAL_SAMPLES"
echo ""

# 查找所有checkpoint目录
for checkpoint in $CHECKPOINT_DIR/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        checkpoint_name=$(basename $checkpoint)
        echo "=========================================="
        echo "评估检查点: $checkpoint_name"
        echo "=========================================="
        
        output_file="$OUTPUT_DIR/evaluation_results_${checkpoint_name}.json"
        
        # 构建命令
        if [ -z "$EVAL_SAMPLES" ] || [ "$EVAL_SAMPLES" == "null" ]; then
            # 完整评估（全部样本）
            python evaluation/evaluate_checkpoint.py \
                --checkpoint_path "$checkpoint" \
                --output_file "$output_file"
        else
            # 快速评估（指定样本数）
            python evaluation/evaluate_checkpoint.py \
                --checkpoint_path "$checkpoint" \
                --eval_samples $EVAL_SAMPLES \
                --output_file "$output_file"
        fi
        
        if [ $? -eq 0 ]; then
            echo "✅ $checkpoint_name 评估完成"
        else
            echo "❌ $checkpoint_name 评估失败"
        fi
        echo ""
    fi
done

echo "=========================================="
echo "批量评估完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

