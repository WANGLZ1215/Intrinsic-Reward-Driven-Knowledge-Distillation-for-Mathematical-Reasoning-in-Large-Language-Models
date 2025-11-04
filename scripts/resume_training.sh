#!/bin/bash

# 从检查点继续训练脚本
# 适用于VAST AI GPU环境

set -e

echo "=========================================="
echo "从检查点继续训练"
echo "=========================================="

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查检查点参数
if [ -z "$1" ]; then
    echo "用法: ./resume_training.sh <checkpoint_dir> [max_steps]"
    echo "示例: ./resume_training.sh checkpoints/rl_model/checkpoint-500 1000"
    exit 1
fi

CHECKPOINT_DIR=$1
MAX_STEPS=${2:-1000}

# 检查检查点是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 检查点目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "检查点路径: $CHECKPOINT_DIR"
echo "最大步数: $MAX_STEPS"
echo ""

# 设置环境变量
echo "检查GPU状态..."
nvidia-smi || echo "⚠️  nvidia-smi不可用"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "检测到 $NUM_GPUS 个GPU"

if [ "$NUM_GPUS" -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "✅ 使用4×GPU配置"
elif [ "$NUM_GPUS" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "✅ 使用2×GPU配置"
elif [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "⚠️  仅使用1个GPU"
else
    echo "⚠️  未检测到GPU，将使用系统默认配置"
fi

# 从检查点继续训练
echo ""
echo "🔄 从检查点恢复训练..."
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --student_model_path ./checkpoints/sft_model \
    --output_dir ./checkpoints/rl_model \
    --max_steps $MAX_STEPS \
    --resume_from_checkpoint "$CHECKPOINT_DIR" \
    --log_level INFO

echo "✅ 训练完成"

