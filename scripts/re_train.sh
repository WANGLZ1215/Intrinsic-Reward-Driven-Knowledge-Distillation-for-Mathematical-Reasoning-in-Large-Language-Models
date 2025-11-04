#!/bin/bash

# 重新开始训练脚本
# 适用于VAST AI GPU环境

set -e

echo "=========================================="
echo "重新开始训练（完全从头开始）"
echo "=========================================="

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 获取参数
MAX_STEPS=${1:-1000}
RESET_CHECKPOINTS=${2:-false}

echo "最大步数: $MAX_STEPS"
echo "重置检查点: $RESET_CHECKPOINTS"
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

# 如果要求重置检查点
if [ "$RESET_CHECKPOINTS" = "true" ]; then
    echo ""
    read -p "⚠️  这将删除所有现有检查点，是否继续？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有检查点..."
        rm -rf ./checkpoints/rl_model/checkpoint-*
        echo "✅ 检查点已删除"
    else
        echo "❌ 操作已取消"
        exit 0
    fi
fi

# 开始新训练
echo ""
echo "🔄 开始新的训练..."
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --student_model_path ./checkpoints/sft_model \
    --output_dir ./checkpoints/rl_model \
    --max_steps $MAX_STEPS \
    --log_level INFO

echo "✅ 训练完成"

