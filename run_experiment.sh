#!/bin/bash

# 内在奖励知识蒸馏实验脚本
# 适用于VAST AI GPU环境
# 支持智能跳过已完成步骤

set -e

echo "=========================================="
echo "内在奖励知识蒸馏实验"
echo "=========================================="

# 设置环境变量 - 自动检测GPU数量
echo "检查GPU状态..."
nvidia-smi || echo "⚠️  nvidia-smi不可用"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "检测到 $NUM_GPUS 个GPU"

# 根据实际GPU数量设置CUDA_VISIBLE_DEVICES
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
    # 如果没有检测到GPU，不设置CUDA_VISIBLE_DEVICES，让Python代码处理
    echo "⚠️  未检测到GPU，将使用系统默认配置"
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 🔥 修复显存碎片化问题

# 🔍 CUDA错误诊断：启用同步执行以获得更详细的错误信息
# 注意：这会让CUDA操作变慢，但能准确定位错误位置
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}  # 默认关闭，需要时设置为1
if [ "$CUDA_LAUNCH_BLOCKING" = "1" ]; then
    echo "⚠️  CUDA_LAUNCH_BLOCKING已启用，这将提供更详细的CUDA错误信息但会降低性能"
fi

# 创建必要的目录
mkdir -p logs
mkdir -p cache
mkdir -p checkpoints

# 检查步骤完成状态的函数
check_sft_completed() {
    if [ -d "./checkpoints/sft_model" ]; then
        # 检查LoRA模式（adapter_model.bin）或完整模型模式（pytorch_model.bin）
        if [ -f "./checkpoints/sft_model/adapter_model.bin" ] || \
           [ -f "./checkpoints/sft_model/adapter_model.safetensors" ] || \
           [ -f "./checkpoints/sft_model/pytorch_model.bin" ]; then
            return 0  # SFT已完成
        fi
    fi
    return 1  # SFT未完成
}

check_rl_completed() {
    if [ -d "./checkpoints/rl_model" ]; then
        # 检查LoRA模式（adapter_model.safetensors或adapter_model.bin）
        if [ -f "./checkpoints/rl_model/adapter_model.safetensors" ] || \
           [ -f "./checkpoints/rl_model/adapter_model.bin" ]; then
            return 0  # RL已完成（有最终模型）
        fi
    fi
    return 1  # RL未完成或只有检查点
}

find_latest_rl_checkpoint() {
    # 查找最新的检查点目录
    latest_checkpoint=""
    latest_step=0
    
    if [ -d "./checkpoints/rl_model" ]; then
        for checkpoint_dir in ./checkpoints/rl_model/checkpoint-*; do
            if [ -d "$checkpoint_dir" ]; then
                # 检查是否有adapter文件（safetensors或bin）
                if [ -f "$checkpoint_dir/adapter_model.safetensors" ] || [ -f "$checkpoint_dir/adapter_model.bin" ]; then
                    # 提取步数（从checkpoint-N目录名）
                    step=$(echo "$checkpoint_dir" | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+')
                    if [ -n "$step" ] && [ "$step" -gt "$latest_step" ] 2>/dev/null; then
                        latest_step=$step
                        latest_checkpoint="$checkpoint_dir"
                    fi
                fi
            fi
        done
    fi
    
    echo "$latest_checkpoint"
}

echo "1. 安装依赖包..."
pip install -r requirements.txt

echo "2. 准备数据..."
python scripts/prepare_data.py --show_samples 5

echo "3. 监督微调 (SFT)..."
if check_sft_completed; then
    echo "✅ SFT训练已完成，跳过此步骤"
    echo "SFT模型路径: ./checkpoints/sft_model"
else
    echo "🔄 开始SFT训练..."
    python scripts/train_sft.py \
        --config config/training_config.yaml \
        --output_dir ./checkpoints/sft_model \
        --log_level INFO
    echo "✅ SFT训练完成"
fi

echo "4. 强化学习训练 (RL)..."
if check_rl_completed; then
    echo "✅ RL训练已完成，跳过此步骤"
    echo "RL模型路径: ./checkpoints/rl_model"
else
    # 检查是否有检查点可以恢复
    latest_checkpoint=$(find_latest_rl_checkpoint)
    if [ -n "$latest_checkpoint" ] && [ -d "$latest_checkpoint" ]; then
        echo "🔄 检测到检查点，从检查点恢复训练..."
        echo "检查点路径: $latest_checkpoint"
        python scripts/train_rl.py \
            --config config/training_config.yaml \
            --student_model_path ./checkpoints/sft_model \
            --output_dir ./checkpoints/rl_model \
            --max_steps 1000 \
            --resume_from_checkpoint "$latest_checkpoint" \
            --log_level INFO
    else
        echo "🔄 开始新的RL训练..."
        python scripts/train_rl.py \
            --config config/training_config.yaml \
            --student_model_path ./checkpoints/sft_model \
            --output_dir ./checkpoints/rl_model \
            --max_steps 1000 \
            --log_level INFO
    fi
    echo "✅ RL训练完成"
fi

echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "📁 模型保存位置："
echo "   - SFT模型: ./checkpoints/sft_model"
echo "   - RL模型: ./checkpoints/rl_model"
echo ""
echo "📊 后续评估："
echo "   评估脚本位于 evaluation/ 目录，可单独运行评估"
echo "   - 评估单个检查点: python evaluation/evaluate_checkpoint.py --checkpoint_path <路径>"
echo "   - 批量评估: ./evaluation/evaluate_all_checkpoints.sh"
echo ""
echo "=========================================="
