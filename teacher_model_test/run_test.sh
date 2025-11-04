#!/bin/bash
# 快速运行测试脚本

echo "=========================================="
echo "Qwen2.5-32B-Instruct GSM8K 测试"
echo "=========================================="

# 设置CUDA_LAUNCH_BLOCKING
export CUDA_LAUNCH_BLOCKING=1
echo "✅ 已设置 CUDA_LAUNCH_BLOCKING=1"

# 进入目录
cd "$(dirname "$0")"

# 检查参数
EVAL_SAMPLES=${1:-10}  # 默认10个样本
echo "评估样本数: $EVAL_SAMPLES"
echo ""

# 运行测试
python test_qwen32b_gsm8k.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --eval_samples $EVAL_SAMPLES \
  --max_length 512 \
  --temperature 0.7 \
  --topk_dist 50 \
  --device_map auto \
  --torch_dtype bfloat16 \
  --out "teacher_gsm8k_${EVAL_SAMPLES}samples.jsonl" \
  --log_level INFO

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="

