# Intrinsic Reward Distillation for Mathematical Reasoning

基于内在奖励学习的数学推理知识蒸馏项目

## 项目概述

本项目实现了一个创新的教师-学生知识蒸馏框架，使用Qwen-32B-instruct作为教师模型，Qwen-7B-math作为学生模型，通过内在奖励学习(IRL)理论提升数学推理能力。

## 核心理论

- **教师模型logits = Q函数**：基于"Generalist Reward Models: Found Inside Large Language Models"论文理论
- **内在奖励计算**：通过逆软Bellman算子从logits中恢复奖励函数
- **奖励组合**：内在奖励 + 答案正确性奖励

## 项目结构

```
intrinsic_reward_distillation/
├── config/                    # 配置文件
├── data/                      # 数据处理
├── models/                    # 模型封装
├── rewards/                   # 奖励计算
├── training/                  # 训练模块
├── evaluation/                # 评估模块
├── utils/                     # 工具函数
├── scripts/                   # 训练脚本
└── results/                   # 结果输出
```

## 快速开始

### 方式1：一键训练（推荐）

```bash
# 运行完整训练流程（SFT + RL）
./run_experiment.sh
```

脚本会自动：
1. 安装依赖
2. 准备数据
3. 执行 SFT 训练（如果未完成）
4. 执行 RL 训练（如果未完成，自动从检查点恢复）

### 方式2：分步执行

#### 1. 环境安装

```bash
pip install -r requirements.txt
```

#### 2. 数据准备

```bash
python scripts/prepare_data.py
```

#### 3. 监督微调（SFT）

```bash
python scripts/train_sft.py --config config/training_config.yaml
```

#### 4. 强化学习训练（RL）

```bash
# 新训练
python scripts/train_rl.py --config config/training_config.yaml

# 从检查点继续训练
python scripts/train_rl.py \
    --config config/training_config.yaml \
    --resume_from_checkpoint checkpoints/rl_model/checkpoint-500
```

#### 5. 评估（单独进行）

```bash
# 评估单个检查点
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/rl_model \
    --eval_samples 500

# 批量评估所有检查点
./evaluation/evaluate_all_checkpoints.sh
```

## 配置说明

主要配置文件：
- `config/training_config.yaml`: 训练配置（包含所有模块的配置）

## 训练输出

训练完成后，模型保存在：
- **SFT模型**: `./checkpoints/sft_model/`
- **RL模型**: `./checkpoints/rl_model/`
- **RL检查点**: `./checkpoints/rl_model/checkpoint-{step}/`

## 评估

评估脚本位于 `evaluation/` 目录，可单独运行：

- **评估单个检查点**: `evaluation/evaluate_checkpoint.py`
- **批量评估**: `evaluation/evaluate_all_checkpoints.sh`
- **导出答案**: `scripts/export_gsm8k_answers.py`（学生模型）
- **对比分析**: `scripts/compare_student_teacher.py`

## 技术特点

- **理论创新**：首次将IRL理论应用于数学推理知识蒸馏
- **计算优化**：教师logits缓存 + LoRA微调
- **稳定训练**：PPO + KL散度惩罚
- **全面评估**：多维度推理质量评估
