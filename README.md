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

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
python scripts/prepare_data.py
```

### 3. 监督微调

```bash
python scripts/train_sft.py --config config/training_config.yaml
```

### 4. 强化学习训练

```bash
python scripts/train_rl.py --config config/training_config.yaml
```

### 5. 评估

```bash
python evaluation/evaluate_checkpoint.py --checkpoint_path ./checkpoints/rl_model
```

## 配置说明

主要配置文件：
- `config/training_config.yaml`: 训练配置（包含所有模块的配置）

## 实验结果

项目将输出以下结果：
- GSM8K准确率提升
- 推理质量评估
- 知识蒸馏效果分析
- 训练效率对比

## 技术特点

- **理论创新**：首次将IRL理论应用于数学推理知识蒸馏
- **计算优化**：教师logits缓存 + LoRA微调
- **稳定训练**：PPO + KL散度惩罚
- **全面评估**：多维度推理质量评估
