# 项目文件结构说明

## 完整项目结构

```
intrinsic_reward_distillation/
├── README.md                          # 项目说明文档
├── requirements.txt                    # Python依赖包
├── run_experiment.sh                   # 快速启动脚本
├── PROJECT_STRUCTURE.md                # 项目结构说明（本文件）
│
├── config/                            # 配置文件目录
│   ├── __init__.py
│   └── training_config.yaml           # 主训练配置（包含所有模块配置）
│
├── data/                              # 数据处理模块
│   ├── __init__.py
│   └── gsm8k_processor.py             # GSM8K数据集处理器
│
├── models/                            # 模型封装模块
│   ├── __init__.py
│   ├── teacher_model.py               # 教师模型封装
│   ├── student_model.py               # 学生模型封装
│   └── cache_manager.py               # 缓存管理器
│
├── rewards/                           # 奖励计算模块
│   ├── __init__.py
│   ├── intrinsic_reward.py            # 内在奖励计算
│   ├── reward_normalizer.py           # 奖励归一化（新增）
│   └── reward_combiner.py             # 奖励组合
│
├── training/                          # 训练模块
│   ├── __init__.py
│   ├── sft_trainer.py                 # 监督微调训练器
│   ├── rl_trainer.py                  # 强化学习训练器
│   └── ppo_utils.py                   # PPO工具函数
│
├── evaluation/                        # 评估模块
│   ├── __init__.py
│   ├── evaluate_checkpoint.py         # 检查点评估脚本
│   ├── reasoning_evaluator.py         # 推理质量评估
│   ├── metrics.py                     # 评估指标
│   ├── evaluate_all_checkpoints.sh    # 批量评估脚本（Linux）
│   └── evaluate_all_checkpoints.bat   # 批量评估脚本（Windows）
│
├── utils/                             # 工具函数模块
│   ├── __init__.py
│   ├── math_utils.py                  # 数学工具函数
│   └── text_utils.py                  # 文本处理工具
│
├── scripts/                           # 训练脚本
│   ├── __init__.py
│   ├── train_sft.py                   # SFT训练脚本
│   ├── train_rl.py                    # RL训练脚本
│   ├── prepare_data.py                # 数据准备脚本
│   ├── check_rl_training.py           # RL训练检查脚本
│   ├── compare_student_teacher.py     # 学生/教师对比脚本
│   ├── export_gsm8k_answers.py        # 导出学生答案脚本
│   └── export_teacher_gsm8k_answers.py # 导出教师答案脚本
│
├── logs/                              # 日志目录（运行时创建）
├── cache/                             # 缓存目录（运行时创建）
├── results/                           # 结果目录（运行时创建）
└── checkpoints/                       # 模型检查点目录（运行时创建）
```

## 文件功能说明

### 配置文件 (config/)
- **training_config.yaml**: 主训练配置文件，包含所有训练参数（模型、奖励、训练等）

### 数据处理 (data/)
- **gsm8k_processor.py**: GSM8K数学推理数据集的加载、预处理和格式化（直接使用HuggingFace数据集）

### 模型封装 (models/)
- **teacher_model.py**: 封装Qwen-32B-instruct教师模型，提供logits计算和缓存
- **student_model.py**: 封装Qwen-7B-math学生模型，支持LoRA微调和PPO训练
- **cache_manager.py**: 管理教师模型logits缓存，提高训练效率

### 奖励计算 (rewards/)
- **intrinsic_reward.py**: 基于论文理论的内在奖励计算，实现逆软Bellman算子
- **reward_normalizer.py**: 多种奖励归一化方法（mean_std、min_max、z_score、robust等）
- **reward_combiner.py**: 内在奖励和外部奖励的组合，支持自适应权重

### 训练模块 (training/)
- **sft_trainer.py**: 监督微调训练器，在GSM8K上微调学生模型
- **rl_trainer.py**: 强化学习训练器，基于内在奖励的PPO训练
- **ppo_utils.py**: PPO相关工具函数（损失计算、缓冲区、调度器等）

### 评估模块 (evaluation/)
- **reasoning_evaluator.py**: 推理质量评估（步骤覆盖率、逻辑一致性、KL散度）
- **metrics.py**: 综合评估指标（数学准确率、知识蒸馏效果、训练稳定性）

### 工具函数 (utils/)
- **math_utils.py**: 数学相关工具（答案提取、验证、表达式处理等）
- **text_utils.py**: 文本处理工具（清洗、格式化、关键词提取等）

### 训练脚本 (scripts/)
- **train_sft.py**: 监督微调训练脚本
- **train_rl.py**: 强化学习训练脚本（支持从检查点恢复）
- **prepare_data.py**: 数据准备脚本
- **check_rl_training.py**: 检查RL训练是否真实进行
- **compare_student_teacher.py**: 对比学生和教师模型的性能
- **export_gsm8k_answers.py**: 导出学生模型在GSM8K上的答案
- **export_teacher_gsm8k_answers.py**: 导出教师模型在GSM8K上的答案

### 评估脚本 (evaluation/)
- **evaluate_checkpoint.py**: 评估RL检查点的性能（主评估脚本）
- **reasoning_evaluator.py**: 推理质量评估
- **metrics.py**: 综合评估指标
- **evaluate_all_checkpoints.sh/bat**: 批量评估所有检查点

## 使用流程

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用快速启动脚本
chmod +x run_experiment.sh
./run_experiment.sh
```

### 2. 数据准备
```bash
python scripts/prepare_data.py --show_samples 5
```

### 3. 监督微调
```bash
python scripts/train_sft.py --config config/training_config.yaml
```

### 4. 强化学习训练
```bash
python scripts/train_rl.py \
    --student_model_path ./checkpoints/sft_model \
    --max_steps 1000
```

### 5. 模型评估
```bash
# 评估单个检查点
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/rl_model \
    --eval_samples 500

# 批量评估所有检查点（Linux）
./evaluation/evaluate_all_checkpoints.sh

# 批量评估所有检查点（Windows）
evaluation\evaluate_all_checkpoints.bat
```

## 关键特性

### 理论创新
- 基于"Generalist Reward Models"论文的内在奖励理论
- 教师模型logits作为Q函数，通过逆软Bellman算子计算奖励
- 无需额外奖励模型，直接利用教师模型的logits分布

### 技术实现
- 支持LoRA微调，减少计算资源需求
- 教师logits缓存机制，提高训练效率
- 多种奖励归一化方法，确保训练稳定性
- 完整的PPO训练流程，支持变长序列

### 评估体系
- 多维度推理质量评估（步骤覆盖率、逻辑一致性）
- 知识蒸馏效果分析（KL散度、余弦相似度）
- 训练稳定性监控（收敛率、学习效率）

## 配置说明

主要配置参数在`config/training_config.yaml`中：

- **模型配置**: 教师模型、学生模型、LoRA参数
- **训练配置**: 学习率、批次大小、训练步数
- **奖励配置**: 权重组合、归一化方法、温度参数
- **设备配置**: GPU设置、数据类型、内存优化

## 输出结果

训练完成后，项目会生成以下文件：

### 1. 模型检查点目录 (`./checkpoints/`)

#### SFT模型 (`./checkpoints/sft_model/`)
- **adapter_model.safetensors**: LoRA适配器权重 (~50-100MB)
- **adapter_config.json**: LoRA配置参数 (<1MB)
- **tokenizer相关文件**: 分词器配置和词汇表 (~3MB)
- **config.json**: 模型配置文件 (<1MB)

#### RL模型 (`./checkpoints/rl_model/`)

**中间检查点** (每50步保存，共20个):
- **checkpoint-{step}/** 目录，每个包含:
  - adapter_model.safetensors: LoRA适配器 (~50-100MB)
  - adapter_config.json: LoRA配置 (<1MB)
  - tokenizer相关文件 (~3MB)
  - training_stats.json: 训练统计数据 (1-10MB)
  - teacher_cache.pkl: 教师缓存 (~0MB，已禁用)

**自适应权重文件** (每50步保存):
- **adaptive_weights_step_{step}.json**: 自适应权重配置 (<1MB)

**最终模型文件**:
- **adapter_model.safetensors**: 最终LoRA适配器 (~50-100MB)
- **adapter_config.json**: LoRA配置 (<1MB)
- **tokenizer相关文件**: 分词器配置 (~3MB)
- **final_training_stats.json**: 最终训练统计 (~50MB)
- **training_config.yaml**: 完整训练配置 (<1MB)
- **model_info.json**: 模型元信息 (<1MB)

### 2. 评估结果目录 (`./results/`)

运行评估脚本后生成:
- **evaluation_results.json**: 完整评估报告 (包含准确率、推理质量、蒸馏效果等)
- **evaluation_results.checkpoint.json**: 评估检查点 (用于中断恢复)

### 3. 训练日志目录 (`./logs/`)

- **TensorBoard日志**: 训练可视化数据 (<100MB)
- **训练日志文件**: 详细训练过程输出 (<50MB)

### 4. W&B日志 (如果启用)

- **wandb日志**: Weights & Biases跟踪数据 (<50MB)

### 总存储空间估算

完整训练完成后，总存储空间约 **2-5 GB**:
- SFT阶段: ~110MB
- RL检查点: ~2-4GB (20个检查点 × 100-150MB)
- RL最终模型: ~160MB
- 评估结果: ~10-100MB
- 日志文件: ~150MB

**注意事项**: 
- 教师模型缓存已禁用 (命中率0%)
- 可选择性删除旧检查点以减少存储
- 建议只保留最终模型用于推理

这个项目结构完整、模块化设计，便于扩展和维护，适合在VAST AI等GPU云平台上运行。

