# RL模型检查点评估工具

本目录包含用于评估RL训练检查点的完整评估工具集。

## 📁 文件结构

```
evaluation/
├── evaluate_checkpoint.py          # 主评估脚本
├── evaluate_all_checkpoints.sh      # 批量评估脚本（Linux/Mac）
├── evaluate_all_checkpoints.bat     # 批量评估脚本（Windows）
├── README.md                        # 本文件
├── MEMORY_REQUIREMENTS.md           # 显存需求详细分析
└── VAST_AI_GUIDE.md                 # VAST AI使用指南
```

## ⚡ 快速开始

### 本地评估（仅学生模型，约20GB显存）

如果您只想评估准确率而不需要与教师模型对比：

```bash
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path checkpoints/rl_model/checkpoint-1000 \
    --eval_samples 100 \
    --output_file evaluation_results.json
```

**注意**：这种方式不会评估蒸馏效果，因为需要同时加载教师模型。

### VAST AI评估（完整评估，推荐）

完整评估需要**90-120 GB显存**，建议使用VAST AI：

1. 租用H200 140GB GPU实例
2. 上传项目文件
3. 运行评估脚本

详细步骤请参考 [VAST_AI_GUIDE.md](VAST_AI_GUIDE.md)

## 📊 显存需求

| 配置 | 显存需求 | 说明 |
|------|---------|------|
| **完整评估** | 90-120 GB | 同时加载教师模型和学生模型 |
| **仅学生模型** | 20 GB | 只评估准确率和推理质量 |
| **量化评估** | 50-60 GB | 使用8bit量化（需修改代码） |

详细分析请查看 [MEMORY_REQUIREMENTS.md](MEMORY_REQUIREMENTS.md)

## 🔧 使用方法

### 基本用法

```bash
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path checkpoints/rl_model/checkpoint-1000 \
    --output_file evaluation_results.json
```

### 完整参数

```bash
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path checkpoints/rl_model/checkpoint-1000 \
    --config config/training_config.yaml \
    --teacher_model_path Qwen/Qwen2.5-32B-Instruct \
    --eval_samples 100 \
    --output_file evaluation_results.json \
    --log_level INFO
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint_path` | ✅ | - | 检查点路径 |
| `--config` | ❌ | `config/training_config.yaml` | 配置文件 |
| `--teacher_model_path` | ❌ | `Qwen/Qwen2.5-32B-Instruct` | 教师模型路径 |
| `--eval_samples` | ❌ | `None`（全部） | 评估样本数 |
| `--output_file` | ❌ | `evaluation_results.json` | 输出文件 |
| `--log_level` | ❌ | `INFO` | 日志级别 |

## 📈 输出格式

评估结果保存在 `evaluation_results.json` 中，包含：

- **准确率**：答案正确率
- **推理质量**：步骤覆盖率、逻辑一致性、答案正确性、KL散度
- **蒸馏效果**：与教师模型的相似度（余弦相似度、KL散度等）
- **统计信息**：总样本数、正确/错误样本数等
- **个体结果**：每个样本的详细评估结果

## 🚀 批量评估

评估所有检查点：

```bash
# Linux/Mac
bash evaluation/evaluate_all_checkpoints.sh

# Windows
evaluation\evaluate_all_checkpoints.bat
```

## 💡 推荐方案

### 方案1：VAST AI（⭐⭐⭐ 强烈推荐）
- **优点**：完全满足显存需求，无需担心硬件限制
- **成本**：完整评估约 $8-16
- **适用**：需要完整评估（包括蒸馏效果）

### 方案2：本地评估（仅学生模型）（⭐⭐）
- **优点**：免费，快速
- **缺点**：无法评估蒸馏效果
- **适用**：仅需评估准确率和推理质量

### 方案3：本地评估（完整）（⭐⭐）
- **优点**：免费
- **缺点**：需要H100/H200等大显存GPU
- **适用**：已有大显存GPU的用户

## 📚 相关文档

- [MEMORY_REQUIREMENTS.md](MEMORY_REQUIREMENTS.md) - 详细的显存需求分析
- [VAST_AI_GUIDE.md](VAST_AI_GUIDE.md) - VAST AI使用详细指南

## ⚠️ 注意事项

1. **显存要求高**：完整评估需要90-120 GB显存
2. **评估时间长**：完整评估1319个样本可能需要1-2小时
3. **检查点格式**：确保检查点包含 `adapter_model.safetensors` 和 `adapter_config.json`
4. **配置文件**：确保配置与训练时一致

## 🐛 故障排除

### 问题1：OOM（显存不足）
- **解决**：使用VAST AI，或减少评估样本数
- **或**：修改脚本使用顺序评估（先学生后教师）

### 问题2：检查点加载失败
- **解决**：检查检查点路径是否正确
- **检查**：确认检查点目录包含adapter文件

### 问题3：评估结果不完整
- **解决**：检查日志文件 `logs/evaluate_checkpoint_*.log`
- **确认**：评估过程中没有异常中断

## 📞 获取帮助

如有问题，请：
1. 查看日志文件
2. 检查 `MEMORY_REQUIREMENTS.md` 了解显存需求
3. 参考 `VAST_AI_GUIDE.md` 了解在线评估流程

