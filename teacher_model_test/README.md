# GSM8K 测试

测试 Qwen2.5-32B-Instruct 在 GSM8K 数据集上的表现。

## 快速开始

### 1. 测试教师模型并导出答案（推荐）

`test_qwen32b_gsm8k.py` 是完全独立版本，不依赖项目其他模块，可单独上传到VAST AI运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试（默认10个样本）
bash run_test.sh

# 指定样本数
bash run_test.sh 200

# 或直接使用Python
python test_qwen32b_gsm8k.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --eval_samples 200 \
  --out teacher_gsm8k.jsonl
```

**输出格式**：生成的 `teacher_gsm8k.jsonl` 文件格式与学生模型完全一致，包含：
- `index`: 样本索引
- `question`: 问题
- `prompt`: 完整提示
- `response`: 教师模型生成的回答
- `top_ids`: prompt处top-50的token IDs（用于蒸馏指标计算）
- `top_probs`: prompt处top-50的概率分布（用于蒸馏指标计算）
- `ground_truth`: 标准答案
- `ground_truth_text` / `ground_truth_num`: 提取的标准答案
- `answer_text` / `answer_num`: 模型回答
- `error`: 错误信息（如果有）

### 2. 对比学生和教师结果

运行完成后，下载 `teacher_gsm8k.jsonl`，然后：

```bash
# 在本地使用项目自带的对比脚本
python scripts/compare_student_teacher.py \
  --student_jsonl student_gsm8k.jsonl \
  --teacher_jsonl teacher_gsm8k.jsonl

# 将计算：
# - 学生和教师的准确率
# - KL散度 (student||teacher) 和 (teacher||student)
# - JS散度
# - 余弦相似度
```

## VAST AI 使用流程

### 步骤1：上传文件到VAST AI

**只需要上传这3个文件**：
1. `test_qwen32b_gsm8k.py` - 教师模型测试脚本
2. `requirements.txt` - 依赖列表
3. `run_test.sh` - 快速运行脚本

### 步骤2：在VAST AI上运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试（生成教师模型结果）
bash run_test.sh 200

# 3. 下载生成的 teacher_gsm8k_200samples.jsonl
```

### 步骤3：在本地对比结果

对比脚本**不需要GPU**，在本地运行即可：

```bash
# 在本地项目目录运行
python scripts/compare_student_teacher.py \
  --student_jsonl student_gsm8k.jsonl \
  --teacher_jsonl teacher_gsm8k_200samples.jsonl
```

## 文件说明

- `test_qwen32b_gsm8k.py` - 教师模型测试脚本（完全独立版本，输出JSONL格式）
- `run_test.sh` - 快速运行测试脚本
- `requirements.txt` - 依赖列表

**说明**：
1. `test_qwen32b_gsm8k.py` 是完全独立版本，不依赖项目其他模块，可单独上传到VAST AI运行
2. 内置 `extract_answer_unified` 函数和 `SimpleTeacherModel` 类，无需外部依赖
3. 输出格式与学生模型完全一致，包含 `top_ids` 和 `top_probs` 用于知识蒸馏对比
4. **仅需上传3个文件即可在VAST AI上生成教师模型结果**

