"""
GSM8K Dataset Processing Module
Function: Load, preprocess and format GSM8K mathematical reasoning dataset
"""  # 顶部模块文档字符串：说明本文件的用途与功能范围

import json  # 标准库：处理 JSON（本文件当前未直接使用，保留以备扩展）
import re  # 正则表达式，用于从文本中提取答案/数字等
from typing import List, Dict, Tuple, Optional  # 类型注解：提高可读性与工具友好
from datasets import load_dataset, Dataset  # Hugging Face Datasets：加载/操作数据集
from transformers import PreTrainedTokenizer  # 分词器类型注解，确保传入的是预训练分词器
import torch  # PyTorch：创建 DataLoader、张量操作等
from utils.math_utils import extract_answer_unified  # 导入统一的答案提取函数


class GSM8KProcessor:  # 定义一个数据处理器类，专门面向 GSM8K 数据集
    """GSM8K Dataset Processor"""  # 类级文档：概述职责（加载/格式化/预处理）

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):  # 构造函数：注入分词器与最大长度
        self.tokenizer = tokenizer  # 保存分词器实例，供后续格式化与分词使用
        self.max_length = max_length  # 保存最大长度，控制输入序列截断

    def load_dataset(self, dataset_name: str = "gsm8k", config: str = "main") -> Dict[str, Dataset]:  # 加载数据集的方法
        """加载GSM8K数据集"""  # 简短说明：返回一个包含train/test的字典
        try:  # 异常捕获：防止网络/权限问题导致崩溃
            dataset = load_dataset(dataset_name, config)  # 调用HF的load_dataset下载并缓存数据
            print(f"Dataset loaded successfully: {dataset_name}")  # 进度提示：数据集已成功加载
            print(f"Training set size: {len(dataset['train'])}")  # 输出训练集样本数
            print(f"Test set size: {len(dataset['test'])}")  # 输出测试集样本数
            return dataset  # 返回包含各split的DatasetDict
        except Exception as e:  # 捕捉异常
            print(f"Failed to load dataset: {e}")  # 打印错误信息
            raise  # 向上抛出，让上层决定如何处理

    def extract_answer(self, answer_text: str) -> Tuple[str, float]:  # 从带推理的文本中提取最终答案及其数字
        """
        从答案文本中提取最终答案数字
        
        注意：此方法现在调用 utils.math_utils.extract_answer_unified 统一实现
        支持多种格式：#### (GSM8K标准), \\boxed{}, "answer:", "The answer is"

        Args:
            answer_text: 包含推理过程和最终答案的文本

        Returns:
            (最终答案文本, 答案数字)
        """
        answer_text_result, answer_num = extract_answer_unified(answer_text)
        
        # 如果数字为None，返回0.0以保持向后兼容
        if answer_num is None:
            answer_num = 0.0
        
        return answer_text_result, answer_num

    def format_prompt(self, question: str) -> str:  # 组装给模型的提示词格式
        """
        格式化问题为模型输入格式

        Args:
            question: 原始问题

        Returns:
            格式化后的提示文本
        """
        return f"Question: {question}\nAnswer: "  # 统一格式：问题 + 换行 + "Answer: "（便于模型续写）

    def format_full_response(self, question: str, answer: str) -> str:  # 拼接完整的训练目标文本
        """
        格式化完整响应（问题+答案）

        Args:
            question: 问题
            answer: 答案

        Returns:
            格式化后的完整文本
        """
        prompt = self.format_prompt(question)  # 先得到标准化的提示部分
        return prompt + answer + self.tokenizer.eos_token  # 拼接答案并追加eos，利于语言模型学习停止

    def preprocess_sft_data(self, examples: Dict) -> Dict[str, List[str]]:  # 为SFT准备“纯文本”字段
        """
        预处理SFT训练数据

        Args:
            examples: 包含question和answer的字典

        Returns:
            预处理后的文本数据
        """
        texts = []  # 用于存放拼接后的完整训练样本文本
        for question, answer in zip(examples["question"], examples["answer"]):  # 遍历成对的问答
            full_text = self.format_full_response(question, answer)  # 组装“Question…Answer…EOS”
            texts.append(full_text)  # 收集到列表
        return {"text": texts}  # 返回一个带"text"键的字典，方便HF datasets的map使用

    def preprocess_rl_data(self, examples: Dict) -> Dict[str, List]:  # 为RL准备问题、原答案与数值真值
        """
        预处理RL训练数据

        Args:
            examples: 包含question和answer的字典

        Returns:
            预处理后的RL数据
        """
        questions = []  # 存模型输入提示（只到“Answer: ”为止，用于让模型生成）
        answers = []  # 存原始参考答案（教师/数据集）
        correct_answers = []  # 存数值型真值（用于正确性奖励）

        for question, answer in zip(examples["question"], examples["answer"]):  # 逐条处理
            questions.append(self.format_prompt(question))  # 存入格式化后的提示
            answers.append(answer)  # 存入原始答案（保留推理文本）
            # 从答案中提取数值真值（用于计算正确性奖励）
            _, correct_num = self.extract_answer(answer)  # 取出浮点数形式
            correct_answers.append(correct_num)  # 收集数值真值

        return {
            "questions": questions,  # RL输入提示
            "answers": answers,  # 参考答案文本（可选用于对比/展示）
            "correct_answers": correct_answers  # 数值真值（打分用）
        }

    def tokenize_function(self, examples: Dict) -> Dict:  # 通用分词函数，给HF datasets的map调用
        """
        分词函数

        Args:
            examples: 包含text的字典

        Returns:
            分词后的数据
        """
        return self.tokenizer(  # 使用注入的分词器进行批量分词
            examples["text"],  # 批量文本
            truncation=True,  # 超长截断
            padding=True,  # 批内对齐填充
            max_length=self.max_length,  # 统一最大长度
            return_tensors="pt"  # 返回 PyTorch 张量（注意：有些流水线更偏好后处理再转tensor）
        )

    def create_data_collator(self):  # 创建语言建模的数据整理器（用于Trainer/DataLoader）
        """创建数据整理器"""  # 简述用途：自动创建labels并对齐padding
        from transformers import DataCollatorForLanguageModeling  # 延迟导入：避免顶层依赖过多

        return DataCollatorForLanguageModeling(  # 返回一个适配Causal LM的collator
            tokenizer=self.tokenizer,  # 分词器（决定pad与特殊token）
            mlm=False,  # 关闭MLM（因我们是自回归语言模型）
            pad_to_multiple_of=8,  # padding到8的倍数（利于部分硬件上的tensor core效率）
        )

    def validate_data(self, dataset: Dataset, num_samples: int = 5) -> None:  # 简易数据健康检查
        """
        验证数据集质量

        Args:
            dataset: 数据集
            num_samples: 验证样本数量
        """
        print(f"Validating dataset, sample count: {len(dataset)}")  # 打印总量，知道在看哪批数据

        for i in range(min(num_samples, len(dataset))):  # 只抽前num_samples条
            sample = dataset[i]  # 取单条样本
            print(f"\nSample {i+1}:")  # 标记索引（从1开始更直观）
            print(f"Question: {sample.get('question', 'N/A')[:100]}...")  # 打印问题前100字符
            print(f"Answer: {sample.get('answer', 'N/A')[:100]}...")  # 打印答案前100字符

            if 'answer' in sample:  # 如果样本里有答案字段
                answer_text, answer_num = self.extract_answer(sample['answer'])  # 尝试提取答案与其数值
                print(f"Extracted answer: {answer_text} (number: {answer_num})")  # 直观看是否解析正确


class MathDataUtils:  # 与数据处理相关的通用小工具类
    """Mathematical Data Processing Tools"""  # 文档：说明包含表达式校验/数字提取/准确性判定等

    @staticmethod  # 静态方法：仅用到传入参数
    def is_valid_math_expression(expression: str) -> bool:  # 判断数学表达式是否合法
        """检查数学表达式是否有效"""
        try:  # 使用 sympy 进行语法级校验
            # 简单的数学表达式检查
            import sympy as sp  # 局部导入：减少模块加载成本
            sp.sympify(expression)  # 尝试解析表达式
            return True  # 成功解析即视为有效
        except:  # 解析异常则判无效
            return False

    @staticmethod
    def extract_numbers(text: str) -> List[float]:  # 从任意文本中抽取所有数字
        """从文本中提取所有数字"""
        numbers = re.findall(r'-?\d+\.?\d*', text)  # 匹配整数/小数/负数
        return [float(num) for num in numbers if num]  # 转成浮点列表（过滤空字符串）

    @staticmethod
    def calculate_answer_accuracy(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:  # 粗略答案准确性
        """
        计算答案准确性

        Args:
            predicted: 预测答案
            ground_truth: 正确答案
            tolerance: 容差

        Returns:
            是否准确
        """
        try:  # 捕捉异常，保证鲁棒
            pred_numbers = MathDataUtils.extract_numbers(predicted)  # 从预测文本中取所有数字
            gt_numbers = MathDataUtils.extract_numbers(ground_truth)  # 从真值文本中取所有数字

            if not pred_numbers or not gt_numbers:  # 若任何一侧没数字，无法数值对比
                return False  # 返回不正确（可按需更换为文本相等判定）

            pred_answer = pred_numbers[-1]  # 默认取最后一个数字为最终答案
            gt_answer = gt_numbers[-1]  # 真值同理
            return abs(pred_answer - gt_answer) < tolerance  # 在容差内视为正确
        except:  # 发生异常（格式/解析问题）即视为不正确
            return False


def create_dataloader(dataset: Dataset, tokenizer: PreTrainedTokenizer,
                     batch_size: int = 8, shuffle: bool = True) -> torch.utils.data.DataLoader:  # 生成可训练的DataLoader
    """
    创建数据加载器

    Args:
        dataset: 数据集
        tokenizer: 分词器
        batch_size: 批次大小
        shuffle: 是否打乱数据

    Returns:
        数据加载器
    """
    from torch.utils.data import DataLoader  # 局部导入：避免顶层引入过多依赖

    def collate_fn(batch):  # 自定义聚合函数：将原始样本拼成模型输入
        # 提取文本
        texts = [item["text"] for item in batch]  # 从每个样本字典中拿到"text"字段（需确保上游map已生成）

        # 分词
        tokenized = tokenizer(  # 使用传入的分词器进行批量编码
            texts,
            truncation=True,  # 截断超长
            padding=True,  # 动态padding
            max_length=512,  # 固定最大长度（可与processor.max_length保持一致）
            return_tensors="pt"  # 返回PyTorch张量
        )

        # 创建标签（与输入相同）——自回归语言模型常用设置
        labels = tokenized["input_ids"].clone()  # 将输入id复制作为labels（即让模型预测下一个token）

        return {
            "input_ids": tokenized["input_ids"],  # 模型输入ids
            "attention_mask": tokenized["attention_mask"],  # 对应的attention mask
            "labels": labels  # 训练目标（shift在模型/训练器内部处理）
        }

    return DataLoader(  # 返回PyTorch的数据加载器
        dataset,  # HF Dataset对象
        batch_size=batch_size,  # 批大小
        shuffle=shuffle,  # 是否打乱（训练时通常为True）
        collate_fn=collate_fn,  # 指定自定义的聚合函数
        num_workers=2  # DataLoader工作进程数（视环境调优）
    )
