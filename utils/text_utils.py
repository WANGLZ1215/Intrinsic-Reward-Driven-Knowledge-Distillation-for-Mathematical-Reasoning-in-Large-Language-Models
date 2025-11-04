"""
文本处理工具函数
功能：提供文本处理、清洗、格式化等工具函数

本模块提供：
1. 通用文本清洗和格式化功能
2. 文本统计和分析功能
3. 文本分割和提取功能
4. 简化的数学文本处理（复杂功能请使用 math_utils.py）

注意：
- 数学答案提取功能已统一到 math_utils.py 的 extract_answer_unified()
- 本文件的数学相关函数主要用于简单的文本预处理
- 复杂的数学计算和验证请使用 math_utils.py
"""

import re
import string
from typing import List, Optional, Dict, Union
import logging


def clean_text(text: str, remove_extra_whitespace: bool = True,
               remove_special_chars: bool = False) -> str:
    """
    清洗文本
    
    Args:
        text: 输入文本
        remove_extra_whitespace: 是否移除多余空白
        remove_special_chars: 是否移除特殊字符
        
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text.strip())
    
    # 移除特殊字符
    if remove_special_chars:
        # 保留字母、数字、基本标点和空格
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    标准化空白字符
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    
    # 替换所有空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    提取句子
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
    """
    if not text:
        return []
    
    # 简单的句子分割（基于句号、问号、感叹号）
    sentences = re.split(r'[.!?]+', text)
    
    # 清理和过滤
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_paragraphs(text: str) -> List[str]:
    """
    提取段落
    
    Args:
        text: 输入文本
        
    Returns:
        段落列表
    """
    if not text:
        return []
    
    # 按双换行符分割段落
    paragraphs = text.split('\n\n')
    
    # 清理和过滤
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def remove_punctuation(text: str, keep_basic: bool = True) -> str:
    """
    移除标点符号
    
    Args:
        text: 输入文本
        keep_basic: 是否保留基本标点
        
    Returns:
        移除标点后的文本
    """
    if not text:
        return ""
    
    if keep_basic:
        # 保留基本标点
        punctuation = string.punctuation.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        text = text.translate(str.maketrans('', '', punctuation))
    else:
        # 移除所有标点
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


def extract_numbers(text: str, include_decimals: bool = True) -> List[str]:
    """
    提取数字
    
    Args:
        text: 输入文本
        include_decimals: 是否包含小数
        
    Returns:
        数字字符串列表
    """
    if not text:
        return []
    
    if include_decimals:
        pattern = r'-?\d+\.?\d*'
    else:
        pattern = r'-?\d+'
    
    numbers = re.findall(pattern, text)
    return numbers


def extract_words(text: str, min_length: int = 1) -> List[str]:
    """
    提取单词
    
    Args:
        text: 输入文本
        min_length: 最小长度
        
    Returns:
        单词列表
    """
    if not text:
        return []
    
    # 提取单词（字母和数字）
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 过滤最小长度
    words = [w for w in words if len(w) >= min_length]
    
    return words


def count_words(text: str) -> int:
    """
    统计单词数量
    
    Args:
        text: 输入文本
        
    Returns:
        单词数量
    """
    words = extract_words(text)
    return len(words)


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    统计字符数量
    
    Args:
        text: 输入文本
        include_spaces: 是否包含空格
        
    Returns:
        字符数量
    """
    if not text:
        return 0
    
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(' ', ''))


def find_keywords(text: str, keywords: List[str], 
                 case_sensitive: bool = False) -> Dict[str, List[int]]:
    """
    查找关键词
    
    Args:
        text: 输入文本
        keywords: 关键词列表
        case_sensitive: 是否区分大小写
        
    Returns:
        关键词及其位置
    """
    if not text or not keywords:
        return {}
    
    if not case_sensitive:
        text = text.lower()
        keywords = [kw.lower() for kw in keywords]
    
    result = {}
    
    for keyword in keywords:
        positions = []
        start = 0
        
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            result[keyword] = positions
    
    return result


def format_text_for_model(text: str, max_length: Optional[int] = None,
                         add_prefix: str = "", add_suffix: str = "") -> str:
    """
    为模型格式化文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
        add_prefix: 添加前缀
        add_suffix: 添加后缀
        
    Returns:
        格式化后的文本
    """
    # 清洗文本
    text = clean_text(text)
    
    # 添加前缀和后缀
    if add_prefix:
        text = add_prefix + text
    if add_suffix:
        text = text + add_suffix
    
    # 截断到最大长度
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # 在单词边界截断
    
    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000,
                          overlap: int = 100) -> List[str]:
    """
    将文本分割成块
    
    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap: 重叠大小
        
    Returns:
        文本块列表
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在单词边界分割
        if end < len(text):
            # 向前查找空格
            while end > start and text[end] not in ' \n\t':
                end -= 1
            
            # 如果找不到空格，强制分割
            if end == start:
                end = start + chunk_size
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 计算下一个块的起始位置
        start = end - overlap if end < len(text) else end
    
    return chunks


def extract_math_expressions(text: str) -> List[str]:
    """
    提取数学表达式（简化版）
    
    注意：此函数用于简单的文本预处理
    更复杂的数学表达式提取请使用 math_utils.extract_math_operations
    
    Args:
        text: 输入文本
        
    Returns:
        数学表达式列表
    """
    if not text:
        return []
    
    # 数学表达式模式（基本模式）
    patterns = [
        r'\d+\s*[+\-*/]\s*\d+',           # 基本运算
        r'\d+\s*=\s*\d+',                 # 等式
        r'[a-zA-Z]\s*=\s*\d+',            # 变量赋值
    ]
    
    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        expressions.extend(matches)
    
    return list(set(expressions))  # 去重


def clean_math_text(text: str) -> str:
    """
    清洗数学文本
    
    Args:
        text: 输入文本
        
    Returns:
        清洗后的数学文本
    """
    if not text:
        return ""
    
    # 标准化数学符号
    text = text.replace('×', '*')
    text = text.replace('÷', '/')
    text = text.replace('²', '^2')
    text = text.replace('³', '^3')
    
    # 移除多余的空白
    text = normalize_whitespace(text)
    
    # 确保运算符周围有空格
    text = re.sub(r'(\d)\s*([+\-*/=])\s*(\d)', r'\1 \2 \3', text)
    
    return text


def format_math_answer(answer: str) -> str:
    """
    格式化数学答案（简化版）
    
    注意：此函数用于简单的答案格式化
    更复杂的数学答案格式化请使用 math_utils.format_math_answer
    
    Args:
        answer: 答案文本
        
    Returns:
        格式化后的答案
    """
    if not answer:
        return ""
    
    # 延迟导入避免循环依赖
    from utils.math_utils import format_math_answer as format_math_answer_util
    
    # 如果答案是纯数字或简单的数字字符串，使用 math_utils 格式化
    try:
        num = float(answer.strip())
        return format_math_answer_util(num)
    except ValueError:
        # 如果不是纯数字，返回原答案
        return answer.strip()


def validate_text_format(text: str, expected_format: str = "math") -> bool:
    """
    验证文本格式
    
    Args:
        text: 输入文本
        expected_format: 期望格式
        
    Returns:
        是否符合格式
    """
    if not text:
        return False
    
    if expected_format == "math":
        # 检查是否包含数学内容
        has_numbers = bool(re.search(r'\d+', text))
        has_operators = bool(re.search(r'[+\-*/=]', text))
        return has_numbers and has_operators
    
    elif expected_format == "reasoning":
        # 检查是否包含推理关键词
        reasoning_words = ['step', 'first', 'then', 'therefore', 'so', 'thus', 'hence']
        return any(word in text.lower() for word in reasoning_words)
    
    return True


def extract_final_answer_from_text(text: str) -> str:
    """
    从文本中提取最终答案（文本格式）
    
    注意：此函数调用 math_utils.extract_answer_unified 统一实现
    避免代码重复，确保逻辑一致性
    
    Args:
        text: 输入文本
        
    Returns:
        最终答案文本
    """
    # 延迟导入避免循环依赖
    from utils.math_utils import extract_answer_unified
    
    answer_text, _ = extract_answer_unified(text)
    return answer_text
