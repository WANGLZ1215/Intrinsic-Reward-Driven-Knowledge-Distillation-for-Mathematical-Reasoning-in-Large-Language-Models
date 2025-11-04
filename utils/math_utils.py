"""
Mathematical Utility Functions
Function: Provide mathematical utility functions including answer extraction, validation, etc.
"""

import re
import sympy as sp
import numpy as np
from typing import Optional, List, Tuple, Union
import logging


def extract_answer_unified(text: str) -> Tuple[str, Optional[float]]:
    """
    统一的答案提取函数 - 支持多种格式
    这是项目中唯一的答案提取实现，其他模块应调用此函数
    
    支持格式：
    - #### (GSM8K标准格式) - 最高优先级
    - \\boxed{} (LaTeX格式)
    - "answer:" 或 "answer："
    - "The answer is"
    - 兜底：最后一个数字
    
    Args:
        text: 输入文本（包含推理过程和答案）
        
    Returns:
        (答案文本, 答案数字)
        - 答案文本: 提取的原始答案字符串
        - 答案数字: 转换为浮点数，如果无法提取则为None
    """
    if not text:
        return "", None
    
    # 清理文本中的特殊字符
    text_clean = re.sub(r'[^\w\s\.,!?;:()\[\]{}"\'-]', '', text)
    
    # 第一优先级：GSM8K标准的 #### 格式
    matches = re.findall(r"####\s*([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)", text_clean)
    if matches:
        # 使用最后一个匹配（避免示例中的干扰）
        last_match = matches[-1]
        # 检查这个匹配是否在示例之后（避免提取示例中的答案）
        match_pos = text_clean.rfind(f"#### {last_match}")
        example_pos = max(text_clean.rfind("Example"), text_clean.rfind("样例"), text_clean.rfind("example"))
        
        if example_pos == -1 or match_pos > example_pos:
            answer_text = last_match
        else:
            # 如果最后一个匹配在示例中，尝试倒数第二个
            if len(matches) > 1:
                answer_text = matches[-2]
            else:
                answer_text = None
        
        if answer_text:
            # 清理数字格式并转换
            num_clean = answer_text.replace('$', '').replace(',', '').strip()
            if num_clean.endswith('%'):
                num_clean = num_clean[:-1]
            try:
                return answer_text, float(num_clean)
            except ValueError:
                pass
    
    # 第二优先级：\boxed{} 格式
    match = re.search(r"\\boxed\{([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)\}", text_clean)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace('$', '').replace(',', '').strip()
        if num_clean.endswith('%'):
            num_clean = num_clean[:-1]
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 第三优先级："answer:" 或 "answer：" 格式
    match = re.search(r"answer[:：]?\s*([\$]?[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)", text_clean, re.IGNORECASE)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace('$', '').replace(',', '').strip()
        if num_clean.endswith('%'):
            num_clean = num_clean[:-1]
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 第四优先级："The answer is" 格式
    match = re.search(r'The answer is\s*[\$]?([-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        answer_text = match.group(1)
        num_clean = answer_text.replace(',', '').strip()
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    # 兜底方案：提取最后一个数字
    numbers = re.findall(r'[-+]?\d{1,7}(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        answer_text = numbers[-1]
        num_clean = answer_text.replace(',', '')
        try:
            return answer_text, float(num_clean)
        except ValueError:
            pass
    
    return "", None


def extract_final_answer(text: str) -> str:
    """
    从文本中提取最终答案（仅返回文本）
    
    注意：此函数调用统一的 extract_answer_unified 函数
    保留此函数以保持向后兼容性
    
    Args:
        text: 输入文本
        
    Returns:
        提取的答案文本
    """
    answer_text, _ = extract_answer_unified(text)
    return answer_text


def extract_gsm8k_answer(text: str) -> str:
    """
    专门针对GSM8K数据集提取答案
    
    注意：此函数调用统一的 extract_answer_unified 函数
    保留此函数以保持向后兼容性
    
    Args:
        text: 包含推理过程和答案的文本
        
    Returns:
        提取的答案文本
    """
    answer_text, _ = extract_answer_unified(text)
    return answer_text


def validate_gsm8k_answer(student_text: str, correct_answer: str) -> bool:
    """
    验证GSM8K答案是否正确
    
    Args:
        student_text: 学生生成的完整文本（包含推理过程）
        correct_answer: 正确答案
        
    Returns:
        是否正确
    """
    # 从学生文本中提取答案
    student_answer = extract_gsm8k_answer(student_text)
    
    # 比较答案
    return is_answer_correct(student_answer, correct_answer)


def extract_number_from_answer(answer: str) -> Optional[float]:
    """
    从答案中提取数字
    
    Args:
        answer: 答案文本
        
    Returns:
        提取的数字，如果失败返回None
    """
    if not answer:
        return None
    
    # 查找数字
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    
    return None


def is_answer_correct(student_answer: str, correct_answer: str, 
                     tolerance: float = 1e-6) -> bool:
    """
    判断学生答案是否正确
    专门针对GSM8K数据集优化
    
    Args:
        student_answer: 学生答案
        correct_answer: 正确答案
        tolerance: 数值容差
        
    Returns:
        是否正确
    """
    # 首先尝试直接字符串比较（处理整数答案）
    if student_answer.strip() == correct_answer.strip():
        return True
    
    # 提取数字进行比较
    student_num = extract_number_from_answer(student_answer)
    correct_num = extract_number_from_answer(correct_answer)
    
    if student_num is None or correct_num is None:
        # 如果无法提取数字，尝试字符串相似度比较
        return student_answer.strip().lower() == correct_answer.strip().lower()
    
    # 比较数值（处理小数和分数）
    return abs(student_num - correct_num) < tolerance


def validate_math_expression(expression: str) -> bool:
    """
    验证数学表达式是否有效
    
    Args:
        expression: 数学表达式
        
    Returns:
        是否有效
    """
    try:
        # 使用sympy验证表达式
        sp.sympify(expression)
        return True
    except:
        return False


def simplify_math_expression(expression: str) -> str:
    """
    简化数学表达式
    
    Args:
        expression: 数学表达式
        
    Returns:
        简化后的表达式
    """
    try:
        expr = sp.sympify(expression)
        simplified = sp.simplify(expr)
        return str(simplified)
    except:
        return expression


def solve_linear_equation(equation: str) -> Optional[float]:
    """
    解线性方程
    
    Args:
        equation: 线性方程字符串
        
    Returns:
        解，如果无解返回None
    """
    try:
        # 解析方程
        expr = sp.sympify(equation)
        
        # 假设只有一个变量x
        x = sp.Symbol('x')
        
        # 求解
        solutions = sp.solve(expr, x)
        
        if solutions:
            return float(solutions[0])
        
        return None
    except:
        return None


def calculate_arithmetic(expression: str) -> Optional[float]:
    """
    计算算术表达式
    
    Args:
        expression: 算术表达式
        
    Returns:
        计算结果，如果失败返回None
    """
    try:
        # 安全的表达式计算
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return None
        
        result = eval(expression)
        return float(result)
    except:
        return None


def extract_numbers_from_text(text: str) -> List[float]:
    """
    从文本中提取所有数字
    
    Args:
        text: 输入文本
        
    Returns:
        数字列表
    """
    numbers = re.findall(r'-?\d+\.?\d*', text)
    result = []
    
    for num_str in numbers:
        try:
            result.append(float(num_str))
        except ValueError:
            continue
    
    return result


def format_math_answer(number: Union[int, float], precision: int = 2) -> str:
    """
    格式化数学答案
    
    Args:
        number: 数字
        precision: 精度
        
    Returns:
        格式化后的答案字符串
    """
    if isinstance(number, int) or number.is_integer():
        return str(int(number))
    else:
        return f"{number:.{precision}f}".rstrip('0').rstrip('.')


def check_math_reasoning_consistency(response: str) -> float:
    """
    检查数学推理的一致性
    
    Args:
        response: 模型响应
        
    Returns:
        一致性得分 (0-1)
    """
    score = 1.0
    
    # 提取所有数字
    numbers = extract_numbers_from_text(response)
    
    if len(numbers) < 2:
        return 0.5
    
    # 检查数字的合理性
    for i, num in enumerate(numbers):
        # 检查是否有异常大的数字
        if abs(num) > 10000:
            score -= 0.1
        
        # 检查是否有负数（在某些上下文中不合理）
        if num < 0 and i < len(numbers) - 1:  # 最终答案可以是负数
            score -= 0.05
    
    # 检查计算逻辑
    lines = response.split('\n')
    for line in lines:
        if '=' in line:
            # 检查等式两边的计算
            parts = line.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                
                left_num = extract_number_from_answer(left)
                right_num = extract_number_from_answer(right)
                
                if left_num is not None and right_num is not None:
                    if abs(left_num - right_num) > 1e-6:
                        score -= 0.2
    
    return max(0.0, score)


def extract_math_operations(text: str) -> List[str]:
    """
    提取数学操作
    
    Args:
        text: 输入文本
        
    Returns:
        数学操作列表
    """
    operations = []
    
    # 查找数学表达式
    patterns = [
        r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',  # 基本运算
        r'\d+\s*=\s*\d+\s*[+\-*/]\s*\d+',  # 另一种格式
        r'[a-zA-Z]\s*=\s*\d+',              # 变量赋值
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        operations.extend(matches)
    
    return operations


def verify_math_step(step: str, context: str = "") -> bool:
    """
    验证数学推理步骤
    
    Args:
        step: 推理步骤
        context: 上下文
        
    Returns:
        是否有效
    """
    # 检查是否包含数学内容
    if not any(op in step for op in ['+', '-', '*', '/', '=']):
        return False
    
    # 检查是否包含数字
    if not re.search(r'\d+', step):
        return False
    
    # 检查逻辑连接词
    logical_words = ['therefore', 'thus', 'so', 'hence', 'because', 'since', 'given']
    has_logical_word = any(word in step.lower() for word in logical_words)
    
    # 如果包含逻辑连接词，认为更有效
    return True


def calculate_confidence_score(student_answer: str, 
                              teacher_answer: str,
                              reasoning_quality: float) -> float:
    """
    计算置信度得分
    
    Args:
        student_answer: 学生答案
        teacher_answer: 教师答案
        reasoning_quality: 推理质量得分
        
    Returns:
        置信度得分 (0-1)
    """
    # 答案正确性权重
    answer_correctness = 1.0 if is_answer_correct(student_answer, teacher_answer) else 0.0
    
    # 推理质量权重
    reasoning_weight = reasoning_quality
    
    # 综合得分
    confidence = 0.7 * answer_correctness + 0.3 * reasoning_weight
    
    return confidence


def format_reasoning_steps(steps: List[str]) -> str:
    """
    格式化推理步骤
    
    Args:
        steps: 推理步骤列表
        
    Returns:
        格式化后的推理文本
    """
    formatted_steps = []
    
    for i, step in enumerate(steps, 1):
        # 确保步骤有编号
        if not step.strip().startswith(('Step', str(i), 'First', 'Then')):
            step = f"Step {i}: {step}"
        
        formatted_steps.append(step)
    
    return '\n'.join(formatted_steps)


def extract_problem_type(question: str) -> str:
    """
    提取问题类型
    
    Args:
        question: 问题文本
        
    Returns:
        问题类型
    """
    question_lower = question.lower()
    
    # 算术问题
    if any(word in question_lower for word in ['add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'times']):
        return "arithmetic"
    
    # 代数问题
    if any(word in question_lower for word in ['solve', 'equation', 'variable', 'x =', 'y =']):
        return "algebra"
    
    # 几何问题
    if any(word in question_lower for word in ['area', 'perimeter', 'volume', 'triangle', 'circle', 'rectangle']):
        return "geometry"
    
    # 概率问题
    if any(word in question_lower for word in ['probability', 'chance', 'likely', 'unlikely']):
        return "probability"
    
    # 应用题
    if any(word in question_lower for word in ['buy', 'sell', 'cost', 'price', 'money', 'dollar']):
        return "word_problem"
    
    return "general"


def get_difficulty_score(question: str, answer: str) -> float:
    """
    估计问题难度
    
    Args:
        question: 问题文本
        answer: 答案文本
        
    Returns:
        难度得分 (1-5)
    """
    difficulty = 1.0
    
    # 基于问题长度
    if len(question) > 200:
        difficulty += 0.5
    
    # 基于数字数量
    numbers_in_question = len(extract_numbers_from_text(question))
    if numbers_in_question > 5:
        difficulty += 0.5
    
    # 基于推理步骤数量
    lines_in_answer = len([line for line in answer.split('\n') if line.strip()])
    if lines_in_answer > 5:
        difficulty += 0.5
    
    # 基于问题类型
    problem_type = extract_problem_type(question)
    if problem_type in ['algebra', 'geometry']:
        difficulty += 0.5
    elif problem_type == 'word_problem':
        difficulty += 0.3
    
    # 基于数学操作的复杂性
    operations = extract_math_operations(answer)
    complex_ops = [op for op in operations if any(symbol in op for symbol in ['*', '/', '^', '**'])]
    if len(complex_ops) > 2:
        difficulty += 0.5
    
    return min(5.0, difficulty)

