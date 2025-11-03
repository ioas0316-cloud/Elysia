# This file will contain the ArithmeticCortex,
# the part of Elly's brain responsible for understanding mathematical truth.
import re
import math
from Project_Sophia.experience_logger import log_experience

class ArithmeticCortex:
    """
    A specialized cortex for handling mathematical calculations.
    It can parse natural language questions to extract and evaluate mathematical expressions.
    """
    # 정규 표현식: 숫자와 연산자로 구성된 계산 요청 패턴을 찾습니다.
    # 예: "5 * 3는?", "계산해줘: 100 / 4", "2 더하기 2"
    _SIMPLE_CALC_PATTERN = re.compile(
        r"(?:계산해줘:|계산:|산수:)?\s*"  # "계산해줘:" 와 같은 접두어 (선택 사항)
        r"((?:\d+\.?\d*\s*[\+\-\*\/]\s*)+\d+\.?\d*)"  # 계산식 그룹 (필수)
        r"(?:은\?|는\?|\?)?"  # "?", "은?", "는?" 과 같은 접미어 (선택 사항)
    )

    _SAFE_EXPR_PATTERN = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]+$")

    def process(self, message: str) -> str | None:
        """
        Processes an incoming message to see if it's a mathematical query.
        If it is, it extracts the expression, evaluates it, and returns a natural language response.
        """
        match = self._SIMPLE_CALC_PATTERN.search(message)
        if not match:
            return None

        expression = match.group(1).strip()
        log_experience('arithmetic_cortex', 'cognition', {'step': 'process_start', 'thought': f'I recognized a mathematical expression "{expression}" in the message "{message}".'})

        result = self.evaluate(expression)

        if result is not None:
            response = f"계산 결과: {expression} = {result}"
            log_experience('arithmetic_cortex', 'action', {'step': 'process_success', 'thought': f'Successfully calculated the result. My response is: "{response}".', 'response': response})
            return response
        else:
            response = f"'{expression}' 식을 계산할 수 없습니다."
            log_experience('arithmetic_cortex', 'cognition', {'step': 'process_fail', 'thought': f'I could not evaluate the expression. My response is: "{response}".', 'response': response})
            return response

    def evaluate(self, expression: str) -> float | None:
        """
        Safely evaluates a mathematical expression string and returns the result.
        """
        log_experience('arithmetic_cortex', 'cognition', {'step': 'evaluate_start', 'thought': f'I will evaluate the expression: "{expression}".'})

        # Security check
        if not self._SAFE_EXPR_PATTERN.match(expression.replace(" ", "")):
            thought = f'The expression "{expression}" contains unsafe characters. I cannot evaluate it.'
            log_experience('arithmetic_cortex', 'cognition', {'step': 'security_fail', 'thought': thought, 'expression': expression})
            return None
        
        try:
            result = eval(expression)
            log_experience('arithmetic_cortex', 'action', {'step': 'evaluate_success', 'expression': expression, 'result': result})
            return result
        except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
            thought = f'I encountered an error while evaluating "{expression}": {e}.'
            log_experience('arithmetic_cortex', 'error', {'step': 'evaluate_error', 'expression': expression, 'error': str(e)})
            return None

    def verify_truth(self, statement: str) -> bool | None:
        """
        Verifies a mathematical statement like "A + B = C".
        """
        log_experience('arithmetic_cortex', 'cognition', {'step': 'verify_start', 'thought': f'I will verify the statement: "{statement}".'})
        
        match = re.match(r"^\s*(.+?)\s*=\s*(.+?)\s*$", statement)
        if not match:
            return None

        expression_part = match.group(1).strip()
        expected_result_str = match.group(2).strip()

        try:
            expected_result = float(expected_result_str)
        except ValueError:
            return None

        actual_result = self.evaluate(expression_part)
        if actual_result is None:
            return None

        return abs(actual_result - expected_result) < 1e-9
