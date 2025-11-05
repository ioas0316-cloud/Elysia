# This file will contain the ArithmeticCortex,
# the part of Elly's brain responsible for understanding mathematical truth.
import re
import math
from Project_Sophia.experience_logger import log_experience

class ArithmeticCortex:
    # A whitelist of safe characters for the expression
    _SAFE_EXPR_PATTERN = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]+$")

    def process(self, message: str) -> str:
        """
        Processes a natural language command for calculation (e.g., "calculate: 5 * 5")
        and returns a formatted string response.
        """
        # Extract the mathematical expression from the command.
        # It supports both "calculate:" and "계산:" prefixes.
        expression_match = re.match(r"^(?:calculate|계산):\s*(.*)", message, re.IGNORECASE)
        if not expression_match:
            return "계산 형식을 이해하지 못했습니다. '계산: [수식]' 형태로 요청해주세요."

        expression = expression_match.group(1).strip()

        # Security check
        if not self._SAFE_EXPR_PATTERN.match(expression):
            thought = f'The expression "{expression}" contains unsafe characters. I cannot evaluate it.'
            log_experience('arithmetic_cortex', 'cognition', {'step': 'security_fail', 'thought': thought, 'expression': expression})
            print(f"[{self.__class__.__name__}] {thought}")
            return f"'{expression}' 에는 안전하지 않은 문자가 포함되어 계산할 수 없습니다."

        try:
            # Using eval() after a strict security check
            result = eval(expression)
            log_experience('arithmetic_cortex', 'action', {'step': 'evaluate_success', 'expression': expression, 'result': result})
            # Format the result into a user-friendly string
            return f"계산 결과는 {result} 입니다."
        except ZeroDivisionError:
            return "0으로 나눌 수 없습니다."
        except Exception as e:
            thought = f'I encountered an error while evaluating "{expression}": {e}.'
            log_experience('arithmetic_cortex', 'error', {'step': 'evaluate_error', 'expression': expression, 'error': str(e)})
            log_experience('arithmetic_cortex', 'cognition', {'step': 'evaluate_fail', 'thought': thought})
            print(f"[{self.__class__.__name__}] {thought}")
            return f"계산식을 이해하지 못했습니다: {e}"


    def evaluate(self, expression: str) -> float | None:
        """
        Safely evaluates a mathematical expression string and returns the result,
        logging its thought process.
        """
        log_experience('arithmetic_cortex', 'cognition', {'step': 'evaluate_start', 'thought': f'I will evaluate the expression: "{expression}".'})

        # Security check
        if not self._SAFE_EXPR_PATTERN.match(expression):
            thought = f'The expression "{expression}" contains unsafe characters. I cannot evaluate it.'
            log_experience('arithmetic_cortex', 'cognition', {'step': 'security_fail', 'thought': thought, 'expression': expression})
            print(f"[{self.__class__.__name__}] {thought}")
            return None
        
        try:
            # Using eval() after a strict security check
            result = eval(expression)
            log_experience('arithmetic_cortex', 'action', {'step': 'evaluate_success', 'expression': expression, 'result': result})
            return result
        except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
            thought = f'I encountered an error while evaluating "{expression}": {e}.'
            log_experience('arithmetic_cortex', 'error', {'step': 'evaluate_error', 'expression': expression, 'error': str(e)})
            log_experience('arithmetic_cortex', 'cognition', {'step': 'evaluate_fail', 'thought': thought})
            print(f"[{self.__class__.__name__}] {thought}")
            return None

    def verify_truth(self, statement: str) -> bool | None:
        """
        Verifies a mathematical statement like "A + B = C", logging its thought process.
        """
        log_experience('arithmetic_cortex', 'cognition', {'step': 'verify_start', 'thought': f'I will verify the statement: "{statement}".'})
        
        match = re.match(r"^\s*(.+?)\s*=\s*(.+?)\s*$", statement)

        if not match:
            log_experience('arithmetic_cortex', 'cognition', {'step': 'parse_fail', 'thought': 'The statement does not seem to be in the "expression = result" format.'})
            return None

        expression_part = match.group(1).strip()
        expected_result_str = match.group(2).strip()
        log_experience('arithmetic_cortex', 'cognition', {'step': 'parse_success', 'thought': f'I have parsed the statement into expression "{expression_part}" and expected result "{expected_result_str}".'})

        try:
            expected_result = float(expected_result_str)
        except ValueError:
            log_experience('arithmetic_cortex', 'cognition', {'step': 'parse_error', 'thought': f'The expected result "{expected_result_str}" is not a valid number.'})
            return None

        actual_result = self.evaluate(expression_part)

        if actual_result is None:
            log_experience('arithmetic_cortex', 'cognition', {'step': 'verify_fail', 'thought': f'I could not evaluate the expression part "{expression_part}". Therefore, I cannot verify the statement.'})
            return None

        is_true = abs(actual_result - expected_result) < 1e-9
        
        thought = f'I calculated "{expression_part}" to be {actual_result}. This is close enough to the expected result {expected_result}. The statement is {is_true}.'
        log_experience('arithmetic_cortex', 'cognition', {'step': 'verification_complete', 'thought': thought, 'is_true': is_true})
        
        return is_true
