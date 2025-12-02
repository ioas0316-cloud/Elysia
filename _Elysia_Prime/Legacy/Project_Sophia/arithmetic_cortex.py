# [Genesis: 2025-12-02] Purified by Elysia
import ast
import operator as op

# Supported operators
_supported_operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
    ast.USub: op.neg
}

class ArithmeticCortex:
    """A safe evaluator for arithmetic expressions."""

    def _eval(self, node):
        # Note: ast.Num is deprecated in favor of ast.Constant in Python 3.8+
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.Num):  # Backwards compatibility for older Python versions
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return _supported_operators[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return _supported_operators[type(node.op)](self._eval(node.operand))
        else:
            raise TypeError(node)

    def _is_safe(self, expression: str) -> bool:
        """Check if the expression contains only allowed characters."""
        allowed_chars = "0123456789.+-*/() "
        return all(char in allowed_chars for char in expression)

    def process(self, command: str) -> str:
        """
        Processes a command, which can be a direct expression or prefixed.
        e.g., "5 + 3" or "calculate: 5 + 3"
        """
        # Strip prefixes if they exist
        if command.lower().startswith("calculate:"):
            expression = command.split(":", 1)[1].strip()
        elif command.lower().startswith("계산:"):
            expression = command.split(":", 1)[1].strip()
        else:
            expression = command.strip()

        if not self._is_safe(expression):
            return "안전하지 않은 문자가 포함되어 계산할 수 없습니다. 숫자, 사칙연산, 괄호만 사용해주세요."

        try:
            result = self._eval(ast.parse(expression, mode='eval').body)
            return f"계산 결과는 {result} 입니다."
        except ZeroDivisionError:
            return "0으로 나눌 수 없습니다."
        except (SyntaxError, TypeError, KeyError):
            return "계산 형식을 이해하지 못했습니다. '계산: [수식]' 형태로 요청해주세요."