import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.FoundationLayer.Foundation.arithmetic_cortex import ArithmeticCortex

class TestArithmeticCortex(unittest.TestCase):

    def setUp(self):
        self.cortex = ArithmeticCortex()

    def test_safe_addition(self):
        """Test basic addition."""
        response = self.cortex.process("5 + 3")
        self.assertEqual(response, "계산 결과는 8 입니다.")

    def test_safe_subtraction(self):
        """Test basic subtraction."""
        response = self.cortex.process("10 - 4")
        self.assertEqual(response, "계산 결과는 6 입니다.")

    def test_safe_multiplication(self):
        """Test basic multiplication."""
        response = self.cortex.process("6 * 7")
        self.assertEqual(response, "계산 결과는 42 입니다.")

    def test_safe_division(self):
        """Test basic division."""
        response = self.cortex.process("20 / 4")
        self.assertEqual(response, "계산 결과는 5.0 입니다.")

    def test_division_by_zero(self):
        """Test division by zero error handling."""
        response = self.cortex.process("10 / 0")
        self.assertEqual(response, "0으로 나눌 수 없습니다.")

    def test_invalid_expression(self):
        """Test invalid expression error handling."""
        response = self.cortex.process("5 +")
        self.assertEqual(response, "계산 형식을 이해하지 못했습니다. '계산: [수식]' 형태로 요청해주세요.")

    def test_unsafe_expression_import(self):
        """Test detection of unsafe code (import)."""
        response = self.cortex.process("__import__('os').system('echo unsafe')")
        self.assertIn("안전하지 않은 문자가 포함되어 계산할 수 없습니다", response)

    def test_unsafe_expression_variable(self):
        """Test detection of unsafe code (variables)."""
        response = self.cortex.process("a + b")
        self.assertIn("안전하지 않은 문자가 포함되어 계산할 수 없습니다", response)

    def test_korean_command(self):
        """Test handling of Korean commands."""
        response = self.cortex.process("계산: 100 + 200")
        self.assertEqual(response, "계산 결과는 300 입니다.")

if __name__ == '__main__':
    unittest.main()
