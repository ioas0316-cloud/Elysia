import pytest
import os

from Project_Sophia.arithmetic_cortex import ArithmeticCortex

class TestArithmeticCortexUnits:
    @pytest.fixture
    def cortex(self):
        return ArithmeticCortex()

    def test_evaluate_simple_expressions(self, cortex):
        assert cortex.evaluate("2 * 3") == 6
        assert cortex.evaluate("10 / 2") == 5

    def test_evaluate_with_parentheses(self, cortex):
        assert cortex.evaluate("(2 + 3) * 4") == 20

    def test_evaluate_unsafe_expression(self, cortex):
        assert cortex.evaluate("__import__('os').system('echo unsafe')") is None

    def test_verify_truth(self, cortex):
        assert cortex.verify_truth("2 * 3 = 6") is True
        assert cortex.verify_truth("10 / 2 = 4") is False

def test_pipeline_integration_question(shared_pipeline):
    """
    Tests if the pipeline correctly processes a calculation in a question format.
    """
    main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
    if os.path.exists(main_memory_path):
        os.remove(main_memory_path)

    response, _ = shared_pipeline.process_message("5 * 3는?")
    assert "15" in response['text']

def test_pipeline_integration_command(shared_pipeline):
    """
    Tests if the pipeline correctly processes a calculation in a command format.
    """
    main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
    if os.path.exists(main_memory_path):
        os.remove(main_memory_path)

    response, _ = shared_pipeline.process_message("계산해줘: 100 / 4")
    assert "25" in response['text']
