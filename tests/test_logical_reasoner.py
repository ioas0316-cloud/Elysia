import pytest
import os
from pathlib import Path

from tools import kg_manager
from tests.mocks.mock_local_llm_cortex import MockLocalLLMCortex

@pytest.fixture
def setup_test_kg(tmp_path):
    """
    Sets up and tears down a temporary Knowledge Graph for testing.
    """
    test_kg_path = tmp_path / "test_kg.json"
    original_kg_path = kg_manager.KG_PATH
    kg_manager.KG_PATH = test_kg_path

    kg_instance = kg_manager.KGManager()
    kg_instance.add_node("소크라테스", properties={"description": "고대 그리스의 철학자"})
    kg_instance.add_node("인간")
    kg_instance.add_edge("소크라테스", "인간", "is_a")
    kg_instance.save()

    yield kg_instance

    kg_manager.KG_PATH = original_kg_path


def test_reasoning_and_response(shared_pipeline, setup_test_kg):
    """
    Tests that the CognitionPipeline can generate a logical response based on the KG.
    """
    main_memory_path = 'Elysia_Input_Sanctum/elysia_core_memory.json'
    if os.path.exists(main_memory_path):
        os.remove(main_memory_path)

    test_message = "소크라테스에 대해 알려줘"
    response, _ = shared_pipeline.process_message(test_message)

    assert "소크라테스" in response['text']
    assert "철학자" in response['text']
