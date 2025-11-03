import pytest
import os
from unittest.mock import patch
from Project_Sophia.core_memory import Memory, EmotionalState
from Project_Sophia.gemini_api import APIKeyError, APIRequestError

@pytest.fixture
def test_memory(shared_pipeline, tmp_path):
    """
    Provides an isolated memory environment for each test.
    """
    original_memory_path = shared_pipeline.core_memory.file_path
    test_memory_path = tmp_path / "test_elysia_core_memory.json"
    shared_pipeline.core_memory.file_path = test_memory_path

    if os.path.exists(test_memory_path):
        os.remove(test_memory_path)

    yield shared_pipeline.core_memory

    shared_pipeline.core_memory.file_path = original_memory_path

# --- Integration Tests ---

def test_conversational_memory_is_retrieved(shared_pipeline, test_memory):
    """
    Tests if the pipeline can retrieve a relevant past experience.
    """
    past_experience = Memory(
        timestamp="2025-01-01T12:00:00",
        content="I enjoy learning about black holes.",
        emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])
    )
    test_memory.add_experience(past_experience)
    response, _ = shared_pipeline.process_message("What do you know about black holes?")
    assert "기억하고 있습니다" in response['text']

def test_inquisitive_mind_is_triggered(shared_pipeline, test_memory):
    """
    Tests if the InquisitiveMind is triggered for unknown topics.
    """
    response, _ = shared_pipeline.process_message("What is a supermassive black hole?")
    assert "모의 응답" in response['text']

# We still need to patch the specific function in InquisitiveMind to simulate an API error
@patch('Project_Sophia.inquisitive_mind.InquisitiveMind.ask_external_llm', side_effect=APIKeyError("Test API Key Error"))
def test_fallback_mechanism_on_api_key_error(mock_ask_external, shared_pipeline, test_memory):
    """
    Tests the fallback mechanism on APIKeyError.
    """
    response, _ = shared_pipeline.process_message("Tell me about photosynthesis?")
    assert "photosynthesis" in response['text'].lower()
    assert "api 키에 문제가 발생하여" in response['text'].lower()

@patch('Project_Sophia.inquisitive_mind.InquisitiveMind.ask_external_llm', side_effect=APIRequestError("Test API Request Error"))
def test_fallback_mechanism_on_api_request_error(mock_ask_external, shared_pipeline, test_memory):
    """
    Tests the fallback mechanism on APIRequestError.
    """
    response, _ = shared_pipeline.process_message("What is the weather like today?")
    assert "weather" in response['text'].lower()
    assert "api 요청에 문제가 발생하여" in response['text'].lower()
