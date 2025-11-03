import pytest
from Project_Sophia.cognition_pipeline import CognitionPipeline
from tests.mocks.mock_local_llm_cortex import MockLocalLLMCortex

@pytest.fixture(scope="session")
def shared_pipeline():
    """
    Provides a shared, mock-injected instance of the CognitionPipeline for the entire test session.

    This fixture acts as a "mock assembly factory":
    1. It creates a standard CognitionPipeline instance.
    2. It replaces the real LocalLLMCortex and its dependent, InquisitiveMind,
       with a mock version (MockLocalLLMCortex).
    This ensures all tests run against a controlled, predictable environment
    without making real API calls or loading heavy models.
    """
    pipeline = CognitionPipeline()

    # Assemble the mock components
    mock_llm_cortex = MockLocalLLMCortex()

    # Replace the real components with mock components
    pipeline.local_llm_cortex = mock_llm_cortex
    pipeline.inquisitive_mind.llm_cortex = mock_llm_cortex

    return pipeline
