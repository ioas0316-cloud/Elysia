class MockLocalLLMCortex:
    """
    A mock version of the LocalLLMCortex for testing purposes.
    It simulates the behavior of the real cortex but returns predefined,
    canned responses without loading the actual language model.
    This ensures that tests are fast, deterministic, and isolated from the
    performance and variability of a real LLM.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the mock object. The arguments are ignored to prevent
        errors when the test environment instantiates it.
        It includes a 'model' attribute to satisfy checks in the CognitionPipeline.
        """
        self.model = True  # To pass the `if self.local_llm_cortex.model:` check
        print("[MockLocalLLMCortex] Initialized.")


    def generate_response(self, prompt: str, max_tokens=150):
        """
        Returns a canned response based on the prompt content.
        This allows tests to verify application logic without the overhead
        of a real LLM.
        """
        print(f"[MockLocalLLMCortex] Generating mock response for prompt: '{prompt[:50]}...'")

        # --- Canned Responses for Specific Tests ---

        # For test_logical_reasoner.py -> test_reasoning_and_response
        if "소크라테스" in prompt:
            return "소크라테스는 고대 그리스의 철학자이며, 서양 철학의 기초를 다진 인물 중 한 명으로 평가받습니다. 그는 '너 자신을 알라'는 명언으로 유명합니다."

        # For test_pipeline_features.py -> test_conversational_memory_is_retrieved
        if "black holes" in prompt.lower():
            return "기억하고 있습니다. 우리는 이전에 블랙홀에 대해 이야기했습니다."

        # For test_pipeline_features.py -> test_inquisitive_mind_is_triggered
        if "supermassive black hole" in prompt.lower():
            return "이것은 모의 응답입니다."

        # For test_arithmetic_cortex.py -> test_pipeline_integration_question
        if "5 * 3" in prompt:
            return "5에 3을 곱하면 결과는 15입니다."

        # For test_arithmetic_cortex.py -> test_pipeline_integration_command
        if "100 / 4" in prompt:
            return "100을 4로 나누면 25입니다."

        # Default response for any other prompt
        return "이것은 기본 모의 응답입니다."
