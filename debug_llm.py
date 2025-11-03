# debug_llm.py
import os
import sys

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

from Project_Sophia.local_llm_cortex import LocalLLMCortex

def run_test():
    """
    Initializes the LocalLLMCortex and tests its response generation
    with a simple Korean prompt.
    """
    print("--- Initializing LocalLLMCortex ---")
    try:
        llm_cortex = LocalLLMCortex()
        if not llm_cortex.model:
            print("--- LLM Cortex failed to initialize. Aborting. ---")
            return

        print("\n--- Testing Korean Prompt ---")
        prompt = "안녕하세요! 오늘 기분이 어떠신가요?"
        print(f"Sending prompt: '{prompt}'")

        response = llm_cortex.generate_response(prompt)

        print(f"\nReceived response:")
        print("--------------------")
        print(response)
        print("--------------------")

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)

if __name__ == "__main__":
    run_test()
