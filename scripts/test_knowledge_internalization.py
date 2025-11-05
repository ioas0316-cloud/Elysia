import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline

def run_test_query(message: str):
    """
    Initializes the pipeline and processes a single message to test Elysia's response.
    """
    print(f"--- Testing internal voice with query: '{message}' ---")

    # Initialize the pipeline in offline mode
    pipeline = CognitionPipeline()
    pipeline.api_available = False # Ensure we are testing the internal voice

    # Process the message
    response, _ = pipeline.process_message(message)

    # Print the result
    print("Elysia's response:")
    print(response.get('text', 'No text response.'))
    print("--------------------------------------------------")

if __name__ == '__main__':
    # Test query to see if Elysia can explain a concept she just learned.
    run_test_query("What is a word?")
    run_test_query("What is a sentence?")
