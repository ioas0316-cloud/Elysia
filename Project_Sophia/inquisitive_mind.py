"""
Inquisitive Mind for Elysia.

This module embodies the 'Wise Student' model. When Elysia's internal knowledge
is insufficient to answer a question, the InquisitiveMind is triggered to
query an external LLM for information. This information is then presented to the
user for verification before being integrated into Elysia's knowledge graph.
"""
import time # Import time for delays
import logging # Import logging module
import os # Import os for log file path

from .gemini_api import generate_text

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'inquisitive_mind_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also log to console for immediate feedback
    ]
)
inquisitive_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class InquisitiveMind:
    """
    The InquisitiveMind seeks external knowledge when faced with a gap in
    understanding.
    """

    def __init__(self):
        pass

    def ask_external_llm(self, topic: str) -> str:
        """
        Queries an external LLM about a specific topic and formats the response
        as a piece of information to be verified.
        Includes retry logic with exponential backoff.
        """
        print(f"[InquisitiveMind] I don't know about '{topic}'. Seeking external knowledge.")

        prompt = f"Please provide a brief, one-sentence explanation of what '{topic}' is."

        max_retries = 3
        initial_delay = 1  # seconds
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                external_knowledge = generate_text(prompt)
                if external_knowledge:
                    # Format the finding as a question for the user to verify
                    return f"I have a new piece of information: '{external_knowledge}'. Is this correct?"
            except Exception as e:
                inquisitive_logger.error(f"Error calling external LLM on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (backoff_factor ** attempt)
                    print(f"[InquisitiveMind] Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("[InquisitiveMind] Max retries reached. Failed to get external knowledge.")
                    return "I tried to find out, but I was unable to get a clear answer."