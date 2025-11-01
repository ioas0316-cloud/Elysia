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
        delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                external_knowledge = generate_text(prompt)
                if external_knowledge:
                    # Format the finding as a question for the user to verify
                    return f"'{topic}'에 대한 흥미로운 사실을 발견했습니다: {external_knowledge} 이 정보가 정확하다면, '네'라고 답해주시겠어요?"
            except Exception as e:
                inquisitive_logger.error(f"Error querying external LLM for '{topic}' on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    return f"'{topic}'에 대한 정보를 찾는 데 어려움을 겪고 있습니다. 잠시 후에 다시 시도해 주시겠어요?"

        return f"'{topic}'에 대한 외부 정보를 조회하는 데 실패했습니다."