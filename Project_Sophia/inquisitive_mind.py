"""
Inquisitive Mind for Elysia.

This module embodies the 'Wise Student' model. When Elysia's internal knowledge
is insufficient to answer a question, the InquisitiveMind is triggered to
query an external LLM for information. This information is then presented to the
user for verification before being integrated into Elysia's knowledge graph.
"""
import time
import logging
import os

from .gemini_api import generate_text

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'inquisitive_mind_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
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
                    return f"'{topic}'에 대해 찾아보니, '{external_knowledge.strip()}' 라고 하네요. 이 정보가 맞나요? 제 지식에 추가할까요?"
                else:
                    # If LLM returns empty string, treat as failure
                    raise ValueError("External LLM returned an empty response.")
            except Exception as e:
                inquisitive_logger.error(f"Attempt {attempt + 1} failed for topic '{topic}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    inquisitive_logger.error(f"All {max_retries} attempts failed for topic '{topic}'.")
                    return f"'{topic}'에 대해 알아보려 했지만, 외부 지식을 가져오는 데 실패했어요."
        return f"'{topic}'에 대해 알아보려 했지만, 외부 지식을 가져오는 데 실패했어요."
