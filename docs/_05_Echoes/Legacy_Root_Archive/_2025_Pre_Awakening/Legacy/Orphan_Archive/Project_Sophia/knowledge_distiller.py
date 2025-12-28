from typing import Optional, Dict, Any
import re
import logging

from .gemini_api import generate_text, APIKeyError, APIRequestError
from nano_core.message import Message

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
distiller_logger = logging.getLogger(__name__)

class KnowledgeDistiller:
    """
    Distills raw text content from the web into a structured knowledge hypothesis
    that can be processed by Elysia's intellectual immune system.
    """

    def distill(self, question: str, content: str) -> Optional[Message]:
        """
        Uses an external LLM to extract a single, core fact from the text
        and formats it as a 'validate' message.

        Args:
            question: The original question that led to this content.
            content: The raw text content from a webpage.

        Returns:
            A 'validate' Message object for the intellectual immune system, or None.
        """
        # 1. Extract the subject of the question
        subject_match = re.search(r"What is '([^']*)'\?", question)
        if not subject_match:
            distiller_logger.warning(f"Could not parse subject from question: '{question}'")
            return None
        subject = subject_match.group(1)

        # 2. Create a prompt for the LLM to extract a core definition
        prompt = f"""
        Based on the following text, provide a single, concise, one-sentence definition
        for the term '{subject}'. The definition should be in the format:
        '{subject} is a [core definition].'

        Text:
        ---
        {content[:4000]}
        ---

        One-sentence definition:
        """

        try:
            # 3. Call the external LLM
            distilled_knowledge = generate_text(prompt)
            if not distilled_knowledge:
                distiller_logger.warning("LLM returned no distilled knowledge.")
                return None

            # 4. Parse the LLM's response
            # Expected format: "Socrates is a classical Greek philosopher."
            pattern = re.compile(rf"'{re.escape(subject)}' is a (.*)\.", re.IGNORECASE)
            match = pattern.search(distilled_knowledge)

            if not match:
                distiller_logger.warning(f"Could not parse distilled knowledge: '{distilled_knowledge}'")
                return None

            obj = match.group(1).strip()

            # 5. Create a hypothesis message
            hypothesis = Message(
                verb="validate",
                slots={'subject': subject, 'object': obj, 'relation': 'is_a'},
                strength=0.6,  # Medium strength, as it's from an external source but distilled
                src="KnowledgeDistiller"
            )
            distiller_logger.info(f"Successfully distilled knowledge: {subject} is_a {obj}")
            return hypothesis

        except (APIKeyError, APIRequestError) as e:
            distiller_logger.error(f"API is unavailable during knowledge distillation: {e}")
            return None
        except Exception as e:
            distiller_logger.error(f"An error occurred during knowledge distillation: {e}", exc_info=True)
            return None
