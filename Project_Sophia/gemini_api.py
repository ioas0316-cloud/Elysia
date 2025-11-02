
import os
import google.generativeai as genai
import logging
from google.api_core import exceptions as api_core_exceptions

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'gemini_api_errors.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path)
    ]
)
gemini_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

_is_configured = False

def _configure_genai_if_needed():
    """Checks if the genai library is configured and configures it if not."""
    global _is_configured
    if _is_configured:
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    _is_configured = True


def get_text_embedding(text: str, model="models/text-embedding-004"):
    """Generates an embedding for the given text using the specified Gemini model."""
    _configure_genai_if_needed()
    try:
        return genai.embed_content(model=model, content=text)["embedding"]
    except (api_core_exceptions.InternalServerError,
            api_core_exceptions.ServiceUnavailable,
            api_core_exceptions.DeadlineExceeded) as e:
        gemini_logger.exception(f"A specific API error occurred during embedding: {e}")
        return None
    except Exception as e:
        gemini_logger.exception(f"An unexpected error occurred during embedding: {e}")
        return None

def generate_text(prompt: str, model_name="gemini-1.5-pro"):
    """Generates text using the specified Gemini model."""
    _configure_genai_if_needed()
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        return response.text
    except (api_core_exceptions.InternalServerError,
            api_core_exceptions.ServiceUnavailable,
            api_core_exceptions.DeadlineExceeded) as e:
        gemini_logger.exception(f"A specific API error occurred during text generation: {e}")
        return None
    except Exception as e:
        gemini_logger.exception(f"An unexpected error occurred during text generation: {e}")
        return None
