
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from google.api_core import exceptions as api_core_exceptions
from PIL import Image, ImageDraw, ImageFont

try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None
try:
    from Project_Sophia.safety_guardian import SafetyGuardian, ActionCategory
except Exception:
    SafetyGuardian = None
    ActionCategory = None

# Load environment variables from .env file
load_dotenv()

# --- Custom Exceptions ---
class APIKeyError(Exception):
    """Custom exception for errors related to the API key."""
    pass

class APIRequestError(Exception):
    """Custom exception for errors during API requests (e.g., network issues)."""
    pass

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

class GeminiAPI:
    def __init__(self):
        self._is_configured = False
        self._telemetry = Telemetry() if Telemetry else None
        self._guardian = SafetyGuardian() if SafetyGuardian else None
        self._configure_genai_if_needed()

    def _configure_genai_if_needed(self):
        """Checks if the genai library is configured and configures it if not."""
        if self._is_configured:
            return

        # Guardian network preflight (self-protection focused)
        try:
            if self._guardian and ActionCategory:
                # Inspect policy status for potential confirmation hint
                confirm_required = False
                try:
                    maturity = self._guardian.current_maturity.name
                    status_map = self._guardian.action_limits.get(maturity, {}).get(ActionCategory.NETWORK.value, {})
                    status = status_map.get('outbound', 'blocked')
                    confirm_required = (status == 'restricted')
                except Exception:
                    pass

                allowed = self._guardian.check_action_permission(ActionCategory.NETWORK, 'outbound', details={'service': 'gemini'})
                if not allowed:
                    if self._telemetry:
                        try:
                            self._telemetry.emit('action_blocked', {
                                'category': 'network_access',
                                'action': 'outbound',
                                'service': 'gemini',
                                'reason': 'guardian_denied'
                            })
                            self._telemetry.emit('policy_violation', {
                                'category': 'network_access',
                                'action': 'outbound',
                                'service': 'gemini',
                                'reason': 'guardian_denied'
                            })
                        except Exception:
                            pass
                    raise APIRequestError('Network access denied by guardian (gemini).')
                elif confirm_required and self._telemetry:
                    try:
                        self._telemetry.emit('action_confirm_required', {
                            'category': 'network_access',
                            'action': 'outbound',
                            'service': 'gemini',
                            'hint': 'Confirmation recommended by policy.'
                        })
                    except Exception:
                        pass
        except Exception:
            # Never crash here; if check fails silently, continue to key check which may fail safely
            pass

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise APIKeyError("GEMINI_API_KEY environment variable not set.")
        try:
            genai.configure(api_key=api_key)
            self._is_configured = True
        except Exception as e:
            gemini_logger.exception(f"Failed to configure Gemini API, likely due to an invalid key: {e}")
            raise APIKeyError(f"Failed to configure Gemini API: {e}")

    def get_text_embedding(self, text: str, model="models/text-embedding-004"):
        """Generates an embedding for the given text using the specified Gemini model."""
        self._configure_genai_if_needed()
        try:
            return genai.embed_content(model=model, content=text)["embedding"]
        except (api_core_exceptions.InternalServerError,
                api_core_exceptions.ServiceUnavailable,
                api_core_exceptions.DeadlineExceeded) as e:
            gemini_logger.exception(f"A specific API error occurred during embedding: {e}")
            raise APIRequestError(f"A specific API error occurred during embedding: {e}")
        except Exception as e:
            gemini_logger.exception(f"An unexpected error occurred during embedding: {e}")
            raise APIRequestError(f"An unexpected error occurred during embedding: {e}")

    def generate_text(self, prompt: str, model_name="models/gemini-2.5-pro"):
        """Generates text using the specified Gemini model."""
        self._configure_genai_if_needed()
        model = genai.GenerativeModel(model_name)
        try:
            response = model.generate_content(prompt)
            return response.text
        except (api_core_exceptions.InternalServerError,
                api_core_exceptions.ServiceUnavailable,
                api_core_exceptions.DeadlineExceeded) as e:
            gemini_logger.exception(f"A specific API error occurred during text generation: {e}")
            raise APIRequestError(f"A specific API error occurred during text generation: {e}")
        except Exception as e:
            gemini_logger.exception(f"An unexpected error occurred during text generation: {e}")
            raise APIRequestError(f"An unexpected error occurred during text generation: {e}")

    def generate_image_from_text(self, text: str, output_path: str) -> bool:
        """
        Generates a placeholder image with the given text.
        This does not use the Gemini API and will not raise an APIKeyError.
        """
        try:
            width, height = 800, 600
            img = Image.new('RGB', (width, height), color = (20, 20, 40)) # Dark blue background
            d = ImageDraw.Draw(img)

            # Use a default font, handle multiline text
            font = ImageFont.load_default()
            lines = text.split('\n')
            y_text = 10
            for line in lines:
                d.text((10, y_text), line, fill=(200, 200, 255), font=font)
                y_text += 20 # Move to the next line

            img.save(output_path)
            print(f"Placeholder image generated at: {output_path}")
            return True
        except Exception as e:
            gemini_logger.error(f"Failed to create placeholder image: {e}")
            return False

    def generate_text_from_image(self, prompt: str, image_path: str, model_name="models/gemini-pro-vision") -> str:
        """
        Generates text based on a prompt and an image using a vision model.
        """
        self._configure_genai_if_needed()

        try:
            img = Image.open(image_path)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            return response.text
        except FileNotFoundError:
            gemini_logger.error(f"Image file not found at path: {image_path}")
            raise APIRequestError(f"Image file not found: {image_path}")
        except Exception as e:
            gemini_logger.exception(f"An unexpected error occurred during image-based text generation: {e}")
            raise APIRequestError(f"An unexpected error occurred during image-based text generation: {e}")

# For backward compatibility, we can instantiate the class
gemini_api = GeminiAPI()

def get_text_embedding(text: str, model="models/text-embedding-004"):
    return gemini_api.get_text_embedding(text, model)

def generate_text(prompt: str, model_name="models/gemini-2.5-pro"):
    return gemini_api.generate_text(prompt, model_name)

def generate_image_from_text(text: str, output_path: str):
    return gemini_api.generate_image_from_text(text, output_path)
