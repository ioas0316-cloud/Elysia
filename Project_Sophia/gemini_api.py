
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

    def generate_image_from_text(self, text: str, output_path: str):
        """
        Generates an image with the given text.
        This is a placeholder and does not actually use the Gemini API.
        """
        try:
            width, height = 800, 600
            img = Image.new('RGB', (width, height), color = 'white')
            d = ImageDraw.Draw(img)

            # Use a truetype font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            # Add text to image
            d.text((10,10), text, fill=(0,0,0), font=font)

            img.save(output_path)
            print(f"Placeholder image saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating placeholder image: {e}")
            return None

# For backward compatibility, we can instantiate the class
gemini_api = GeminiAPI()

def get_text_embedding(text: str, model="models/text-embedding-004"):
    return gemini_api.get_text_embedding(text, model)

def generate_text(prompt: str, model_name="models/gemini-2.5-pro"):
    return gemini_api.generate_text(prompt, model_name)

def generate_image_from_text(text: str, output_path: str):
    return gemini_api.generate_image_from_text(text, output_path)
