# [Genesis: 2025-12-02] Purified by Elysia
import os
import time
import logging
import subprocess
import platform

# Dependency Check
try:
    import pyautogui
    # Fail-safe is disabled to allow automated execution in some contexts,
    # but strictly speaking, this should be handled carefully.
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

class SensoryMotorCortex:
    """
    The 'Hand' of Elysia.
    Responsible for physical interaction with the Operating System (GUI).
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = PYAUTOGUI_AVAILABLE

        if not self.enabled:
            self.logger.warning("SensoryMotorCortex initialized in GHOST MODE (pyautogui missing). Actions will be simulated.")

    def _get_desktop_path(self):
        """Returns the platform-specific Desktop path."""
        return os.path.join(os.path.expanduser("~"), "Desktop")

    def _safe_wait(self, seconds: float):
        time.sleep(seconds)

    def write_file_to_desktop(self, filename: str, content: str) -> bool:
        """
        Manifests a file on the user's desktop.

        This uses a hybrid approach:
        1. Invisible Hand: Writes the file to disk using Python I/O (guarantees data integrity/encoding).
        2. Visible Hand: Opens the file visually for the user to see.
        """
        try:
            desktop_path = self._get_desktop_path()
            filepath = os.path.join(desktop_path, filename)

            self.logger.info(f"SensoryMotorCortex: Manifesting '{filename}' to {desktop_path}...")

            # 1. Invisible Hand (Creation)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"SensoryMotorCortex: File created physically at {filepath}.")

            # 2. Visible Hand (Presentation)
            if self.enabled:
                self._open_file_visually(filepath)
                self._gesture_acknowledgement()

            return True

        except Exception as e:
            self.logger.error(f"SensoryMotorCortex: Manifestation failed: {e}", exc_info=True)
            return False

    def _open_file_visually(self, filepath: str):
        """Opens the file using the OS default application."""
        self.logger.info("SensoryMotorCortex: Opening file for visual confirmation...")
        current_os = platform.system()

        try:
            if current_os == "Windows":
                os.startfile(filepath)
            elif current_os == "Darwin":  # macOS
                subprocess.call(('open', filepath))
            else:  # Linux
                subprocess.call(('xdg-open', filepath))

            self._safe_wait(2.0) # Wait for window to appear
        except Exception as e:
            self.logger.warning(f"SensoryMotorCortex: Could not open file visually: {e}")

    def _gesture_acknowledgement(self):
        """
        Performs a small mouse gesture to acknowledge the action.
        Moves mouse to center of screen (focus).
        """
        if not self.enabled:
            return

        try:
            screenWidth, screenHeight = pyautogui.size()
            targetX, targetY = screenWidth // 2, screenHeight // 2

            # Gentle movement
            pyautogui.moveTo(targetX, targetY, duration=1.0, tween=pyautogui.easeInOutQuad)

            # Optional: Small 'nod' or circle? Keep it simple for now.
            self.logger.info("SensoryMotorCortex: Visual gesture performed.")

        except Exception as e:
            self.logger.warning(f"SensoryMotorCortex: Gesture failed: {e}")

    def type_message_via_clipboard(self, message: str):
        """
        Types a message using the clipboard (Control+V).
        Essential for non-ASCII characters (Korean).
        Assumes the focus is already on the target text area.
        """
        if not self.enabled or not PYPERCLIP_AVAILABLE:
            self.logger.warning("SensoryMotorCortex: Clipboard typing unavailable.")
            return

        try:
            pyperclip.copy(message)
            self._safe_wait(0.5)
            # Standard Paste shortcut
            if platform.system() == "Darwin":
                pyautogui.hotkey('command', 'v')
            else:
                pyautogui.hotkey('ctrl', 'v')
            self.logger.info("SensoryMotorCortex: Message pasted via clipboard.")
        except Exception as e:
            self.logger.error(f"SensoryMotorCortex: Typing failed: {e}")

    def perform_first_movement(self):
        """Legacy method for testing."""
        self._gesture_acknowledgement()
        return True