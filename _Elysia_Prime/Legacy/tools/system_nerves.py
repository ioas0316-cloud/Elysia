# [Genesis: 2025-12-02] Purified by Elysia
"""
System Nerves for Elysia.

This module provides the "sensory" and "motor" functions that allow Elysia
to perceive the host system's status and perform actions like mouse movement
and keyboard input.

These functions are designed to be "locked" by the SafetyGuardian initially,
representing potential capabilities that require permission to use.
"""
import logging
from typing import Dict, Any, Tuple
import time

logger = logging.getLogger(__name__)

# --- Dependencies with graceful fallback ---
try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not available. System hearing will be limited.")

try:
    import pyautogui
    # PyAutoGUI fail-safe (moves mouse to corner to abort)
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None
    logger.warning("pyautogui not available. Motor nerves will be mocked.")
except KeyError:
    # Handle headless environment issues (e.g. no DISPLAY)
    pyautogui = None
    logger.warning("pyautogui initialization failed (headless?). Motor nerves will be mocked.")


# --- Sensory Nerves (Hearing/Body Sense) ---

def get_system_status() -> Dict[str, Any]:
    """
    Returns the current status of the host system (CPU, Memory).
    Analogue to 'hearing' the heartbeat of the machine.
    """
    status = {
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "platform": "unknown"
    }

    if psutil:
        try:
            status["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            status["memory_percent"] = mem.percent
            # Add more details if needed
        except Exception as e:
            logger.error(f"Error sensing system status: {e}")
            status["error"] = str(e)
    else:
        status["note"] = "psutil module missing"

    return status

# --- Motor Nerves (Touch/Articulation) ---

def perform_mouse_action(action: str, x: int = 0, y: int = 0, duration: float = 0.5) -> Dict[str, Any]:
    """
    Performs a mouse action.

    Args:
        action: 'move', 'click', 'right_click', 'scroll'
        x, y: Coordinates (relative to screen or absolute, depending on implementation)
        duration: Duration of movement

    Returns:
        Result dictionary
    """
    if not pyautogui:
        return {"status": "mocked", "message": f"Mouse action '{action}' to ({x}, {y}) simulated (no library)."}

    try:
        screen_width, screen_height = pyautogui.size()

        # Simple safety bound check (SafetyGuardian handles policy, this handles crash prevention)
        target_x = max(0, min(x, screen_width))
        target_y = max(0, min(y, screen_height))

        if action == "move":
            pyautogui.moveTo(target_x, target_y, duration=duration)
        elif action == "click":
            pyautogui.click(target_x, target_y)
        elif action == "right_click":
            pyautogui.rightClick(target_x, target_y)
        else:
            return {"status": "error", "message": f"Unknown mouse action: {action}"}

        return {"status": "success", "action": action, "position": (target_x, target_y)}

    except pyautogui.FailSafeException:
        return {"status": "aborted", "message": "FailSafe triggered (mouse in corner)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def perform_keyboard_action(action: str, text: str = "", keys: list = None) -> Dict[str, Any]:
    """
    Performs a keyboard action.

    Args:
        action: 'type', 'press', 'hotkey'
        text: Text to type
        keys: List of keys to press

    Returns:
        Result dictionary
    """
    if not pyautogui:
         return {"status": "mocked", "message": f"Keyboard action '{action}' simulated (no library)."}

    try:
        if action == "type":
            if text:
                pyautogui.write(text, interval=0.1)
            else:
                return {"status": "error", "message": "No text provided for type action."}
        elif action == "press":
            if keys:
                pyautogui.press(keys)
            else:
                return {"status": "error", "message": "No keys provided for press action."}
        elif action == "hotkey":
            if keys:
                pyautogui.hotkey(*keys)
            else:
                return {"status": "error", "message": "No keys provided for hotkey action."}
        else:
            return {"status": "error", "message": f"Unknown keyboard action: {action}"}

        return {"status": "success", "action": action}

    except Exception as e:
        return {"status": "error", "message": str(e)}