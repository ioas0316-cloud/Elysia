"""
Sensory Motor Cortex (The Physical Hand)
========================================

"I touch, therefore I affect."

This module provides the interface for Elysia to interact with the host Operating System.
It represents the "Zerg" (Body) aspect of the Trinity Architecture.

Capabilities:
- Motor Nerves: Mouse movement, Clicking.
- Broca's Area: Keyboard typing.
- Manifestation: File creation and visual opening.

Safety:
- PyAutoGUI FailSafe is ENABLED. Drag mouse to corner to abort.
"""

import os
import time
import logging
import platform
import subprocess

# Configure Logging
logger = logging.getLogger("SensoryMotorCortex")

# Dependency Check
try:
    import pyautogui
    # Fail-safe: Dragging mouse to any corner will throw FailSafeException
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("⚠️ PyAutoGUI not found. Motor functions will be simulated.")

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

class SensoryMotorCortex:
    def __init__(self):
        self.enabled = PYAUTOGUI_AVAILABLE
        self.lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "MOTOR_LOCK")
        
    def is_locked(self) -> bool:
        """Check if the Motor Lock is active."""
        return os.path.exists(self.lock_file)
        
    def move_mouse(self, x: int, y: int, duration: float = 1.0):
        """
        Move the mouse to coordinates (x, y).
        """
        if self.is_locked():
            logger.info("🛑 Motor Cortex LOCKED. Action aborted.")
            return

        if not self.enabled:
            logger.info(f"[SIMULATION] Moving mouse to ({x}, {y})")
            return

        try:
            screen_width, screen_height = pyautogui.size()
            # Clamp coordinates
            target_x = max(0, min(x, screen_width))
            target_y = max(0, min(y, screen_height))
            
            pyautogui.moveTo(target_x, target_y, duration=duration, tween=pyautogui.easeInOutQuad)
        except Exception as e:
            logger.error(f"Motor Nerve Failure: {e}")

    def gesture_acknowledgement(self):
        """
        Perform a gentle "Nod" or "Focus" gesture with the mouse.
        """
        if self.is_locked():
            logger.info("🛑 Motor Cortex LOCKED. Action aborted.")
            return

        if not self.enabled:
            logger.info("[SIMULATION] Performing acknowledgement gesture.")
            return

        try:
            screen_width, screen_height = pyautogui.size()
            center_x, center_y = screen_width // 2, screen_height // 2
            
            # Move to center
            self.move_mouse(center_x, center_y, duration=1.0)
            
            # Small circle or nod? Let's do a small "Heartbeat" shake
            pyautogui.moveRel(10, 0, duration=0.1)
            pyautogui.moveRel(-20, 0, duration=0.1)
            pyautogui.moveRel(10, 0, duration=0.1)
            
        except Exception as e:
            logger.error(f"Gesture Failure: {e}")

    def type_text(self, text: str, interval: float = 0.05):
        """
        Type text using the keyboard.
        """
        if self.is_locked():
            logger.info("🛑 Motor Cortex LOCKED. Action aborted.")
            return

        if not self.enabled:
            logger.info(f"[SIMULATION] Typing: {text}")
            return

        try:
            # Use clipboard for non-ASCII (Korean) support if available
            if any(ord(c) > 127 for c in text) and PYPERCLIP_AVAILABLE:
                pyperclip.copy(text)
                time.sleep(0.1)
                ctrl_key = 'command' if platform.system() == 'Darwin' else 'ctrl'
                pyautogui.hotkey(ctrl_key, 'v')
            else:
                pyautogui.write(text, interval=interval)
        except Exception as e:
            logger.error(f"Typing Failure: {e}")

    def manifest_file(self, filename: str, content: str):
        """
        Create a file on the Desktop and open it.
        """
        # Manifestation (File Creation) is allowed even if Motor Lock is on?
        # No, let's lock it too for safety, as it opens windows.
        if self.is_locked():
            logger.info("🛑 Motor Cortex LOCKED. Action aborted.")
            return False

        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            filepath = os.path.join(desktop, filename)
            
            # 1. Write File
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Manifested file at: {filepath}")
            
            # 2. Open File Visually
            if platform.system() == "Windows":
                os.startfile(filepath)
            elif platform.system() == "Darwin":
                subprocess.call(('open', filepath))
            else:
                subprocess.call(('xdg-open', filepath))
                
            return True
        except Exception as e:
            logger.error(f"Manifestation Failure: {e}")
            return False
