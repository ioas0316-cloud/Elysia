"""
Visual Cortex (The Eyes)
========================

"I see, therefore I witness."

This module provides the interface for Elysia to perceive the visual output of the host system.
It represents the sensory input aspect of the Zerg (Body) pillar.

Capabilities:
- Retina: Screen capture (Screenshot).
- Pattern Recognition: Finding specific images/icons on the screen (Template Matching).

Dependencies:
- pyautogui (Required for capture)
- opencv-python (Recommended for matching)
- pillow (Required for image handling)
"""

import os
import time
import logging
import platform
from typing import Tuple, Optional

# Configure Logging
logger = logging.getLogger("VisualCortex")

# Dependency Check
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("⚠️ PyAutoGUI not found. Vision will be simulated.")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("⚠️ OpenCV not found. Advanced pattern matching disabled.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    
    # Check for local Tesseract folder (Portable Mode)
    local_tesseract = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Tesseract-OCR", "tesseract.exe")
    if os.path.exists(local_tesseract):
        pytesseract.pytesseract.tesseract_cmd = local_tesseract
        
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("⚠️ Pytesseract not found. Reading disabled.")

class VisualCortex:
    def __init__(self):
        self.enabled = PYAUTOGUI_AVAILABLE
        
    def read_screen_text(self, region: Tuple[int, int, int, int] = None) -> str:
        """
        Read text from the screen (OCR).
        Args:
            region: Optional (left, top, width, height) to read from.
        """
        if not PYTESSERACT_AVAILABLE:
            return "Error: OCR Engine (pytesseract) not installed."
            
        if not self.enabled:
            return "[SIMULATION] Reading screen text..."
            
        try:
            # Capture
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
                
            # Read
            text = pytesseract.image_to_string(screenshot)
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR Failure: {e}")
            return f"Error: {e}"

    def capture_screen(self, filename: str = None) -> Optional[str]:
        """
        Capture the current screen content.
        If filename is provided, saves to that path.
        Returns the path to the saved file, or None if failed.
        """
        if not self.enabled:
            logger.info("[SIMULATION] Capturing screen...")
            return None

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not filename:
                filename = f"vision_{timestamp}.png"
                
            # Default to Desktop for visibility
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            filepath = os.path.join(desktop, filename)
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            logger.info(f"👁️ Vision captured: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Retina Failure: {e}")
            return None

    def analyze_brightness(self, filepath: str) -> str:
        """
        Analyze the average brightness of the captured image.
        Returns a description (e.g., "Bright", "Dark").
        """
        if not OPENCV_AVAILABLE or not os.path.exists(filepath):
            return "Unknown (Analysis Unavailable)"
            
        try:
            img = cv2.imread(filepath)
            avg_color_per_row = np.average(img, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            brightness = np.mean(avg_color)
            
            if brightness > 200:
                return "Blindingly Bright"
            elif brightness > 128:
                return "Bright"
            elif brightness > 64:
                return "Dim"
            else:
                return "Dark"
                
        except Exception as e:
            logger.error(f"Analysis Failure: {e}")
            return "Error"

    def find_image(self, template_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Find the coordinates (x, y) of a template image on the screen.
        """
        if not self.enabled:
            return None
            
        if not os.path.exists(template_path):
            logger.warning(f"Template not found: {template_path}")
            return None
            
        try:
            # Use PyAutoGUI's built-in locate function (uses Pillow/OpenCV internally)
            location = pyautogui.locateCenterOnScreen(template_path, confidence=confidence)
            if location:
                logger.info(f"🎯 Pattern found at: {location}")
                return location
            else:
                logger.info(f"Pattern not found: {template_path}")
                return None
                
        except Exception as e:
            logger.error(f"Pattern Recognition Failure: {e}")
            return None
