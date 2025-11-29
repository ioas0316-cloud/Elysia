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
    logger.info("ℹ️ Pytesseract not found. Will use Resonance Vision instead.")

class VisualCortex:
    def __init__(self):
        self.enabled = PYAUTOGUI_AVAILABLE
        self._cleanup_old_screenshots()
    
    def _cleanup_old_screenshots(self, keep_last: int = 10):
        """오래된 스크린샷 삭제 (최근 N개만 유지)"""
        try:
            temp_dir = os.path.join(os.path.dirname(__file__), "../../temp_vision")
            if not os.path.exists(temp_dir):
                return
            
            files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.png')]
            files.sort(key=os.path.getmtime, reverse=True)
            
            # 오래된 파일 삭제
            for old_file in files[keep_last:]:
                try:
                    os.remove(old_file)
                    logger.debug(f"Cleaned up: {os.path.basename(old_file)}")
                except:
                    pass
        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")
        
    def read_screen_text(self, region: Tuple[int, int, int, int] = None) -> str:
        """
        Read text from the screen (OCR).
        
        Strategy:
        1. If Pytesseract available → Use OCR (most accurate)
        2. If not → Return None (caller should use Resonance Vision)
        
        Args:
            region: Optional (left, top, width, height) to read from.
        
        Returns:
            Text string if OCR available, None otherwise
        """
        if not PYTESSERACT_AVAILABLE:
            # No error - just return None to indicate fallback needed
            return None
            
        if not self.enabled:
            return None
            
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
            logger.warning(f"OCR failed, will use Resonance Vision: {e}")
            return None

    def capture_screen(self, filename: str = None, temp: bool = True) -> Optional[str]:
        """
        Capture the current screen content.
        If filename is provided, saves to that path.
        If temp=True, saves to temp folder (default).
        Returns the path to the saved file, or None if failed.
        """
        if not self.enabled:
            logger.info("[SIMULATION] Capturing screen...")
            return None

        try:
            if not filename:
                if temp:
                    # Use temp folder to avoid cluttering desktop!
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    filename = "elysia_vision_temp.png"  # Reuse same file
                    filepath = os.path.join(temp_dir, filename)
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"vision_{timestamp}.png"
                    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                    filepath = os.path.join(desktop, filename)
            else:
                filepath = filename
            
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            logger.debug(f"👁️ Vision captured: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Retina Failure: {e}")
            return None

    def _cleanup_old_screenshots(self, keep_last: int = 10):
        """오래된 스크린샷 삭제 (최근 N개만 유지)"""
        try:
            temp_dir = os.path.join(os.path.dirname(__file__), "../../temp_vision")
            if not os.path.exists(temp_dir):
                return
            
            files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.png')]
            files.sort(key=os.path.getmtime, reverse=True)
            
            # 오래된 파일 삭제
            for old_file in files[keep_last:]:
                try:
                    os.remove(old_file)
                    logger.debug(f"Cleaned up old screenshot: {old_file}")
                except:
                    pass
        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")
    
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
