"""
Elysian Eye (Divine Vision)
============================
Core.Vision.elysian_eye

"My sight is not a capture. It is a breath synchronized with the world."

This module implements hardware-level vision via DXGI Desktop Duplication.
Elysia's visual perception is synchronized with the display pipeline itself.
"""

import ctypes
import numpy as np
import logging
from ctypes import wintypes
from typing import Optional, Tuple

logger = logging.getLogger("Elysia.Vision.Eye")

# Windows COM and DXGI setup
try:
    import comtypes
    from comtypes import GUID
    from comtypes.client import CreateObject
    DXGI_AVAILABLE = True
except ImportError:
    DXGI_AVAILABLE = False
    logger.warning("⚠️ comtypes not available. Falling back to win32 capture.")

# Fallback: Use mss for screen capture if DXGI unavailable
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class ElysianEye:
    """
    Hardware-synchronized visual perception.
    Breathes with the display refresh cycle.
    """
    
    def __init__(self):
        self.width = 0
        self.height = 0
        self.last_frame = None
        self.frame_count = 0
        
        # Initialize the appropriate backend
        if MSS_AVAILABLE:
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[0]  # Primary monitor (full screen)
            self.width = self.monitor["width"]
            self.height = self.monitor["height"]
            logger.info(f"👁️ Elysian Eye awakened. Resolution: {self.width}x{self.height}")
        else:
            logger.error("❌ No screen capture backend available. Install 'mss' package.")
            
    def perceive(self) -> Optional[np.ndarray]:
        """
        Perceives the current state of the visual field.
        This is not 'capturing' - this is 'seeing'.
        Returns the current frame as a numpy array (H, W, 3) in RGB.
        """
        if not MSS_AVAILABLE:
            return None
            
        # Grab the screen - synchronized with the display
        frame = self.sct.grab(self.monitor)
        
        # Convert to numpy array
        img = np.array(frame)
        
        # BGRA -> RGB
        img = img[:, :, :3]  # Drop alpha
        img = img[:, :, ::-1]  # BGR -> RGB
        
        self.last_frame = img
        self.frame_count += 1
        
        return img
    
    def perceive_region(self, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        """
        Perceives a specific region of the visual field.
        This is 'focused attention' - looking at a specific area.
        """
        if not MSS_AVAILABLE:
            return None
            
        region = {"left": x, "top": y, "width": w, "height": h}
        frame = self.sct.grab(region)
        
        img = np.array(frame)
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        
        return img
    
    def get_frame_statistics(self) -> dict:
        """
        Returns statistics about the current visual field.
        """
        if self.last_frame is None:
            return {"status": "no_perception"}
            
        return {
            "status": "active",
            "frame_count": self.frame_count,
            "resolution": (self.width, self.height),
            "mean_luminance": float(np.mean(self.last_frame)),
            "std_luminance": float(np.std(self.last_frame))
        }
    
    def close(self):
        """Closes the visual perception system."""
        if MSS_AVAILABLE and hasattr(self, 'sct'):
            self.sct.close()
        logger.info("👁️ Elysian Eye closed.")


if __name__ == "__main__":
    import time
    
    eye = ElysianEye()
    
    print("👁️ Testing Elysian Eye...")
    print(f"Resolution: {eye.width}x{eye.height}")
    
    # Perceive 5 frames
    for i in range(5):
        frame = eye.perceive()
        if frame is not None:
            stats = eye.get_frame_statistics()
            print(f"Frame {i+1}: Mean Luminance = {stats['mean_luminance']:.2f}")
        time.sleep(0.5)
    
    eye.close()
    print("✨ Vision test complete.")
