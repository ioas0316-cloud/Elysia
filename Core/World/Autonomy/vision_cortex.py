"""
Vision Cortex (The Digital Eye)
===============================
Provides sensory perception of Visual Data using OpenCV.
Extracts 'Qualia' (Color, Entropy, Motion) from images and video.
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("VisionCortex")

class VisionCortex:
    def __init__(self):
        self.last_frame = None
        
    def analyze_image(self, image_path: str) -> dict:
        """
        Reads an image and extracts sensory qualia.
        Returns: { 'brightness': float, 'entropy': float, 'warmth': float, 'dominant_color_hex': str }
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return {}
            
        try:
            # Load Image
            img = cv2.imread(image_path)
            if img is None:
                return {}
                
            # 1. Brightness (V in HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = hsv[:,:,2].mean() / 255.0
            
            # 2. Entropy (Complexity)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum() # Normalize
            # Entropy formula: -sum(p * log2(p))
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            # Normalize entropy (max for 256 bins is 8)
            complexity = min(entropy / 8.0, 1.0)
            
            # 3. Warmth (Red/Orange vs Blue)
            # Simple heuristic: Ratio of Red/Yellow to Blue
            b, g, r = cv2.split(img)
            avg_r = r.mean()
            avg_b = b.mean()
            # Warmth: -1.0 (Cold) to 1.0 (Warm)
            warmth = (avg_r - avg_b) / 255.0
            
            return {
                "brightness": float(brightness),
                "entropy": float(complexity),
                "warmth": float(warmth),
                "path": image_path
            }
            
        except Exception as e:
            logger.error(f"Vision Error: {e}")
            return {}

    def analyze_flow_frame(self, frame) -> dict:
        """
        Analyzes motion between previous frame and current frame.
        """
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {"motion": 0.0}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(self.last_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        avg_motion = np.mean(mag)
        self.last_frame = gray
        
        return {"motion": float(avg_motion)}
