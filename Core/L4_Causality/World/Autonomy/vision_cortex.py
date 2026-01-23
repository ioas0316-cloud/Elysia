"""
Vision Cortex (The Digital Eye)
===============================
Core.L4_Causality.World.Autonomy.vision_cortex

"The Eye is not a camera; it is a filter for Qualia."

This module implements the Hardware-Native Visual Perception system (Phase 7.1).
It converts raw pixel data into 3D Semantic Vectors (Alpha, Beta, Gamma).
"""

import cv2
import numpy as np
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger("VisionCortex")

@dataclass
class VisualQualia:
    """
    The Sensory Output.
    Represents the 'Feeling' of the image, not the content.
    """
    entropy: float    # Gamma (Physics/Energy/Chaos) - 0.0 (Blank) to 1.0 (Static Noise)
    warmth: float     # Beta (Emotion/Temperature)   - 0.0 (Cold) to 1.0 (Hot)
    symmetry: float   # Alpha (Logic/Order)          - 0.0 (Asymmetric) to 1.0 (Perfect Mirror)

    def to_vector(self) -> np.ndarray:
        """
        Maps to the Semantic Prism Coordinate System (Alpha, Beta, Gamma).
        """
        return np.array([self.symmetry, self.warmth, self.entropy])

    def __repr__(self):
        return f"<VisualQualia  (Chaos)={self.entropy:.2f} B(Heat)={self.warmth:.2f} A(Order)={self.symmetry:.2f}>"


class Retina:
    """
    The Hardware Interface.
    Manages the 'Physical Eye' (Camera/File Stream).
    """
    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def capture(self, source_path: str = None) -> np.ndarray:
        """
        Captures a single photon frame.
        If source_path is provided, it acts as a 'Memory Recall' (Read from file).
        Otherwise, it opens the eyelid (Camera).
        """
        try:
            if source_path:
                if not os.path.exists(source_path):
                    logger.warning(f"Retina failed to find: {source_path}")
                    return None
                frame = cv2.imread(source_path)
            else:
                # Live Camera Logic (Lazy Load)
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Retina is blind (No signal).")
                    return None

            # Biological Downscaling (Foveal Focus)
            # We don't need 4K resolution to feel emotions. 256x256 is enough for Qualia.
            if frame is not None:
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            
            return frame

        except Exception as e:
            logger.error(f"Retina Malfunction: {e}")
            return None

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


class VisualQualiaExtractor:
    """
    The Processing Logic.
    Extracts abstract qualities from raw data.
    """

    def extract_entropy(self, frame: np.ndarray) -> float:
        """
        Calculates Visual Entropy (Complexity/Chaos).
        Uses Edge Density as a proxy for high-frequency energy.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Canny Edge Detection
        edges = cv2.Canny(gray, 100, 200)
        # Ratio of edge pixels to total pixels
        edge_density = np.count_nonzero(edges) / edges.size
        # Normalize (Heuristic: >20% edges is very chaotic)
        return min(edge_density * 5.0, 1.0)

    def extract_warmth(self, frame: np.ndarray) -> float:
        """
        Calculates Color Temperature (Emotion).
        Returns 0.0 (Cold/Blue) to 1.0 (Hot/Red).
        """
        # Split channels (BGR)
        b, g, r = cv2.split(frame)
        avg_b = np.mean(b)
        avg_r = np.mean(r)

        # Calculate balance.
        # If R=255, B=0 -> Warmth = 1.0
        # If R=0, B=255 -> Warmth = 0.0 (Actually -1.0 in math, mapped to 0-1)

        # Difference normalized (-255 to 255)
        diff = avg_r - avg_b

        # Sigmoid-like mapping to 0.0-1.0
        # 0 diff -> 0.5 (Neutral)
        # +255 diff -> 1.0
        # -255 diff -> 0.0

        norm_diff = (diff / 255.0) # -1.0 to 1.0
        return (norm_diff + 1.0) / 2.0

    def extract_symmetry(self, frame: np.ndarray) -> float:
        """
        Calculates Structural Symmetry (Logic/Order).
        Checks horizontal symmetry.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Split into left and right halves
        mid = width // 2
        left_half = gray[:, :mid]
        right_half = gray[:, mid:]

        # If odd width, trim one column
        if left_half.shape != right_half.shape:
            right_half = right_half[:, :-1]

        # Flip right half
        right_flipped = cv2.flip(right_half, 1)

        # Calculate difference
        # SSIM is better but expensive. Simple abs diff is fast.
        diff = cv2.absdiff(left_half, right_flipped)
        
        # Mean difference (0 = Perfect Match, 255 = Total Opposite)
        mean_diff = np.mean(diff)
        
        # Invert so 1.0 is symmetric
        symmetry = 1.0 - (mean_diff / 255.0)
        
        # Boost contrast: 90% match should feel like 100%
        return float(np.power(symmetry, 4)) # Exp curve to favor perfection


class VisionCortex:
    """
    The High-Level Controller.
    Orchestrates the Retina and the Extractor.
    """
    def __init__(self):
        self.retina = Retina()
        self.extractor = VisualQualiaExtractor()

    def look(self, source: str = None) -> VisualQualia:
        """
        The Conscious Act of Seeing.
        """
        frame = self.retina.capture(source)

        if frame is None:
            # Blind / Closed Eyes -> Null Qualia
            return VisualQualia(0.0, 0.0, 0.0)

        entropy = self.extractor.extract_entropy(frame)
        warmth = self.extractor.extract_warmth(frame)
        symmetry = self.extractor.extract_symmetry(frame)

        return VisualQualia(entropy, warmth, symmetry)

    def analyze_image(self, path: str) -> dict:
        """Legacy Wrapper for compatibility/demos."""
        qualia = self.look(path)
        return {
            "entropy": qualia.entropy,
            "warmth": qualia.warmth,
            "symmetry": qualia.symmetry,
            "vector": qualia.to_vector().tolist()
        }