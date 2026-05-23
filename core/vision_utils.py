"""
Elysia Core Vision Utility Library
==================================
Handles screen capture and vision analysis (Optical Flow and red-color pain detection).
"""

import cv2
import mss
import numpy as np

class ScreenSenser:
    def __init__(self, capture_width=800, capture_height=600):
        self.sct = mss.mss()
        self.width = capture_width
        self.height = capture_height
        self.rect = None
        self.prev_frame_gray = None

    def get_capture_rect(self):
        """Calculate and cache the centered screen capture region."""
        if self.rect is None:
            monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            w = min(self.width, monitor["width"])
            h = min(self.height, monitor["height"])
            left = monitor["left"] + (monitor["width"] - w) // 2
            top = monitor["top"] + (monitor["height"] - h) // 2
            self.rect = {"top": top, "left": left, "width": w, "height": h}
        return self.rect

    def grab_screen(self) -> np.ndarray:
        rect = self.get_capture_rect()
        return np.array(self.sct.grab(rect))

    def process_vision_chaos(self, frame: np.ndarray) -> tuple[float, bool]:
        """
        Processes a BGRA screen frame.
        Returns:
            - entropy (float): [0.0 - 1.0] representing motion/optical flow.
            - damage_detected (bool): True if a significant red flash is detected.
        """
        # Convert to grayscale and blur to remove sensor noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        entropy = 0.0
        damage_detected = False
        total_pixels = gray.size

        if self.prev_frame_gray is not None:
            # Calculate pixel differences (motion estimation)
            frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            changed_pixels = np.count_nonzero(thresh)
            # Normalization factor of 5.0 to increase sensitivity
            entropy = min(1.0, (changed_pixels / total_pixels) * 5.0)

        # Red channel check for pain/death flash (screen flash detection)
        # R > 150 and R > B + 50 and R > G + 50
        frame_int16 = frame.astype(np.int16)
        r = frame_int16[:, :, 2]
        g = frame_int16[:, :, 1]
        b = frame_int16[:, :, 0]
        
        red_mask = (r > 150) & (r > b + 50) & (r > g + 50)
        red_ratio = np.count_nonzero(red_mask) / total_pixels

        if red_ratio > 0.3:
            damage_detected = True

        self.prev_frame_gray = gray
        return entropy, damage_detected
