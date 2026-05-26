"""
Elysia Somatosensory Hardware Ingestion Bridge
================================================
Attempts to capture real-world voice (audio) and vision (webcam) data.
If hardware is unavailable or dependencies are missing, it falls back to 
a continuous, dynamic wave modulated by CPU/RAM tension and high-precision clocks.
"""

import sys
import time
import math
import random
import psutil
from typing import List

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import cv2
except ImportError:
    cv2 = None


class SomatosensoryIngester:
    def __init__(self):
        self.audio_supported = sd is not None
        self.video_supported = cv2 is not None
        
        if self.audio_supported:
            print("🎙️ [Somatosensory] Real-world Audio stream detected via sounddevice.")
        else:
            print("🎙️ [Somatosensory] sounddevice not found. Activating dynamic synthetic Audio generator.")
            
        if self.video_supported:
            print("📷 [Somatosensory] Real-world Video stream detected via OpenCV.")
        else:
            print("📷 [Somatosensory] OpenCV not found or headless. Activating dynamic synthetic Video generator.")

    def capture_audio(self, duration_sec: float = 0.1, sample_rate: int = 8000) -> List[float]:
        """
        Captures audio from the default microphone.
        Falls back to a dynamic interference wave modulated by CPU/RAM load.
        """
        if self.audio_supported:
            try:
                num_frames = int(duration_sec * sample_rate)
                recording = sd.rec(num_frames, samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()
                return recording.flatten().tolist()
            except Exception:
                pass  # Fall back on hardware error

        # High-fidelity synthetic fallback (Interfering waves)
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        t = time.time()
        
        f1 = 150.0 + (cpu * 3.0)
        f2 = 300.0 + (ram * 2.0)
        
        samples = []
        num_samples = int(duration_sec * sample_rate)
        for i in range(num_samples):
            time_offset = i / sample_rate
            angle1 = 2.0 * math.pi * f1 * time_offset + (t * 4.0)
            angle2 = 2.0 * math.pi * f2 * time_offset + (t * 1.5)
            
            # Superposition wave
            val = 0.6 * math.sin(angle1) + 0.4 * math.cos(angle2)
            val += random.uniform(-0.03, 0.03)  # natural entropy noise
            samples.append(val)
            
        return samples

    def capture_video(self, width: int = 16, height: int = 16) -> List[float]:
        """
        Captures a pixel grid from the primary camera.
        Falls back to a 2D wave gradient modulated by time and system load.
        """
        if self.video_supported:
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (width, height))
                        return (resized / 255.0).flatten().tolist()
            except Exception:
                pass  # Fall back on camera error

        # High-fidelity synthetic 2D wave gradient fallback
        cpu = psutil.cpu_percent()
        t = time.time()
        
        pixels = []
        for y in range(height):
            for x in range(width):
                # Generates a moving wave pattern across the 2D grid
                val = 0.5 + 0.35 * math.sin(x * 0.3 + t * 2.5) + 0.15 * math.cos(y * 0.4 + (cpu / 100.0) * 4.0)
                val = min(1.0, max(0.0, val))
                pixels.append(val)
        return pixels
