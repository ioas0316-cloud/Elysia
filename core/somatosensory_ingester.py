"""
Elysia Somatosensory Hardware Ingestion Bridge
================================================
Attempts to capture real-world voice (audio) and vision (webcam) data.
If hardware is unavailable or dependencies are missing, it falls back to 
a continuous, dynamic wave modulated by CPU/RAM tension and high-precision clocks.

[Hardened Version: Process Isolation & Client/Worker Decoupling]
Prevents C-level driver crashes in third-party libraries (like sounddevice/PortAudio)
from terminating the main Elysia daemon.
"""

import sys
import time
import math
import random
import psutil
import os
from typing import List, Union
import numpy as np

# Global variables for dynamically imported hardware libraries
sd = None
cv2 = None
ImageGrab = None


class SomatosensoryIngester:
    def __init__(self, hardware_mode: bool = False):
        self.hardware_mode = hardware_mode
        self.cache_path = r"c:\Elysia\data\somatosensory_cache.npz"
        self.worker_process = None
        self.last_restart_time = 0.0
        
        if self.hardware_mode:
            # Dynamically import hardware driver wrappers inside the isolated worker process
            global sd, cv2, ImageGrab
            try:
                import sounddevice as sd
            except ImportError:
                sd = None
                
            try:
                import cv2
            except ImportError:
                cv2 = None
                
            try:
                from PIL import ImageGrab
            except ImportError:
                ImageGrab = None
                
            self.audio_supported = sd is not None
            self.video_supported = cv2 is not None
            
            if self.audio_supported:
                print("🎙️ [Somatosensory Worker] Real-world Audio stream detected via sounddevice.")
            else:
                print("🎙️ [Somatosensory Worker] sounddevice not found. Activating dynamic synthetic Audio generator.")
                
            if self.video_supported:
                print("📷 [Somatosensory Worker] Real-world Video stream detected via OpenCV.")
            elif ImageGrab is not None:
                print("🖥️ [Somatosensory Worker] PIL.ImageGrab detected. Activating Omniscient Screen Capture.")
            else:
                print("📷 [Somatosensory Worker] OpenCV/PIL not found. Activating dynamic synthetic Video generator.")
        else:
            # Client mode (used by the main daemon): Protect process by not loading driver DLLs
            self.audio_supported = False
            self.video_supported = False
            print("🛡️ [Somatosensory Bypass] Client mode activated. Main process isolated from hardware drivers.")
            self._ensure_worker_alive()

    def _ensure_worker_alive(self):
        if self.hardware_mode:
            return
            
        # Check if the isolated worker subprocess needs to be spawned or restarted
        if self.worker_process is None or self.worker_process.poll() is not None:
            now = time.time()
            # Cooldown of 10 seconds to avoid tight spawn loops if hardware is permanently broken
            if now - self.last_restart_time > 10.0:
                self.last_restart_time = now
                print("🔄 [Somatosensory Bypass] Spawning isolated hardware sensor worker process...")
                try:
                    import subprocess
                    worker_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "somatosensory_worker.py"))
                    self.worker_process = subprocess.Popen(
                        [sys.executable, worker_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"⚠️ [Somatosensory Bypass] Failed to spawn worker process: {e}")

    def capture_audio(self, duration_sec: float = 0.1, sample_rate: int = 8000) -> List[float]:
        """
        Captures audio from the default microphone (in hardware mode) or reads from cache (in client mode).
        Falls back to a dynamic interference wave modulated by CPU/RAM load.
        """
        self._ensure_worker_alive()
        
        if not self.hardware_mode:
            # Client mode: try reading cached somatic wave from isolated worker
            try:
                if os.path.exists(self.cache_path):
                    # Verify cache freshness (< 2.0 seconds) to prevent frozen inputs
                    mtime = os.path.getmtime(self.cache_path)
                    if time.time() - mtime < 2.0:
                        data = np.load(self.cache_path)
                        if "audio" in data:
                            return data["audio"].tolist()
            except Exception:
                pass
            # Cache missing or stale -> fluid fallback to inner thought wave
            return self._generate_synthetic_audio(duration_sec, sample_rate)
            
        # Hardware Mode (executed in the isolated worker process)
        if self.audio_supported and sd is not None:
            try:
                num_frames = int(duration_sec * sample_rate)
                recording = sd.rec(num_frames, samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()
                return recording.flatten().tolist()
            except Exception:
                pass  # Fall back on hardware error
                
        return self._generate_synthetic_audio(duration_sec, sample_rate)

    def _generate_synthetic_audio(self, duration_sec: float, sample_rate: int) -> List[float]:
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

    def capture_video(self) -> np.ndarray:
        """
        Captures screen/camera (in hardware mode) or reads from cache (in client mode).
        Returns the raw C-level memory buffer (numpy array).
        """
        self._ensure_worker_alive()
        
        if not self.hardware_mode:
            # Client mode: try reading cached somatic pixels
            try:
                if os.path.exists(self.cache_path):
                    mtime = os.path.getmtime(self.cache_path)
                    if time.time() - mtime < 2.0:
                        data = np.load(self.cache_path)
                        if "video" in data:
                            return data["video"]
            except Exception:
                pass
            return self._generate_synthetic_video()
            
        # Hardware Mode (executed in the isolated worker process)
        # 1. Screen Capture (Omniscient Observation)
        if ImageGrab is not None:
            try:
                screen = ImageGrab.grab()
                screen_gray = screen.convert('L')
                return np.array(screen_gray) / 255.0
            except Exception:
                pass
                
        # 2. Camera Fallback
        if self.video_supported and cv2 is not None:
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        return np.array(gray) / 255.0
            except Exception:
                pass  # Fall back on camera error

        return self._generate_synthetic_video()

    def _generate_synthetic_video(self) -> np.ndarray:
        # High-fidelity synthetic 2D wave gradient fallback
        cpu = psutil.cpu_percent()
        t = time.time()
        
        width, height = 64, 64
        y, x = np.mgrid[0:height, 0:width]
        val = 0.5 + 0.35 * np.sin(x * 0.3 + t * 2.5) + 0.15 * np.cos(y * 0.4 + (cpu / 100.0) * 4.0)
        val = np.clip(val, 0.0, 1.0)
        return val

    def __del__(self):
        # Gracefully terminate the worker subprocess upon destructor garbage collection
        if hasattr(self, 'worker_process') and self.worker_process is not None:
            try:
                self.worker_process.terminate()
                self.worker_process.wait(timeout=1)
            except:
                pass
