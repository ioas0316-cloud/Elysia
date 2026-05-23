"""
Multimodal Streaming Injector (Pure Frequency Pipeline)
======================================================
"Pure Resonance, No Residuals."

Captures real-time audio and video (screen) streams and injects them
directly into the Elysia Rotor Engine as pure frequency arrays.
"""

import numpy as np
import threading
import time
from typing import Optional, Callable
from Core.System.gateway_interfaces import SensoryChannel

# [HARDWARE ABSTRACTION] Mocking libraries for sandbox environments
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False

try:
    import mss
    import cv2
    HAS_MSS = True
except (ImportError, OSError):
    HAS_MSS = False

class MultimodalStreamingInjector(SensoryChannel):
    """
    Directly taps into the system's sound and monitor pixels.
    Converts raw data into pure frequency/phase vectors for the Rotor Engine.
    """
    def __init__(self, fps: int = 15, sample_rate: int = 44100):
        super().__init__("MultimodalStreamingInjector")
        self.fps = fps
        self.sample_rate = sample_rate
        self.running = False
        self.sct = None
        self.audio_buffer = np.zeros(1024, dtype=np.float32)
        self.video_buffer = None
        self._lock = threading.Lock()
        self.has_mss_internal = HAS_MSS
        self.has_sd_internal = HAS_SOUNDDEVICE

        # Target WoW window if available (logic can be expanded to find HWND)
        self.target_window = None

    def _init_hardware(self):
        """Lazy init hardware to handle headless environments."""
        if self.has_mss_internal:
            try:
                self.sct = mss.mss()
            except Exception as e:
                # print(f"⚠️ [Injector] Display capture failed (Headless?): {e}")
                self.has_mss_internal = False

    def _capture_loop(self):
        """Main loop for captures - no disk I/O, only RAM."""
        frame_time = 1.0 / self.fps
        self._init_hardware()

        # Audio stream setup (Mock if hardware missing)
        if self.has_sd_internal:
            try:
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._audio_callback
                )
                stream.start()
            except Exception as e:
                # print(f"⚠️ [Injector] Audio capture failed: {e}")
                self.has_sd_internal = False

        while self.running:
            start_tick = time.time()

            # 1. Capture Screen (Video)
            if self.has_mss_internal and self.sct:
                try:
                    monitor = self.sct.monitors[1]
                    screenshot = self.sct.grab(monitor)
                    img = np.array(screenshot)
                    img_small = cv2.resize(img, (64, 64))
                    with self._lock:
                        self.video_buffer = img_small
                except Exception:
                    pass
            else:
                # Mock video if no display
                t = time.time()
                mock_pattern = np.zeros((64, 64, 3), dtype=np.uint8)
                mock_pattern[:, :, 0] = int(127 + 127 * np.sin(t))
                with self._lock:
                    self.video_buffer = mock_pattern

            # 2. Mock Audio if no hardware
            if not self.has_sd_internal:
                t = time.time()
                # Simulate a resonant frequency for testing (~727Hz)
                t_vals = np.linspace(t, t + 0.02, 1024)
                with self._lock:
                    self.audio_buffer = np.sin(2 * np.pi * 727 * t_vals).astype(np.float32)

            # 3. Push to callback if registered
            if self.callback:
                with self._lock:
                    packet = {
                        "audio_freq": self.audio_buffer.copy(),
                        "video_pixels": self.video_buffer.copy(),
                        "timestamp": time.time()
                    }
                self.callback(packet)

            # Sleep to maintain FPS
            elapsed = time.time() - start_tick
            wait = max(0, frame_time - elapsed)
            time.sleep(wait)

        if self.has_sd_internal and 'stream' in locals():
            stream.stop()

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal audio stream callback."""
        if status:
            pass
        with self._lock:
            self.audio_buffer = indata.flatten()

    def start(self):
        if self.running: return
        self.running = True
        print("⚡ [Injector] Multimodal Streaming Injector starting...")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        print("🔇 [Injector] Multimodal Streaming Injector stopped.")

if __name__ == "__main__":
    # Test stub
    injector = MultimodalStreamingInjector()
    injector.register_callback(lambda p: print(f"Resonating at {p['timestamp']}: Audio size {len(p['audio_freq'])}"))
    injector.start()
    time.sleep(1)
    injector.stop()
